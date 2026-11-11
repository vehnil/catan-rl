"""SQLite store for policies, runs, matches and ratings (stdlib ``sqlite3`` only)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from catan.sim.runner import GameRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- agent registry name (greedy, ppo, ...)
    path        TEXT NOT NULL,          -- policy directory on disk
    parent_id   TEXT,                   -- policy this one was trained from
    run_id      TEXT,
    manifest    TEXT NOT NULL,          -- JSON copy of manifest.json
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,          -- tournament | train | sim
    name        TEXT,
    config      TEXT NOT NULL,          -- JSON
    status      TEXT NOT NULL DEFAULT 'running',
    metrics     TEXT,                   -- JSON, filled in at the end
    created_at  REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    seed        INTEGER NOT NULL,
    rules       TEXT NOT NULL,          -- JSON RulesConfig
    winner_seat INTEGER NOT NULL,       -- -1 if truncated
    turns       INTEGER NOT NULL,
    steps       INTEGER NOT NULL,
    truncated   INTEGER NOT NULL,
    duration_s  REAL NOT NULL,
    replay_path TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS match_players (
    match_id    INTEGER NOT NULL REFERENCES matches(id),
    seat        INTEGER NOT NULL,
    agent       TEXT NOT NULL,          -- agent spec, e.g. "greedy" or "policy:abc123"
    vp          INTEGER NOT NULL,
    rank        INTEGER NOT NULL,
    is_winner   INTEGER NOT NULL,
    PRIMARY KEY (match_id, seat)
);
CREATE TABLE IF NOT EXISTS ratings (
    agent       TEXT PRIMARY KEY,
    mu          REAL NOT NULL,
    sigma       REAL NOT NULL,
    games       INTEGER NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_match_players_agent ON match_players(agent);
CREATE INDEX IF NOT EXISTS idx_matches_run ON matches(run_id);
"""


class Store:
    def __init__(self, path: str | Path = "data/catan.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- runs

    def start_run(self, kind: str, config: dict[str, Any], name: str | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO runs (id, kind, name, config, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, kind, name, json.dumps(config, default=str), time.time()),
        )
        self.conn.commit()
        return run_id

    def finish_run(self, run_id: str, metrics: dict[str, Any], status: str = "done") -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, metrics = ?, finished_at = ? WHERE id = ?",
            (status, json.dumps(metrics, default=str), time.time(), run_id),
        )
        self.conn.commit()

    def runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- matches

    def add_match(
        self, rec: GameRecord, run_id: str | None = None, replay_path: str | None = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO matches (run_id, seed, rules, winner_seat, turns, steps, truncated,"
            " duration_s, replay_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                rec.seed,
                json.dumps(rec.rules),
                rec.winner,
                rec.turns,
                rec.steps,
                int(rec.truncated),
                rec.duration_s,
                replay_path,
                time.time(),
            ),
        )
        match_id = int(cur.lastrowid or 0)
        self.conn.executemany(
            "INSERT INTO match_players (match_id, seat, agent, vp, rank, is_winner) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (match_id, seat, spec, rec.vp[seat], rec.ranks[seat], int(seat == rec.winner))
                for seat, spec in enumerate(rec.agents)
            ],
        )
        self.conn.commit()
        return match_id

    def agent_stats(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Per-agent games, wins, win rate, mean VP and mean rank."""
        where, args = ("WHERE m.run_id = ?", (run_id,)) if run_id else ("", ())
        rows = self.conn.execute(
            f"""SELECT mp.agent AS agent, COUNT(*) AS games, SUM(mp.is_winner) AS wins,
                       AVG(mp.is_winner) AS win_rate, AVG(mp.vp) AS mean_vp,
                       AVG(mp.rank) AS mean_rank
                FROM match_players mp JOIN matches m ON m.id = mp.match_id {where}
                GROUP BY mp.agent ORDER BY win_rate DESC""",
            args,
        ).fetchall()
        return [dict(r) for r in rows]

    def matches(self, run_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        where, args = ("WHERE run_id = ?", (run_id, limit)) if run_id else ("", (limit,))
        rows = self.conn.execute(
            f"SELECT * FROM matches {where} ORDER BY id DESC LIMIT ?", args
        ).fetchall()
        return [dict(r) for r in rows]

    def replays(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Matches that have a replay file, with agent specs in seat order (newest first)."""
        rows = self.conn.execute(
            """SELECT m.id, m.run_id, m.seed, m.winner_seat, m.turns, m.truncated, m.replay_path,
                      (SELECT GROUP_CONCAT(agent, ';') FROM
                          (SELECT agent FROM match_players WHERE match_id = m.id ORDER BY seat))
                      AS agents
               FROM matches m WHERE m.replay_path IS NOT NULL ORDER BY m.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [{**dict(r), "agents": (r["agents"] or "").split(";")} for r in rows]

    # ---------------------------------------------------------------- policies

    def add_policy(self, manifest: dict[str, Any], path: str | Path) -> str:
        self.conn.execute(
            "INSERT OR REPLACE INTO policies (id, name, kind, path, parent_id, run_id, manifest,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                manifest["id"],
                manifest.get("name", manifest["id"]),
                manifest["kind"],
                str(path),
                manifest.get("parent_id"),
                manifest.get("run_id"),
                json.dumps(manifest, default=str),
                manifest.get("created_at", time.time()),
            ),
        )
        self.conn.commit()
        return str(manifest["id"])

    def policy(self, ref: str) -> dict[str, Any] | None:
        """By id or exact name; otherwise the newest policy from training config ``ref``.

        Trained policies are named ``<config name>-<run id>``, so a curriculum stage
        can warm-start from "whatever the previous stage produced last".
        """
        row = self.conn.execute(
            "SELECT * FROM policies WHERE id = ? OR name = ? ORDER BY created_at DESC LIMIT 1",
            (ref, ref),
        ).fetchone()
        if row is None:
            row = self.conn.execute(
                "SELECT * FROM policies WHERE substr(name, 1, ?) = ? ORDER BY created_at DESC"
                " LIMIT 1",
                (len(ref) + 1, ref + "-"),
            ).fetchone()
        return dict(row) if row else None

    def policies(self, kind: str | None = None) -> list[dict[str, Any]]:
        q, args = ("WHERE kind = ?", (kind,)) if kind else ("", ())
        rows = self.conn.execute(
            f"SELECT id, name, kind, path, parent_id, run_id, created_at FROM policies {q}"
            " ORDER BY created_at DESC",
            args,
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- ratings

    def get_ratings(self, agents: Iterable[str] | None = None) -> dict[str, dict[str, float]]:
        rows = self.conn.execute("SELECT * FROM ratings").fetchall()
        out = {r["agent"]: {"mu": r["mu"], "sigma": r["sigma"], "games": r["games"]} for r in rows}
        if agents is not None:
            wanted = set(agents)
            out = {k: v for k, v in out.items() if k in wanted}
        return out

    def set_ratings(self, ratings: dict[str, dict[str, float]]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT OR REPLACE INTO ratings (agent, mu, sigma, games, updated_at) VALUES (?, ?, ?, ?, ?)",
            [(a, r["mu"], r["sigma"], int(r["games"]), now) for a, r in ratings.items()],
        )
        self.conn.commit()

    def leaderboard(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT agent, mu, sigma, mu - 3 * sigma AS conservative, games FROM ratings"
            " ORDER BY conservative DESC"
        ).fetchall()
        return [dict(r) for r in rows]
