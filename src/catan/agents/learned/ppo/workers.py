"""Parallel rollout collection: one env per worker process, each with a copy of the policy.

The engine is pure Python, so collection is the bottleneck and threads don't
help. Each worker runs the learner's network itself (one thread, batch of
one), so the main process only ships weights out and transitions back; it is
not in the per-step loop.

Workers collect in chunks of ``CHUNK`` steps and the main process hands out
chunks until the rollout is full, so fast workers (games against random,
performance cores) are never held up by slow ones (games against greedy,
the M1's efficiency cores). A rollout therefore holds slightly more than
``rollout_steps`` (less than one chunk extra). Episodes carry over between
rollouts.

Workers each keep their own copy of the league; snapshots are broadcast as
state dicts. Evaluation games are spread over the workers too.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from catan.agents.learned.ppo.agent import PPOAgent
from catan.agents.learned.ppo.league import League
from catan.agents.learned.ppo.model import ActorCritic
from catan.agents.registry import make_agent
from catan.config.models import LeagueConfig
from catan.engine.actions import NUM_ACTIONS
from catan.engine.state import RulesConfig
from catan.env.gym_single import CatanSingleAgentEnv
from catan.player.features import FEATURE_DIM
from catan.sim.runner import play_game

CHUNK = 64


def eval_games(
    model: ActorCritic,
    hidden: list[int],
    opponents: list[str],
    games: list[int],
    rules: RulesConfig,
    seed: int,
    data_dir: str | Path = "data",
) -> list[tuple[int, bool, int, int]]:
    """Play the given game numbers; the learner sits in seat ``g % 4``.

    Returns (g, won, vp, turns) per game.
    """
    out = []
    for g in games:
        seat = g % 4
        lineup = [make_agent(s, data_dir) for s in opponents]
        lineup.insert(seat, PPOAgent(hidden, model=model))
        rec = play_game(lineup, seed + g, rules)
        out.append((g, rec.winner == seat, rec.vp[seat], rec.turns))
    return out


class _Actor:
    """One env plus a local policy copy; lives in a worker process."""

    def __init__(
        self,
        rules: RulesConfig,
        env_kwargs: dict[str, Any],
        gamma: float,
        league_cfg: LeagueConfig,
        hidden: list[int],
        seed: int,
        data_dir: str | Path,
    ) -> None:
        torch.manual_seed(seed)
        self.gamma = gamma
        self.hidden = hidden
        self.data_dir = data_dir
        self.league = League(league_cfg, hidden, np.random.default_rng(seed), data_dir)
        self.env = CatanSingleAgentEnv(
            opponents=self.league.sample_opponents, config=rules, **env_kwargs
        )
        self.model = ActorCritic(hidden).eval()
        self.obs, self.info = self.env.reset(seed=seed)
        self.ep_return = 0.0

    def collect(self, steps: int) -> dict[str, Any]:
        seg: dict[str, Any] = {
            "obs": np.zeros((steps, FEATURE_DIM), np.float32),
            "masks": np.zeros((steps, NUM_ACTIONS), bool),
            "actions": np.zeros(steps, np.int64),
            "logprobs": np.zeros(steps, np.float32),
            "values": np.zeros(steps, np.float32),
            "rewards": np.zeros(steps, np.float32),
            "dones": np.zeros(steps, np.float32),
        }
        episodes: list[tuple[bool, float, int]] = []  # (won, return, turns)
        for t in range(steps):
            mask = self.info["action_mask"]
            a, logp, v = self.model.act(
                torch.as_tensor(self.obs).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0)
            )
            next_obs, r, term, trunc, next_info = self.env.step(int(a))
            if trunc:  # bootstrap through the turn cap instead of treating it as a loss
                with torch.no_grad():
                    r += self.gamma * self.model(torch.as_tensor(next_obs).unsqueeze(0))[1].item()
            seg["obs"][t], seg["masks"][t], seg["actions"][t] = self.obs, mask, int(a)
            seg["logprobs"][t], seg["values"][t] = logp.item(), v.item()
            seg["rewards"][t], seg["dones"][t] = r, term or trunc
            self.ep_return += r
            if term or trunc:
                won = next_info.get("winner") == next_info["seat"]
                episodes.append((won, self.ep_return, next_info["turns"]))
                self.ep_return = 0.0
                self.obs, self.info = self.env.reset()
            else:
                self.obs, self.info = next_obs, next_info
        with torch.no_grad():
            seg["last_value"] = self.model(torch.as_tensor(self.obs).unsqueeze(0))[1].item()
        seg["episodes"] = episodes
        return seg


def _worker(conn: Connection, *actor_args: Any) -> None:
    torch.set_num_threads(1)  # one process per core; don't oversubscribe
    actor = _Actor(*actor_args)
    while True:
        cmd, *args = conn.recv()
        try:
            if cmd == "weights":
                actor.model.load_state_dict(args[0])
                conn.send(("ok", None))
            elif cmd == "collect":
                conn.send(("ok", actor.collect(args[0])))
            elif cmd == "snapshot":
                model = ActorCritic(actor.hidden)
                model.load_state_dict(args[0])
                actor.league.add_snapshot(model)
                conn.send(("ok", None))
            elif cmd == "eval":
                state, hidden, opponents, games, rules, seed = args
                model = ActorCritic(hidden)
                model.load_state_dict(state)
                model.eval()
                result = eval_games(model, hidden, opponents, games, rules, seed, actor.data_dir)
                conn.send(("ok", result))
            elif cmd == "close":
                conn.close()
                return
        except Exception:
            conn.send(("error", traceback.format_exc()))


def _state(model: ActorCritic) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class RolloutWorkers:
    """``n`` worker processes (spawned, so this works on macOS)."""

    def __init__(
        self,
        n: int,
        rules: RulesConfig,
        env_kwargs: dict[str, Any],
        gamma: float,
        league_cfg: LeagueConfig,
        hidden: list[int],
        seed: int,
        data_dir: str | Path = "data",
    ) -> None:
        self.n = n
        self.max_snapshots = league_cfg.max_snapshots
        self.snapshots = 0
        ctx = mp.get_context("spawn")
        self.conns: list[Connection] = []
        self.procs = []
        for i in range(n):
            parent, child = ctx.Pipe()
            args = (rules, env_kwargs, gamma, league_cfg, hidden, seed * 1000 + i, data_dir)
            proc = ctx.Process(target=_worker, args=(child, *args), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    @staticmethod
    def _unwrap(conn: Connection) -> Any:
        status, payload = conn.recv()
        if status == "error":
            raise RuntimeError(f"rollout worker failed:\n{payload}")
        return payload

    def _broadcast(self, *msg: Any) -> list[Any]:
        for c in self.conns:
            c.send(msg)
        return [self._unwrap(c) for c in self.conns]

    def collect(self, model: ActorCritic, steps: int) -> list[tuple[int, dict[str, Any]]]:
        """At least ``steps`` transitions as (worker, segment), each worker's in order."""
        self._broadcast("weights", _state(model))
        index = {c: i for i, c in enumerate(self.conns)}
        for c in self.conns:
            c.send(("collect", CHUNK))
        busy, received, out = set(self.conns), 0, []
        while busy:
            for ready in wait(list(busy)):
                conn = cast(Connection, ready)
                seg = self._unwrap(conn)
                out.append((index[conn], seg))
                received += len(seg["actions"])
                # Hand out another chunk only while the ones in flight can't fill the rollout.
                if received + CHUNK * (len(busy) - 1) < steps:
                    conn.send(("collect", CHUNK))
                else:
                    busy.discard(conn)
        return out

    def add_snapshot(self, model: ActorCritic) -> None:
        self._broadcast("snapshot", _state(model))
        self.snapshots = min(self.snapshots + 1, self.max_snapshots)

    def evaluate(
        self,
        model: ActorCritic,
        hidden: list[int],
        opponents: list[str],
        games: int,
        rules: RulesConfig,
        seed: int,
    ) -> list[tuple[int, bool, int, int]]:
        state = _state(model)
        for i, c in enumerate(self.conns):
            c.send(("eval", state, hidden, opponents, list(range(i, games, self.n)), rules, seed))
        return sorted(r for c in self.conns for r in self._unwrap(c))

    def close(self) -> None:
        for c in self.conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                c.send(("close",))
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
