"""Local viewer server (stdlib only).

Routes:
  GET /                       the viewer page
  GET /api/replays            stored replays (from the DB, or by scanning data/replays/)
  GET /api/timeline?path=...  ``build_timeline`` for one replay; path is relative to data_dir
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from catan.sim.replay import load_replay
from catan.sim.store import Store
from catan.sim.timeline import build_timeline


@lru_cache(maxsize=16)
def _timeline_bytes(path: str, mtime: float) -> bytes:
    del mtime  # part of the cache key only
    return json.dumps(build_timeline(load_replay(path)), separators=(",", ":")).encode()


def _relative(path: str | Path, root: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def list_replays(data_dir: Path) -> list[dict[str, Any]]:
    root = data_dir.resolve()
    out: list[dict[str, Any]] = []
    db = data_dir / "catan.db"
    if db.exists():
        with Store(db) as store:
            for r in store.replays():
                rel = _relative(r["replay_path"], root)
                if rel and (root / rel).exists():
                    out.append({**r, "path": rel})
    if not out:  # no DB (or it points elsewhere): scan the replay folder
        for p in sorted((root / "replays").glob("*/*.json.gz"), reverse=True)[:1000]:
            rec = load_replay(p)
            out.append(
                {
                    "path": p.relative_to(root).as_posix(),
                    "run_id": p.parent.name,
                    "seed": rec.seed,
                    "winner_seat": rec.winner,
                    "turns": rec.turns,
                    "truncated": rec.truncated,
                    "agents": rec.agents,
                }
            )
    return out


def make_server(data_dir: str | Path = "data", port: int = 8765) -> ThreadingHTTPServer:
    root = Path(data_dir).resolve()
    page = resources.files("catan.viewer").joinpath("static/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, msg: str) -> None:
            self._send(json.dumps({"error": msg}).encode(), "application/json", status)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif url.path == "/api/replays":
                body = json.dumps(list_replays(root)).encode()
                self._send(body, "application/json")
            elif url.path == "/api/timeline":
                rel = parse_qs(url.query).get("path", [""])[0]
                p = (root / rel).resolve()
                if not rel or not p.is_relative_to(root):
                    self._error(HTTPStatus.FORBIDDEN, "path must be inside the data directory")
                elif not p.is_file():
                    self._error(HTTPStatus.NOT_FOUND, f"no replay at {rel}")
                else:
                    self._send(_timeline_bytes(str(p), p.stat().st_mtime), "application/json")
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(
    data_dir: str | Path = "data",
    replay: str | Path | None = None,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    server = make_server(data_dir, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    if replay is not None:
        rel = _relative(replay, Path(data_dir).resolve())
        if rel is None:
            raise ValueError(f"{replay} is not inside the data directory {data_dir}")
        url += f"?replay={quote(rel)}"
    print(f"Catan viewer at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
