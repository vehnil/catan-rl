"""Policy checkpoints on disk.

Layout::

    data/policies/<id>/
        manifest.json         # always: kind, constructor config, lineage, metrics
        weights.safetensors   # neural agents only
        state.json            # optional extra agent state (e.g. bandit statistics)

Non-neural agents are fully described by ``manifest.json`` (plain text).
Neural weights use safetensors; ``export_policy`` can also write ONNX (to run
outside Python) or JSON weights (debugging small nets only).
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from catan.player.base import Agent

FORMAT_VERSION = 1


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_policy(
    agent: Agent,
    data_dir: str | Path = "data",
    name: str | None = None,
    run_id: str | None = None,
    parent_id: str | None = None,
    metrics: dict[str, Any] | None = None,
    store: Any = None,
) -> tuple[dict[str, Any], Path]:
    """Write ``agent`` to ``data_dir/policies/<id>`` and optionally register it in ``store``."""
    pid = uuid.uuid4().hex[:10]
    path = Path(data_dir) / "policies" / pid
    path.mkdir(parents=True, exist_ok=False)
    files = agent.save_files(path)
    manifest = {
        "format_version": FORMAT_VERSION,
        "id": pid,
        "name": name or f"{agent.name}-{pid}",
        "kind": agent.name,
        "config": agent.config(),
        "files": files,
        "parent_id": parent_id,
        "run_id": run_id,
        "metrics": metrics or {},
        "git_sha": _git_sha(),
        "created_at": time.time(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if store is not None:
        store.add_policy(manifest, path)
    return manifest, path


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "manifest.json").read_text())


def load_policy(path: str | Path) -> Agent:
    from catan.agents.registry import agent_class

    path = Path(path)
    manifest = read_manifest(path)
    agent = agent_class(manifest["kind"])(**manifest["config"])
    agent.load_files(path, manifest)
    agent.spec = f"policy:{manifest['id']}"  # type: ignore[attr-defined]
    return agent


def resolve_policy_path(ref: str, data_dir: str | Path = "data") -> Path:
    """Accept a directory path, a policy id, or a policy name registered in the store."""
    p = Path(ref)
    if (p / "manifest.json").exists():
        return p
    direct = Path(data_dir) / "policies" / ref
    if (direct / "manifest.json").exists():
        return direct
    db = Path(data_dir) / "catan.db"
    if db.exists():
        from catan.sim.store import Store

        with Store(db) as store:
            row = store.policy(ref)
        if row:
            return Path(row["path"])
    raise FileNotFoundError(f"no policy {ref!r} (looked in {direct} and {db})")


def export_policy(path: str | Path, fmt: str, out: str | Path | None = None) -> Path:
    """Export a stored policy. ``fmt``: "json" (manifest + weights as lists) or "onnx"."""
    path = Path(path)
    manifest = read_manifest(path)
    agent = load_policy(path)
    if fmt == "json":
        payload: dict[str, Any] = {"manifest": manifest}
        weights = path / "weights.safetensors"
        if weights.exists():
            from safetensors.numpy import load_file

            payload["weights"] = {k: v.tolist() for k, v in load_file(str(weights)).items()}
        dest = Path(out or path / "export.json")
        dest.write_text(json.dumps(payload))
        return dest
    if fmt == "onnx":
        exporter = getattr(agent, "export_onnx", None)
        if exporter is None:
            raise ValueError(f"{manifest['kind']} policies have no neural network to export")
        dest = Path(out or path / "policy.onnx")
        exporter(dest)
        return dest
    raise ValueError(f"unknown export format {fmt!r}")
