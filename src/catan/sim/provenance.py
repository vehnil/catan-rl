"""What produced a run: code version, uncommitted changes, config hash, package versions.

Written next to each training run (``runs/<id>/provenance.json``) so a result can
be traced to exact code and settings. The experiment runner also refuses to start
from a working tree with uncommitted changes unless told otherwise.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = ["catan", "torch", "numpy", "gymnasium", "openskill", "pydantic"]


def _git(args: list[str], cwd: str | Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(cwd: str | Path = ".") -> dict[str, Any]:
    """Commit, branch and uncommitted changes (tracked edits and untracked files)."""
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None, "changes": []}
    changes = (_git(["status", "--porcelain"], cwd) or "").splitlines()
    return {
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": bool(changes),
        "changes": changes,
    }


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def provenance(config: dict[str, Any], cwd: str | Path = ".") -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "created_at": time.time(),
        "git": git_state(cwd),
        "config_hash": config_hash(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
