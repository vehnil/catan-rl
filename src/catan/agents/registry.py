"""Build agents from spec strings, so configs and the CLI can name any agent.

Spec grammar::

    name                      e.g. "greedy"
    name:key=value,key=value  e.g. "ismcts:iterations=300,determinizations=4"
    policy:<id-or-path>       a stored policy (see ``catan.sim.artifacts``)
    policy:<ref>#argmax       the same policy, always playing its most likely action

Values are parsed as int, float, bool (true/false) or left as strings.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from catan.player.base import Agent

# name -> "module:attribute" (imported lazily so torch etc. load only when needed)
_REGISTRY: dict[str, str] = {
    "random": "catan.agents.baselines.random:RandomAgent",
    "greedy": "catan.agents.baselines.greedy:GreedyAgent",
    "opening_bandit": "catan.agents.bandits.opening_bandit:OpeningBanditAgent",
    "meta_bandit": "catan.agents.bandits.meta_bandit:MetaBanditAgent",
    "maxn": "catan.agents.search.maxn:MaxNAgent",
    "paranoid": "catan.agents.search.paranoid:ParanoidAgent",
    "ismcts": "catan.agents.search.ismcts:ISMCTSAgent",
    "ppo": "catan.agents.learned.ppo.agent:PPOAgent",
}

_custom: dict[str, Callable[..., Agent]] = {}


def register(name: str, factory: Callable[..., Agent]) -> None:
    """Register a custom agent factory at runtime (e.g. from a notebook)."""
    _custom[name] = factory


def available() -> list[str]:
    return sorted({*_REGISTRY, *_custom, "policy"})


def _parse_value(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    name, _, rest = spec.partition(":")
    if name == "policy":
        return name, {"ref": rest}
    kwargs: dict[str, Any] = {}
    if rest:
        for part in rest.split(","):
            k, _, v = part.partition("=")
            kwargs[k.strip()] = _parse_value(v.strip())
    return name, kwargs


def agent_class(name: str) -> Callable[..., Agent]:
    if name in _custom:
        return _custom[name]
    if name not in _REGISTRY:
        raise KeyError(f"unknown agent {name!r}; available: {available()}")
    module, attr = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), attr)


def make_agent(spec: str, data_dir: str | Path = "data") -> Agent:
    name, kwargs = parse_spec(spec)
    if name == "policy":
        from catan.sim.artifacts import load_policy, resolve_policy_path

        ref, _, mode = kwargs["ref"].partition("#")
        policy = load_policy(resolve_policy_path(ref, data_dir))
        if mode == "argmax" and hasattr(policy, "deterministic"):
            policy.deterministic = True
        elif mode:
            raise ValueError(f"unknown policy mode {mode!r} in {spec!r} (only #argmax)")
        return policy
    agent = agent_class(name)(**kwargs)
    agent.spec = spec  # type: ignore[attr-defined]
    return agent
