"""Strategy selection as a multi-armed bandit across games.

Arms are greedy strategy presets (``heuristics.PRESETS``). At the start of each
game the agent picks a preset (UCB1 or Thompson), plays the whole game with
it, and updates that arm with the result (1 = win, otherwise a small
rank-based credit).

Statistics persist across games through ``stats_path`` (a JSON file), so the
bandit keeps learning across a tournament, even though each game builds a
fresh agent. Writes are atomic but not merged, so use ``workers=1`` when a
meta-bandit's learning curve matters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from catan.agents.bandits.thompson import BetaThompson
from catan.agents.bandits.ucb import UCB1
from catan.agents.baselines.greedy import GreedyAgent
from catan.agents.heuristics import PRESETS
from catan.engine import events as ev


class MetaBanditAgent(GreedyAgent):
    name = "meta_bandit"

    def __init__(
        self,
        arms: list[str] | str | None = None,
        algo: str = "thompson",
        c: float = 1.0,
        stats_path: str | None = None,
        **greedy_kwargs: Any,
    ) -> None:
        super().__init__(**greedy_kwargs)
        if isinstance(arms, str):
            arms = arms.split("|")
        self.arms = list(arms or PRESETS)
        self.algo = algo
        self.c = c
        self.stats_path = stats_path
        self.bandit = self._new_bandit()
        if stats_path and Path(stats_path).exists():
            self._load_stats(json.loads(Path(stats_path).read_text()))
        self.arm = 0

    def _new_bandit(self) -> UCB1 | BetaThompson:
        return UCB1(len(self.arms), self.c) if self.algo == "ucb" else BetaThompson(len(self.arms))

    def _load_stats(self, d: dict[str, Any]) -> None:
        if d.get("arms") == self.arms and d.get("algo") == self.algo:
            cls = UCB1 if self.algo == "ucb" else BetaThompson
            self.bandit = cls.from_dict(d["bandit"])

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "arms": self.arms,
            "algo": self.algo,
            "c": self.c,
            "stats_path": self.stats_path,
        }

    def reset(self, seat: int, rng: np.random.Generator) -> None:
        super().reset(seat, rng)
        if self.stats_path and Path(self.stats_path).exists():
            self._load_stats(json.loads(Path(self.stats_path).read_text()))
        self.arm = self.bandit.select(rng)
        self.w = PRESETS[self.arms[self.arm]]

    def observe(self, event_view: dict[str, Any]) -> None:
        super().observe(event_view)
        if event_view["kind"] == ev.GAME_OVER:
            winner = event_view.get("winner", -1)
            self.bandit.update(self.arm, 1.0 if winner == self.seat else 0.0)
            self._persist()

    def _persist(self) -> None:
        if not self.stats_path:
            return
        path = Path(self.stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.stats()))
        os.replace(tmp, path)

    def stats(self) -> dict[str, Any]:
        return {
            "arms": self.arms,
            "algo": self.algo,
            "bandit": self.bandit.to_dict(),
            "means": dict(zip(self.arms, self.bandit.means().round(3).tolist(), strict=True)),
        }

    def save_files(self, directory: Path) -> dict[str, str]:
        (directory / "state.json").write_text(json.dumps(self.stats(), indent=2))
        return {"state": "state.json"}

    def load_files(self, directory: Path, manifest: dict[str, Any]) -> None:
        f = directory / "state.json"
        if f.exists():
            self._load_stats(json.loads(f.read_text()))
