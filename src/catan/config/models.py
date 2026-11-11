"""Typed experiment configs, loaded from YAML. Unknown keys are rejected (catches typos)."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catan.engine.state import RulesConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RulesModel(_Strict):
    vp_to_win: int = 10
    discard_limit: int = 7
    board: Literal["standard", "random"] = "random"
    avoid_adjacent_red: bool = True
    max_trade_proposals_per_turn: int = Field(3, ge=0)
    max_actions_per_turn: int = Field(50, ge=1)
    max_turns_per_game: int = Field(500, ge=1)

    def to_rules(self) -> RulesConfig:
        return RulesConfig(**self.model_dump())


assert {f.name for f in fields(RulesConfig)} == set(RulesModel.model_fields), "keep in sync"


class TournamentConfig(_Strict):
    """``format``:

    * ``fixed``: the same four ``agents`` every game, seats rotated.
    * ``round_robin``: every 4-agent lineup drawn from ``agents`` (with
      repetition if fewer than four), seats rotated.
    * ``gauntlet``: ``candidate`` in every game, opponents sampled from ``agents``.
    """

    name: str = "tournament"
    format: Literal["fixed", "round_robin", "gauntlet"] = "round_robin"
    agents: list[str]
    candidate: str | None = None
    games: int = Field(100, ge=1)
    seed: int = 0
    workers: int = Field(1, ge=1)
    rules: RulesModel = RulesModel()
    save_replays: bool = True
    update_ratings: bool = True
    data_dir: str = "data"

    @model_validator(mode="after")
    def _check(self) -> TournamentConfig:
        if self.format == "fixed" and len(self.agents) != 4:
            raise ValueError("fixed format needs exactly 4 agents")
        if self.format == "gauntlet" and not self.candidate:
            raise ValueError("gauntlet format needs a candidate")
        return self


class PPOConfig(_Strict):
    total_steps: int = Field(200_000, ge=1)
    rollout_steps: int = Field(2048, ge=64)
    epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: list[int] = [512, 256]
    vp_reward_scale: float = 0.05
    # Bonus for winning sooner, linear from the full amount at turn 0 to 0 at fast_win_turns.
    fast_win_bonus: float = 0.0
    fast_win_turns: int = Field(200, ge=1)
    device: str = "cpu"
    num_envs: int = Field(1, ge=1)  # >1: one env + policy copy per worker process


class LeagueConfig(_Strict):
    """Opponent sampling for self-play. Fixed specs + snapshots of the learner."""

    fixed_opponents: list[str] = ["random", "greedy"]
    snapshot_every: int = Field(20_000, ge=1)  # env steps between snapshots
    max_snapshots: int = 5
    self_play_prob: float = Field(0.5, ge=0, le=1)  # P(an opponent seat is a past snapshot)
    # True: fixed_opponents is the exact 3-seat lineup (shuffled each game) instead of a pool.
    lineup: bool = False

    @model_validator(mode="after")
    def _check_lineup(self) -> LeagueConfig:
        if self.lineup and len(self.fixed_opponents) != 3:
            raise ValueError("lineup: true needs exactly 3 fixed_opponents")
        return self


class TrainConfig(_Strict):
    name: str = "ppo"
    seed: int = 0
    rules: RulesModel = RulesModel()
    ppo: PPOConfig = PPOConfig()
    league: LeagueConfig = LeagueConfig()
    init_policy: str | None = None  # policy ref to warm-start from
    eval_every: int = Field(50_000, ge=1)
    eval_games: int = Field(40, ge=1)
    eval_opponents: list[str] = ["random", "random", "random"]
    # Save the policy every N steps as "<name>@<steps>-<run id>" (see `catan curve`).
    checkpoint_every: int | None = Field(None, ge=1)
    data_dir: str = "data"


def load_config[M: BaseModel](path: str | Path, model: type[M]) -> M:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return model.model_validate(raw)
