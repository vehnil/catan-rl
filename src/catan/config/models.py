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


class ReferenceConfig(_Strict):
    """A fixed set of opponents that candidates are rated against (``catan rate``)."""

    name: str = "reference"
    agents: list[str]
    calibration_games: int = Field(1400, ge=1)  # round-robin among the references, once
    gauntlet_games: int = Field(400, ge=1)  # per candidate
    seed: int = 9000
    rules: RulesModel = RulesModel()


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
    # Stop an update's epochs early once their mean KL from the rollout policy exceeds this.
    target_kl: float | None = Field(None, gt=0)
    lr_schedule: Literal["constant", "linear"] = "constant"  # linear: decay to 0 at total_steps
    hidden: list[int] = [512, 256]
    vp_reward_scale: float = 0.05
    # Bonus for winning sooner, linear from the full amount at turn 0 to 0 at fast_win_turns.
    fast_win_bonus: float = 0.0
    fast_win_turns: int = Field(200, ge=1)
    device: str = "cpu"
    num_envs: int = Field(1, ge=1)  # >1: one env + policy copy per worker process
    # >1: each worker runs this many games at once and batches their network decisions
    # (the learner's, and opponents' that share a model). Needs num_envs > 1.
    games_per_worker: int = Field(1, ge=1)
    # True: every worker collects a fixed share of each rollout and the shares are merged in
    # worker order, so a parallel run doesn't depend on timing (same config + seed + code ->
    # same weights). Fast workers then wait for slow ones. False: hand out chunks as workers
    # finish (faster; reproducible only up to chunk arrival order).
    deterministic_collection: bool = False

    @model_validator(mode="after")
    def _check_games(self) -> PPOConfig:
        if self.games_per_worker > 1 and self.num_envs < 2:
            raise ValueError("games_per_worker > 1 needs num_envs > 1 (worker processes)")
        return self


class LeagueConfig(_Strict):
    """Opponent sampling for self-play. Fixed specs + snapshots of the learner."""

    fixed_opponents: list[str] = ["random", "greedy"]
    snapshot_every: int = Field(20_000, ge=1)  # env steps between snapshots
    max_snapshots: int = 5
    self_play_prob: float = Field(0.5, ge=0, le=1)  # P(an opponent seat is a past snapshot)
    # P(an opponent seat is the current policy), drawn before snapshots and fixed opponents.
    latest_prob: float = Field(0.0, ge=0, le=1)
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


class CurveSpec(_Strict):
    """Checkpoints (one per ``every`` steps, plus start and final) against fixed lineups."""

    every: int = Field(2_000_000, ge=1)
    games: int = Field(200, ge=1)
    seed: int = 5000
    lineups: dict[str, list[str]] = {
        "greedy3": ["greedy", "greedy", "greedy"],
        "mix": ["random", "random", "greedy"],
    }


class RateSpec(_Strict):
    """Reference ratings of the same checkpoints (``catan rate``)."""

    reference: str = "experiments/reference.yaml"
    every: int = Field(2_000_000, ge=1)
    games: int | None = None  # default: the reference set's gauntlet_games


class CompareSpec(_Strict):
    """Groups of policies on the same games. ``self`` is this experiment's final policies;
    other groups list policy refs (ids, names, or training config names)."""

    groups: dict[str, Literal["self"] | list[str]]
    lineups: dict[str, list[str]] = {
        "greedy3": ["greedy", "greedy", "greedy"],
        "mix": ["random", "random", "greedy"],
    }
    games: int = Field(400, ge=1)
    seed: int = 6000


class AnalysisSpec(_Strict):
    health: bool = True
    curve: CurveSpec | None = CurveSpec()
    rate: RateSpec | None = RateSpec()
    compare: CompareSpec | None = None
    workers: int = Field(8, ge=1)


class ExperimentConfig(_Strict):
    """``catan experiment run``: train several configs at once, then analyse them."""

    name: str
    description: str = ""
    comparison: str = ""  # what counts as better, written down before the run
    runs: list[str]  # training config paths, relative to the experiment file
    max_resumes: int = Field(3, ge=0)  # per run, after crashes
    analysis: AnalysisSpec = AnalysisSpec()

    @model_validator(mode="after")
    def _check_runs(self) -> ExperimentConfig:
        if not self.runs:
            raise ValueError("an experiment needs at least one run")
        return self


def load_config[M: BaseModel](path: str | Path, model: type[M]) -> M:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return model.model_validate(raw)
