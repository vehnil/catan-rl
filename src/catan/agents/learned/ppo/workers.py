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

With ``games_per_worker > 1`` a worker runs that many games side by side
(``_BatchActor``) and batches network decisions that share a model across them:
the learner's, and the current-policy opponents'. At batch size 1 per-call
overhead and reading the weights from memory dominate, so this is cheaper per
decision. Each game is a separate env for GAE (id ``worker * games_per_worker +
game``).
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
) -> list[tuple[int, bool, int, int, int, int]]:
    """Play the given game numbers; the learner sits in seat ``g % 4``.

    Returns (g, won, vp, turns, rank, vp margin over the best opponent) per game.
    """
    out = []
    for g in games:
        seat = g % 4
        lineup = [make_agent(s, data_dir) for s in opponents]
        lineup.insert(seat, PPOAgent(hidden, model=model))
        rec = play_game(lineup, seed + g, rules)
        margin = rec.vp[seat] - max(v for s, v in enumerate(rec.vp) if s != seat)
        out.append((g, rec.winner == seat, rec.vp[seat], rec.turns, rec.ranks[seat], margin))
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
        self.league.latest = self.model  # updated in place with each rollout's weights
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


def _is_network(agent: Any) -> bool:
    return isinstance(agent, PPOAgent)


class _BatchActor:
    """``games`` envs played side by side, with their network decisions batched.

    Every env always waits on exactly one network decision: the learner's, or a network
    opponent's (random/greedy opponents move inline). Each round runs the largest group
    of envs waiting on the same model as one batch, then advances those envs to their
    next network decision. The learner and the current-policy opponents share a model
    across envs and batch well; past snapshots are spread over many models and mostly
    run one at a time. With one env this is exactly ``_Actor``'s order of play.
    """

    def __init__(
        self,
        games: int,
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
        self.model = ActorCritic(hidden).eval()
        self.league.latest = self.model  # updated in place with each rollout's weights
        self.envs = [
            CatanSingleAgentEnv(opponents=self.league.sample_opponents, config=rules, **env_kwargs)
            for _ in range(games)
        ]
        self.obs: list[np.ndarray] = [np.empty(0)] * games
        self.info: list[dict[str, Any]] = [{}] * games
        self.ep_return = [0.0] * games
        # Per env: "learner" (waiting on the learner), else the opponent agent it waits on,
        # with mode "step" (a learner action awaits its reward) or "reset".
        self.waiting: list[Any] = [None] * games
        self.mode = ["reset"] * games
        self._rows: list[dict[str, list[Any]]] = []
        self._episodes: list[list[tuple[bool, float, int]]] = []
        self._new_segments()
        for k, env in enumerate(self.envs):
            env.begin_reset(seed=seed * games + k)
            self._advance(k)

    def _new_segments(self) -> None:
        self._rows = [{f: [] for f in _ROW_FIELDS} for _ in self.envs]
        self._episodes = [[] for _ in self.envs]

    def _advance(self, k: int) -> None:
        """Run env ``k`` until it waits on a network decision."""
        env = self.envs[k]
        while True:
            agent = env.run_opponents(pause_for=_is_network)
            if agent is not None:
                self.waiting[k] = agent
                return
            if self.mode[k] == "reset":
                self.obs[k], self.info[k] = env.finish_reset()
                self.waiting[k] = "learner"
                return
            next_obs, r, term, trunc, next_info = env.finish_step()
            if trunc:  # bootstrap through the turn cap instead of treating it as a loss
                with torch.no_grad():
                    r += self.gamma * self.model(torch.as_tensor(next_obs).unsqueeze(0))[1].item()
            rows = self._rows[k]
            rows["rewards"][-1], rows["dones"][-1] = r, float(term or trunc)
            self.ep_return[k] += r
            if not (term or trunc):
                self.obs[k], self.info[k] = next_obs, next_info
                self.waiting[k] = "learner"
                return
            won = next_info.get("winner") == next_info["seat"]
            self._episodes[k].append((won, self.ep_return[k], next_info["turns"]))
            self.ep_return[k] = 0.0
            env.begin_reset()
            self.mode[k] = "reset"

    def _learner_round(self, ks: list[int]) -> None:
        obs = np.stack([self.obs[k] for k in ks])
        masks = np.stack([self.info[k]["action_mask"] for k in ks])
        a, logp, v = self.model.act(torch.as_tensor(obs), torch.as_tensor(masks))
        for i, k in enumerate(ks):
            rows = self._rows[k]
            rows["obs"].append(obs[i])
            rows["masks"].append(masks[i])
            rows["actions"].append(int(a[i]))
            rows["logprobs"].append(float(logp[i]))
            rows["values"].append(float(v[i]))
            rows["rewards"].append(0.0)
            rows["dones"].append(0.0)
            self.mode[k] = "step"
            self.envs[k].begin_step(int(a[i]))
            self._advance(k)

    def _opponent_round(self, ks: list[int]) -> None:
        agents = [cast(PPOAgent, self.waiting[k]) for k in ks]
        pending = [self.envs[k].pending_observation() for k in ks]
        model = agents[0].model
        if len(ks) == 1:  # the 1-D path, exactly as PPOAgent.act
            logits = model.policy_logits_numpy(agents[0].features(pending[0]))[None]
        else:
            x = np.stack([a.features(o) for a, o in zip(agents, pending, strict=True)])
            logits = model.policy_logits_numpy(x)
        for i, k in enumerate(ks):
            self.envs[k].opponent_move(agents[i].choose(pending[i], logits[i]))
            self._advance(k)

    def _round(self, learner: bool = True) -> bool:
        """Run the largest waiting group (the learner's only if ``learner``); False if none."""
        groups: dict[Any, list[int]] = {}
        for k, w in enumerate(self.waiting):
            if w == "learner":
                if learner:
                    groups.setdefault("learner", []).append(k)
            else:
                groups.setdefault(id(w.model), []).append(k)
        if not groups:
            return False
        key = max(groups, key=lambda g: (len(groups[g]), g == "learner"))
        if key == "learner":
            self._learner_round(groups[key])
        else:
            self._opponent_round(groups[key])
        return True

    def collect(self, steps: int) -> list[tuple[int, dict[str, Any]]]:
        """At least ``steps`` learner transitions in total, one segment per env (each env's
        rows in order; every row complete, so envs end waiting on the learner)."""
        while sum(len(r["actions"]) for r in self._rows) < steps:
            self._round()
        while self._round(learner=False):  # finish learner steps still awaiting a reward
            pass
        with torch.no_grad():
            last = self.model(torch.as_tensor(np.stack(self.obs)))[1]
        out = []
        for k, rows in enumerate(self._rows):
            seg: dict[str, Any] = {
                f: np.array(rows[f], dtype) if rows[f] else np.zeros((0, *shape), dtype)
                for f, (dtype, shape) in _ROW_FIELDS.items()
            }
            seg["last_value"] = float(last[k])
            seg["episodes"] = self._episodes[k]
            out.append((k, seg))
        self._new_segments()
        return out


_ROW_FIELDS: dict[str, tuple[Any, tuple[int, ...]]] = {
    "obs": (np.float32, (FEATURE_DIM,)),
    "masks": (bool, (NUM_ACTIONS,)),
    "actions": (np.int64, ()),
    "logprobs": (np.float32, ()),
    "values": (np.float32, ()),
    "rewards": (np.float32, ()),
    "dones": (np.float32, ()),
}


def _worker(conn: Connection, games_per_worker: int, *actor_args: Any) -> None:
    torch.set_num_threads(1)  # one process per core; don't oversubscribe
    actor: _Actor | _BatchActor = (
        _Actor(*actor_args) if games_per_worker == 1 else _BatchActor(games_per_worker, *actor_args)
    )
    while True:
        cmd, *args = conn.recv()
        try:
            if cmd == "weights":
                actor.model.load_state_dict(args[0])
                conn.send(("ok", None))
            elif cmd == "collect":
                out = actor.collect(args[0])
                conn.send(("ok", [(0, out)] if isinstance(out, dict) else out))
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
        games_per_worker: int = 1,
    ) -> None:
        self.n = n
        self.games = games_per_worker
        # Transitions requested per "collect" message. A batched worker drains its games to
        # learner decisions at the end of each message, which breaks up batches, so it gets
        # longer messages; it can return up to one extra row per game (one learner round).
        self.chunk_steps = CHUNK if games_per_worker == 1 else max(CHUNK, 16 * games_per_worker)
        # At most this many transitions per message; a rollout overshoots by less than this.
        self.chunk_rows = self.chunk_steps + games_per_worker - 1
        self.max_snapshots = league_cfg.max_snapshots
        self.snapshot_states: list[dict[str, torch.Tensor]] = []  # kept for resuming
        ctx = mp.get_context("spawn")
        self.conns: list[Connection] = []
        self.procs = []
        for i in range(n):
            parent, child = ctx.Pipe()
            args = (
                rules,
                env_kwargs,
                gamma,
                league_cfg,
                hidden,
                seed * 1000 + i,
                data_dir,
            )
            proc = ctx.Process(target=_worker, args=(child, games_per_worker, *args), daemon=True)
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
        """At least ``steps`` transitions as (env, segment), each env's in order. Env ids are
        ``worker * games_per_worker + game``."""
        self._broadcast("weights", _state(model))
        index = {c: i for i, c in enumerate(self.conns)}
        for c in self.conns:
            c.send(("collect", self.chunk_steps))
        busy, received, out = set(self.conns), 0, []
        while busy:
            for ready in wait(list(busy)):
                conn = cast(Connection, ready)
                for k, seg in self._unwrap(conn):
                    out.append((index[conn] * self.games + k, seg))
                    received += len(seg["actions"])
                # Hand out another chunk only while the ones in flight can't fill the rollout.
                if received + self.chunk_rows * (len(busy) - 1) < steps:
                    conn.send(("collect", self.chunk_steps))
                else:
                    busy.discard(conn)
        return out

    @property
    def snapshots(self) -> int:
        return len(self.snapshot_states)

    def add_snapshot(self, model: ActorCritic) -> None:
        self.add_snapshot_state(_state(model))

    def add_snapshot_state(self, state: dict[str, torch.Tensor]) -> None:
        self._broadcast("snapshot", state)
        self.snapshot_states = [*self.snapshot_states, state][-self.max_snapshots :]

    def evaluate(
        self,
        model: ActorCritic,
        hidden: list[int],
        opponents: list[str],
        games: int,
        rules: RulesConfig,
        seed: int,
    ) -> list[tuple[int, bool, int, int, int, int]]:
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
