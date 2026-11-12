"""PPO with action masking, trained against a league of fixed agents and past snapshots.

Writes per-update metrics to ``data/runs/<run_id>/metrics.jsonl``, registers
the run and the final policy in the SQLite store, and saves the policy under
``data/policies/<id>``. With ``checkpoint_every``, the full training state is
also saved to ``data/runs/<run_id>/train_state.pt`` so a stopped run can resume.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from catan.agents.learned.ppo.agent import PPOAgent
from catan.agents.learned.ppo.buffer import RolloutBuffer
from catan.agents.learned.ppo.league import League
from catan.agents.learned.ppo.model import ActorCritic
from catan.agents.learned.ppo.workers import RolloutWorkers, eval_games
from catan.config.models import TrainConfig
from catan.engine.actions import NUM_ACTIONS
from catan.engine.state import RulesConfig
from catan.env.gym_single import CatanSingleAgentEnv
from catan.player.features import FEATURE_DIM
from catan.sim.artifacts import load_policy, read_manifest, resolve_policy_path, save_policy
from catan.sim.store import Store


def evaluate_policy(
    model: ActorCritic,
    hidden: list[int],
    opponents: list[str],
    games: int,
    rules: RulesConfig,
    seed: int,
    data_dir: str | Path = "data",
) -> dict[str, float]:
    """Learner rotates through all four seats against the given opponents."""
    results = eval_games(model, hidden, opponents, list(range(games)), rules, seed, data_dir)
    return _eval_summary(results)


def _eval_summary(results: list[tuple[int, bool, int, int, int, int]]) -> dict[str, float]:
    wins = sum(r[1] for r in results)
    return {
        "eval_win_rate": wins / len(results),
        "eval_mean_vp": float(np.mean([r[2] for r in results])),
        "eval_mean_turns": float(np.mean([r[3] for r in results])),
        "eval_mean_rank": float(np.mean([r[4] for r in results])),
        "eval_vp_margin": float(np.mean([r[5] for r in results])),
    }


def _pin_policies(
    specs: list[str], data_dir: Path, log: Callable[[str], Any], logged: set[str]
) -> list[str]:
    """Resolve each ``policy:<ref>`` to its id once, so every worker and game uses the same one."""
    out = []
    for spec in specs:
        if spec.startswith("policy:"):
            ref = spec.removeprefix("policy:")
            pid = read_manifest(resolve_policy_path(ref, data_dir))["id"]
            if ref != pid and spec not in logged:
                log(f"  {spec} -> policy:{pid}")
                logged.add(spec)
            spec = f"policy:{pid}"
        out.append(spec)
    return out


def _ppo_update(
    model: ActorCritic, opt: torch.optim.Optimizer, buf: RolloutBuffer, cfg, rng
) -> dict[str, float]:  # noqa: ANN001
    stats = {
        "pg_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "grad_norm": 0.0,
    }
    n = 0
    epochs_run = 0
    model.train()
    for _ in range(cfg.epochs):
        epoch_kl, epoch_n = 0.0, 0
        for mb in buf.minibatches(cfg.minibatch, rng, cfg.device):
            logp, ent, value = model.evaluate(mb["obs"], mb["masks"], mb["actions"])
            log_ratio = logp - mb["logprobs"]
            ratio = log_ratio.exp()
            adv = mb["advantages"]
            pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            vloss = 0.5 * (value - mb["returns"]).pow(2).mean()
            loss = pg + cfg.value_coef * vloss - cfg.entropy_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                # Low-variance, non-negative KL estimate ((r - 1) - log r), as in the 37 PPO details.
                kl = ((ratio - 1) - log_ratio).mean().item()
                stats["pg_loss"] += pg.item()
                stats["value_loss"] += vloss.item()
                stats["entropy"] += ent.mean().item()
                stats["approx_kl"] += kl
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                stats["grad_norm"] += float(gnorm)
            epoch_kl += kl
            epoch_n += 1
            n += 1
        epochs_run += 1
        if cfg.target_kl is not None and epoch_kl / max(epoch_n, 1) > cfg.target_kl:
            break
    model.eval()
    out = {k: v / max(n, 1) for k, v in stats.items()}
    out["epochs_run"] = epochs_run
    return out


def _rollout_stats(buf: RolloutBuffer) -> dict[str, float]:
    """Explained variance of the value head on this rollout's returns: near 0 means the
    advantages are mostly noise; 1 means the value predicts returns perfectly."""
    values, returns = buf.values[: buf.ptr], buf.returns
    var = float(np.var(returns))
    ev = float("nan") if var == 0 else 1 - float(np.var(returns - values)) / var
    return {"explained_var": ev, "value_mean": float(values.mean())}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-write leaves the previous state intact


def train(
    cfg: TrainConfig, log: Callable[[str], Any] = print, resume: str | None = None
) -> dict[str, Any]:
    """Train from scratch, or with ``resume`` (a run id or config name) continue a stopped run
    from its last ``train_state.pt``. A single-env run resumes exactly; with ``num_envs > 1``
    the workers start new games but keep the snapshot pool."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    p = cfg.ppo
    data_dir = Path(cfg.data_dir)
    rules = cfg.rules.to_rules()

    store = Store(data_dir / "catan.db")
    saved: dict[str, Any] | None = None
    if resume is None:
        run_id = store.start_run("train", cfg.model_dump(), cfg.name)
    else:
        run_id = store.find_run(resume)["id"]
        state_path = data_dir / "runs" / run_id / "train_state.pt"
        if not state_path.exists():
            store.close()
            raise FileNotFoundError(f"run {run_id} has no train_state.pt (needs checkpoint_every)")
        saved = torch.load(state_path, weights_only=False)
        store.reopen_run(run_id)
    run_dir = data_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if saved is not None and metrics_path.exists():  # drop rows logged after the saved state
        kept = [
            line
            for line in metrics_path.read_text().splitlines()
            if line.strip() and json.loads(line)["steps"] <= saved["steps"]
        ]
        metrics_path.write_text("".join(line + "\n" for line in kept))
    metrics_file = metrics_path.open("a")
    log(f"run {run_id}: metrics -> {metrics_path}")
    if saved is not None:
        log(f"  resuming at step {saved['steps']}")

    model = ActorCritic(p.hidden).to(p.device)
    parent_id = None
    if saved is not None:
        parent_id = saved["parent_id"]
    elif cfg.init_policy:
        init = load_policy(resolve_policy_path(cfg.init_policy, data_dir))
        assert isinstance(init, PPOAgent), "init_policy must be a PPO policy"
        model.load_state_dict(init.model.state_dict())
        parent_id = getattr(init, "spec", "").removeprefix("policy:") or None
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=p.lr, eps=1e-5)
    if saved is not None:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        rng = saved["rng"]

    logged: set[str] = set()
    league_cfg = cfg.league.model_copy(
        update={"fixed_opponents": _pin_policies(cfg.league.fixed_opponents, data_dir, log, logged)}
    )
    eval_opponents = _pin_policies(cfg.eval_opponents, data_dir, log, logged)
    env_kwargs: dict[str, Any] = {
        "vp_reward_scale": p.vp_reward_scale,
        "fast_win_bonus": p.fast_win_bonus,
        "fast_win_turns": p.fast_win_turns,
    }
    workers = None
    ep_return = 0.0
    if p.num_envs > 1:
        # New worker seeds on resume, so the restarted workers don't replay the same games.
        worker_seed = cfg.seed if saved is None else cfg.seed + saved["steps"]
        workers = RolloutWorkers(
            p.num_envs,
            rules,
            env_kwargs,
            p.gamma,
            league_cfg,
            p.hidden,
            worker_seed,
            data_dir,
            p.games_per_worker,
        )
        for snap in saved["snapshots"] if saved is not None else []:
            workers.add_snapshot_state(snap)
        buf = RolloutBuffer(p.rollout_steps + workers.chunk_rows, FEATURE_DIM, NUM_ACTIONS)
    else:
        buf = RolloutBuffer(p.rollout_steps, FEATURE_DIM, NUM_ACTIONS)
        if saved is None:
            league = League(league_cfg, p.hidden, rng, data_dir)
            league.latest = model
            env = CatanSingleAgentEnv(opponents=league.sample_opponents, config=rules, **env_kwargs)
            obs, info = env.reset(seed=cfg.seed)
        else:  # pickled together with rng, so they still share it
            league, env = saved["league"], saved["env"]
            obs, info, ep_return = saved["obs"], saved["info"], saved["ep_return"]
            league.latest = model  # the "latest" opponent is the network being trained

    def run_eval(seed: int) -> dict[str, float]:
        if workers is None:
            return evaluate_policy(
                model, p.hidden, eval_opponents, cfg.eval_games, rules, seed, data_dir
            )
        return _eval_summary(
            workers.evaluate(model, p.hidden, eval_opponents, cfg.eval_games, rules, seed)
        )

    steps, episodes, t0 = 0, 0, time.perf_counter()
    checkpointed = 0  # steps at the last checkpoint check
    recent_wins: deque[float] = deque(maxlen=100)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_turns: deque[int] = deque(maxlen=100)
    history: list[dict[str, Any]] = []
    last_eval: dict[str, float] = {}
    if saved is not None:
        steps, episodes, checkpointed = saved["steps"], saved["episodes"], saved["steps"]
        recent_wins.extend(saved["recent_wins"])
        recent_returns.extend(saved["recent_returns"])
        recent_turns.extend(saved["recent_turns"])
        last_eval = saved["last_eval"]
        history = [saved["last_row"]] if saved["last_row"] else []
        torch.set_rng_state(saved["torch_rng"])
    t_steps = steps  # steps/sec counts from here

    try:
        while steps < p.total_steps:
            if p.lr_schedule == "linear":
                for g in opt.param_groups:
                    g["lr"] = p.lr * max(0.0, 1 - steps / p.total_steps)
            buf.reset()
            if workers is not None:
                before = steps
                last_values = {}
                for w, seg in workers.collect(model, p.rollout_steps):
                    buf.add_segment(seg, env=w)
                    last_values[w] = seg["last_value"]
                    for won, ret, turns in seg["episodes"]:
                        episodes += 1
                        recent_wins.append(float(won))
                        recent_returns.append(ret)
                        recent_turns.append(turns)
                steps += buf.ptr
                if steps // cfg.league.snapshot_every > before // cfg.league.snapshot_every:
                    workers.add_snapshot(model)
                if steps // cfg.eval_every > before // cfg.eval_every:
                    last_eval = run_eval(cfg.seed + steps)
                    log(f"  eval @ {steps}: {last_eval}")
            else:
                while not buf.full:
                    x = torch.as_tensor(obs, device=p.device).unsqueeze(0)
                    m = torch.as_tensor(info["action_mask"], device=p.device).unsqueeze(0)
                    a, logp, v = model.act(x, m)
                    next_obs, r, term, trunc, next_info = env.step(int(a))
                    if trunc:  # bootstrap through the turn cap instead of treating it as a loss
                        with torch.no_grad():
                            r += p.gamma * model(torch.as_tensor(next_obs).unsqueeze(0))[1].item()
                    buf.add(
                        obs, info["action_mask"], int(a), logp.item(), v.item(), r, term or trunc
                    )
                    steps += 1
                    ep_return += r
                    if term or trunc:
                        episodes += 1
                        recent_wins.append(float(next_info.get("winner") == next_info["seat"]))
                        recent_returns.append(ep_return)
                        recent_turns.append(next_info["turns"])
                        ep_return = 0.0
                        obs, info = env.reset()
                    else:
                        obs, info = next_obs, next_info
                    if steps % cfg.league.snapshot_every == 0:
                        league.add_snapshot(model)
                    if steps % cfg.eval_every == 0:
                        last_eval = run_eval(cfg.seed + steps)
                        log(f"  eval @ {steps}: {last_eval}")

                with torch.no_grad():
                    last_values = {0: model(torch.as_tensor(obs).unsqueeze(0))[1].item()}
            buf.compute_returns(last_values, p.gamma, p.gae_lambda)
            upd = {**_rollout_stats(buf), **_ppo_update(model, opt, buf, p, rng)}
            ce = cfg.checkpoint_every
            save_state = bool(ce and steps // ce > checkpointed // ce)
            if save_state:
                ckpt, _ = save_policy(
                    PPOAgent(p.hidden, model=model),
                    data_dir,
                    name=f"{cfg.name}@{steps}-{run_id}",
                    run_id=run_id,
                    parent_id=parent_id,
                    metrics={"steps": steps, "episodes": episodes, "checkpoint": True},
                    store=store,
                )
                log(f"  checkpoint @ {steps}: {ckpt['id']}")
            checkpointed = steps
            row = {
                "steps": steps,
                "episodes": episodes,
                "win_rate_100": float(np.mean(recent_wins)) if recent_wins else None,
                "return_100": float(np.mean(recent_returns)) if recent_returns else None,
                "turns_100": float(np.mean(recent_turns)) if recent_turns else None,
                "sps": (steps - t_steps) / (time.perf_counter() - t0),
                "snapshots": len(league.snapshots) if workers is None else workers.snapshots,
                "lr": opt.param_groups[0]["lr"],
                **upd,
                **last_eval,
            }
            history.append(row)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            if save_state:
                state: dict[str, Any] = {
                    "steps": steps,
                    "episodes": episodes,
                    "parent_id": parent_id,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "rng": rng,
                    "torch_rng": torch.get_rng_state(),
                    "recent_wins": list(recent_wins),
                    "recent_returns": list(recent_returns),
                    "recent_turns": list(recent_turns),
                    "last_eval": last_eval,
                    "last_row": row,
                }
                if workers is None:
                    state.update(league=league, env=env, obs=obs, info=info, ep_return=ep_return)
                else:
                    state["snapshots"] = workers.snapshot_states
                _save_state(run_dir / "train_state.pt", state)
            log(
                f"steps {steps:>8} ep {episodes:>5} win100 {row['win_rate_100'] or 0:.2f} "
                f"ent {upd['entropy']:.2f} kl {upd['approx_kl']:.4f} clip {upd['clip_frac']:.2f} "
                f"ev {upd['explained_var']:.2f} sps {row['sps']:.0f}"
            )

        final_eval = run_eval(cfg.seed + 10**6)
        manifest, path = save_policy(
            PPOAgent(p.hidden, model=model),
            data_dir,
            name=f"{cfg.name}-{run_id}",
            run_id=run_id,
            parent_id=parent_id,
            metrics={"steps": steps, "episodes": episodes, **final_eval},
            store=store,
        )
        result = {
            "run_id": run_id,
            "policy_id": manifest["id"],
            "policy_path": str(path),
            **final_eval,
        }
        store.finish_run(run_id, {**result, "last_update": history[-1] if history else {}})
        log(f"saved policy {manifest['id']} -> {path}")
        return result
    except BaseException:
        store.finish_run(run_id, {"steps": steps}, status="failed")
        raise
    finally:
        if workers is not None:
            workers.close()
        metrics_file.close()
        store.close()
