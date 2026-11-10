"""PPO with action masking, trained against a league of fixed agents and past snapshots.

Writes per-update metrics to ``data/runs/<run_id>/metrics.jsonl``, registers
the run and the final policy in the SQLite store, and saves the policy under
``data/policies/<id>``.
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
from catan.agents.registry import make_agent
from catan.config.models import TrainConfig
from catan.engine.actions import NUM_ACTIONS
from catan.engine.state import RulesConfig
from catan.env.gym_single import CatanSingleAgentEnv
from catan.player.features import FEATURE_DIM
from catan.sim.artifacts import load_policy, resolve_policy_path, save_policy
from catan.sim.runner import play_game
from catan.sim.store import Store


def evaluate_policy(
    model: ActorCritic,
    hidden: list[int],
    opponents: list[str],
    games: int,
    rules: RulesConfig,
    seed: int,
) -> dict[str, float]:
    """Learner rotates through all four seats against the given opponents."""
    wins, vps = 0, []
    for g in range(games):
        seat = g % 4
        lineup = [make_agent(s) for s in opponents]
        lineup.insert(seat, PPOAgent(hidden, model=model))
        rec = play_game(lineup, seed + g, rules)
        wins += rec.winner == seat
        vps.append(rec.vp[seat])
    return {"eval_win_rate": wins / games, "eval_mean_vp": float(np.mean(vps))}


def _ppo_update(
    model: ActorCritic, opt: torch.optim.Optimizer, buf: RolloutBuffer, cfg, rng
) -> dict[str, float]:  # noqa: ANN001
    stats = {"pg_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n = 0
    model.train()
    for _ in range(cfg.epochs):
        for mb in buf.minibatches(cfg.minibatch, rng, cfg.device):
            logp, ent, value = model.evaluate(mb["obs"], mb["masks"], mb["actions"])
            ratio = (logp - mb["logprobs"]).exp()
            adv = mb["advantages"]
            pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            vloss = 0.5 * (value - mb["returns"]).pow(2).mean()
            loss = pg + cfg.value_coef * vloss - cfg.entropy_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["pg_loss"] += pg.item()
                stats["value_loss"] += vloss.item()
                stats["entropy"] += ent.mean().item()
                stats["approx_kl"] += (mb["logprobs"] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            n += 1
    model.eval()
    return {k: v / max(n, 1) for k, v in stats.items()}


def train(cfg: TrainConfig, log: Callable[[str], Any] = print) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    p = cfg.ppo
    data_dir = Path(cfg.data_dir)
    rules = cfg.rules.to_rules()

    store = Store(data_dir / "catan.db")
    run_id = store.start_run("train", cfg.model_dump(), cfg.name)
    run_dir = data_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (run_dir / "metrics.jsonl").open("a")
    log(f"run {run_id}: metrics -> {run_dir / 'metrics.jsonl'}")

    model = ActorCritic(p.hidden).to(p.device)
    parent_id = None
    if cfg.init_policy:
        init = load_policy(resolve_policy_path(cfg.init_policy, data_dir))
        assert isinstance(init, PPOAgent), "init_policy must be a PPO policy"
        model.load_state_dict(init.model.state_dict())
        parent_id = getattr(init, "spec", "").removeprefix("policy:") or None
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=p.lr, eps=1e-5)

    league = League(cfg.league, p.hidden, rng)
    env = CatanSingleAgentEnv(
        opponents=league.sample_opponents, config=rules, vp_reward_scale=p.vp_reward_scale
    )
    buf = RolloutBuffer(p.rollout_steps, FEATURE_DIM, NUM_ACTIONS)
    obs, info = env.reset(seed=cfg.seed)

    steps, episodes, t0 = 0, 0, time.perf_counter()
    recent_wins: deque[float] = deque(maxlen=100)
    recent_returns: deque[float] = deque(maxlen=100)
    ep_return = 0.0
    history: list[dict[str, Any]] = []
    last_eval: dict[str, float] = {}

    try:
        while steps < p.total_steps:
            buf.reset()
            while not buf.full:
                x = torch.as_tensor(obs, device=p.device).unsqueeze(0)
                m = torch.as_tensor(info["action_mask"], device=p.device).unsqueeze(0)
                a, logp, v = model.act(x, m)
                next_obs, r, term, trunc, next_info = env.step(int(a))
                if trunc:  # bootstrap through the turn cap instead of treating it as a loss
                    with torch.no_grad():
                        r += p.gamma * model(torch.as_tensor(next_obs).unsqueeze(0))[1].item()
                buf.add(obs, info["action_mask"], int(a), logp.item(), v.item(), r, term or trunc)
                steps += 1
                ep_return += r
                if term or trunc:
                    episodes += 1
                    recent_wins.append(float(next_info.get("winner") == next_info["seat"]))
                    recent_returns.append(ep_return)
                    ep_return = 0.0
                    obs, info = env.reset()
                else:
                    obs, info = next_obs, next_info
                if steps % cfg.league.snapshot_every == 0:
                    league.add_snapshot(model)
                if steps % cfg.eval_every == 0:
                    last_eval = evaluate_policy(
                        model, p.hidden, cfg.eval_opponents, cfg.eval_games, rules, cfg.seed + steps
                    )
                    log(f"  eval @ {steps}: {last_eval}")

            with torch.no_grad():
                last_value = model(torch.as_tensor(obs).unsqueeze(0))[1].item()
            buf.compute_returns(last_value, p.gamma, p.gae_lambda)
            upd = _ppo_update(model, opt, buf, p, rng)
            row = {
                "steps": steps,
                "episodes": episodes,
                "win_rate_100": float(np.mean(recent_wins)) if recent_wins else None,
                "return_100": float(np.mean(recent_returns)) if recent_returns else None,
                "sps": steps / (time.perf_counter() - t0),
                "snapshots": len(league.snapshots),
                **upd,
                **last_eval,
            }
            history.append(row)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            log(
                f"steps {steps:>8} ep {episodes:>5} win100 {row['win_rate_100'] or 0:.2f} "
                f"ent {upd['entropy']:.2f} kl {upd['approx_kl']:.4f} sps {row['sps']:.0f}"
            )

        final_eval = evaluate_policy(
            model, p.hidden, cfg.eval_opponents, cfg.eval_games, rules, cfg.seed + 10**6
        )
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
        metrics_file.close()
        store.close()
