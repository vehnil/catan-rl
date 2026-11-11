# Training (PPO self-play)

The trainer runs the learner in `CatanSingleAgentEnv`, draws opponents from the agent registry, and saves results through the simulation layer's store and policy artifacts.

## PPO (`agents/learned/ppo/`)

| File | Role |
|---|---|
| `model.py` | `ActorCritic`: tanh MLP trunk, policy head (488 logits), value head; orthogonal init |
| `buffer.py` | `RolloutBuffer`: on-policy storage tagged by env, GAE(λ) per env, normalized advantages, minibatches |
| `league.py` | `League`: opponent pool of fixed specs plus frozen snapshots of the learner |
| `agent.py` | `PPOAgent`: playable wrapper, safetensors save/load, ONNX export |
| `workers.py` | `RolloutWorkers`: parallel collection, one env and policy copy per process (`num_envs > 1`) |
| `trainer.py` | `train(cfg)`: the loop below |

### Loop
1. Collect `rollout_steps` learner decisions in the single-agent env. Opponents are resampled from the league every episode: each seat is a past snapshot with probability `self_play_prob`, otherwise a fixed spec. Illegal actions are masked.
2. On truncation, bootstrap with γ·V(s′) rather than treating the turn cap as a loss.
3. GAE → clipped PPO update for `epochs` × minibatches, with entropy bonus, value loss and gradient clipping.
4. Every `snapshot_every` steps, freeze a copy of the learner into the league (keeping the last `max_snapshots`).
5. Every `eval_every` steps, play `eval_games` against `eval_opponents`, rotating the learner through all 4 seats.
6. At the end: a final evaluation, then the policy is saved to `data/policies/<id>` and registered in the store with lineage (`parent_id` if warm-started via `init_policy`).

Metrics per update go to `data/runs/<run_id>/metrics.jsonl`: steps, episodes, win rate, return and game length (turns) over the last 100 games, steps/sec, losses, entropy, approx KL, clip fraction, and the latest eval.

### Config (`TrainConfig`, YAML)

Full example: [`experiments/v1_examples/ppo_league.yaml`](../experiments/v1_examples/ppo_league.yaml). All fields and defaults are in `config/models.py`. Unknown keys are rejected, so typos fail immediately. The fields that matter most:

| Field | Default | Why it matters |
|---|---|---|
| `init_policy` | null | Policy id, name, or training config name (newest policy from that config) to warm-start from; recorded as `parent_id` |
| `ppo.gamma` | 0.995 | Games are long (~60+ turns, hundreds of decisions) |
| `ppo.vp_reward_scale` | 0.05 | Dense ΔVP shaping on top of the ±1 outcome |
| `ppo.num_envs` | 1 | Parallel rollout workers; see below |
| `ppo.fast_win_bonus` / `fast_win_turns` | 0 / 200 | Extra reward for a win, linear from the full bonus at turn 0 to 0 at `fast_win_turns`; losses get none |
| `league.fixed_opponents` | `[random, greedy]` | Non-snapshot opponent pool. `policy:<ref>` entries are pinned to a policy id at start, loaded once and kept frozen |
| `league.self_play_prob` | 0.5 | P(an opponent seat is a past snapshot) |
| `league.snapshot_every` / `max_snapshots` | 20,000 / 5 | Snapshot cadence and pool size |
| `league.lineup` | false | true: `fixed_opponents` is the exact 3-seat lineup (shuffled per game) instead of a pool |
| `eval_every` / `eval_games` / `eval_opponents` | 50,000 / 40 / 3× random | Periodic evaluation |
| `checkpoint_every` | null | Save the policy every N steps as `<name>@<steps>-<run id>`, for `catan curve`. Checkpoints don't affect `init_policy: <config name>`, which still picks the final policy |

## Results so far

`experiments/v1_examples/ppo_small.yaml` (20k steps, about 30 s on CPU, random opponents): training win rate 0.33 → 0.46; final eval 40% over 20 games, where chance is 25%. That only shows the loop learns. A real result needs millions of steps against greedy.

## Practical notes

- **Throughput depends on the opponents.** About 1,000 learner steps/sec against random in one process, much less against greedy or search agents.
- **Parallel workers (`ppo.num_envs`).** With `num_envs > 1`, each worker process runs one env with its own copy of the policy (one torch thread). Each rollout goes like this:
  - The trainer sends the current weights.
  - It hands out chunks of 64 steps until `rollout_steps` is reached, so a rollout can run up to 63 steps over.
  - It runs the update.

  Episodes continue across rollouts. Snapshots and evals happen between rollouts, and eval games are split across the workers. `num_envs: 1` (the default) keeps the original single-process loop exactly.

  On an M1 (4 performance + 4 efficiency cores), 8 workers collected about 2,500 steps/s against random/greedy/snapshot opponents, against about 1,000 in one process. That was measured with a background system process using more than one core. Workers sit idle during the update, which is about 15% of a rollout.
- **Scripts that call `train()` with `num_envs > 1` need an `if __name__ == "__main__":` guard.** Workers are spawned, and spawning re-imports the main script. The `catan train` CLI is already guarded.
- **Curriculum:** start against `random`, move to `greedy` once the win rate is above ~60%, then raise `self_play_prob`. Warm-start each stage with `init_policy`.
- **Credit assignment:** a game is hundreds of decisions with a reward only at the end; keep `gamma` high. `vp_reward_scale` helps early, but too much of it trains a VP-grabber that ignores blocking.
- **Measure progress with tournaments,** not training win rate: `catan tournament` with a gauntlet of `policy:<id>` against greedy presets, and watch `catan ratings`.
