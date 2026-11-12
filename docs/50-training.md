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
1. Collect `rollout_steps` learner decisions in the single-agent env. Opponents are resampled from the league every episode. Each seat is the current policy with probability `latest_prob`, else a past snapshot with probability `self_play_prob`, otherwise a fixed spec. Illegal actions are masked.
2. On truncation, bootstrap with γ·V(s′) rather than treating the turn cap as a loss.
3. GAE → clipped PPO update for `epochs` × minibatches, with entropy bonus, value loss and gradient clipping. With `target_kl` set, the remaining epochs are skipped once an epoch's mean KL exceeds it.
4. Every `snapshot_every` steps, freeze a copy of the learner into the league (keeping the last `max_snapshots`).
5. Every `eval_every` steps, play `eval_games` against `eval_opponents`, rotating the learner through all 4 seats.
6. At the end: a final evaluation, then the policy is saved to `data/policies/<id>` and registered in the store with lineage (`parent_id` if warm-started via `init_policy`).

Metrics per update go to `data/runs/<run_id>/metrics.jsonl`: steps, episodes, win rate, return and game length (turns) over the last 100 games, steps/sec, learning rate, losses, entropy, approx KL (the `(r − 1) − log r` estimator), clip fraction, gradient norm (before clipping), epochs run, and the latest eval. Two rollout statistics:
- **`explained_var`:** `1 − Var(returns − V) / Var(returns)`. This is the main health check: near 0 means the value head predicts nothing, so the advantages are mostly noise.
- **`value_mean`:** the mean value prediction.

`catan health <run>` averages these over windows of updates and flags warning signs (explained variance < 0.1, KL > 0.05, clip fraction > 0.3).

### Config (`TrainConfig`, YAML)

Full example: [`experiments/v1_examples/ppo_league.yaml`](../experiments/v1_examples/ppo_league.yaml). All fields and defaults are in `config/models.py`. Unknown keys are rejected, so typos fail immediately. The fields that matter most:

| Field | Default | Why it matters |
|---|---|---|
| `init_policy` | null | Policy id, name, or training config name (newest policy from that config) to warm-start from; recorded as `parent_id` |
| `ppo.gamma` | 0.995 | Games are long (~60+ turns, hundreds of decisions) |
| `ppo.vp_reward_scale` | 0.05 | Dense ΔVP shaping on top of the ±1 outcome |
| `ppo.num_envs` | 1 | Parallel rollout workers; see below |
| `ppo.games_per_worker` | 1 | Games each worker plays side by side, batching their network decisions; see below. Needs `num_envs > 1` |
| `ppo.target_kl` | null | Stop an update's epochs early once their mean KL exceeds this |
| `ppo.lr_schedule` | constant | `linear` decays the learning rate to 0 at `total_steps`, standard for long runs |
| `ppo.fast_win_bonus` / `fast_win_turns` | 0 / 200 | Extra reward for a win, linear from the full bonus at turn 0 to 0 at `fast_win_turns`; losses get none |
| `league.fixed_opponents` | `[random, greedy]` | Non-snapshot opponent pool. `policy:<ref>` entries are pinned to a policy id at start, loaded once and kept frozen |
| `league.latest_prob` | 0 | P(an opponent seat is the current policy itself), drawn first. 0 keeps the old behaviour exactly |
| `league.self_play_prob` | 0.5 | P(an opponent seat is a past snapshot, chosen uniformly), for seats not taken by `latest_prob` |
| `league.snapshot_every` / `max_snapshots` | 20,000 / 5 | Snapshot cadence and pool size |
| `league.lineup` | false | true: `fixed_opponents` is the exact 3-seat lineup (shuffled per game) instead of a pool |
| `eval_every` / `eval_games` / `eval_opponents` | 50,000 / 40 / 3× random | Periodic evaluation |
| `checkpoint_every` | null | Save the policy every N steps as `<name>@<steps>-<run id>`, for `catan curve`, plus the full training state for `--resume`. Checkpoints don't affect `init_policy: <config name>`, which still picks the final policy |

### Resuming a stopped run

With `checkpoint_every` set, each checkpoint also writes `data/runs/<run_id>/train_state.pt`, replacing the previous one atomically. It holds:
- the model and optimizer state;
- the step and episode counters;
- the RNG states;
- the recent-game windows;
- the opponent snapshot pool.

```bash
catan train experiments/v7_baseline/baseline_s1.yaml --resume <run id or config name>
```

- **What carries on:** the run keeps its id and metrics file. Metrics rows logged after the saved state are dropped, so nothing is counted twice. Checkpoints re-saved after the resume point replace the older ones in `catan curve`.
- **Changing the config on resume:** you can pass a config with a larger `total_steps` to extend a finished run. Other changes take effect from the resume point, but the run's stored config isn't updated.
- **How exact it is:**
  - A single-env run (`num_envs: 1`) resumes exactly. The result matches an uninterrupted run, weights and metrics included.
  - With parallel workers, the workers start new games (with new seeds) but get the saved snapshot pool back.

## Results so far

`experiments/v1_examples/ppo_small.yaml` (20k steps, about 30 s on CPU, random opponents): training win rate 0.33 → 0.46; final eval 40% over 20 games, where chance is 25%. That only shows the loop learns. A real result needs millions of steps against greedy.

## Practical notes

- **Throughput depends on the opponents.** About 1,000 learner steps/sec against random in one process, much less against greedy or search agents.
- **Parallel workers (`ppo.num_envs`).** With `num_envs > 1`, each worker process runs one env with its own copy of the policy (one torch thread). Each rollout goes like this:
  - The trainer sends the current weights.
  - It hands out chunks of 64 steps until `rollout_steps` is reached, so a rollout can run up to 63 steps over.
  - It runs the update.

  Episodes continue across rollouts. Snapshots and evals happen between rollouts, and eval games are split across the workers. `num_envs: 1` (the default) keeps the original single-process loop exactly. Parallel runs are reproducible only up to timing: the order in which chunks arrive decides the rollout's row order, so floating-point sums (and later the weights) can differ in the last digits between two runs of the same config. Use `num_envs: 1` when you need runs that match exactly.
- **Batched games (`ppo.games_per_worker`).** With more than one, each worker plays that many games side by side. Every game always waits on exactly one network decision: the learner's, or a network opponent's (random and greedy opponents move inline). Each round, the worker runs the largest group of games waiting on the same model as one batched forward pass.
  - **What batches:** the learner, and current-policy opponents (`latest_prob`), share one model across games, so they batch well. Past snapshots are spread over many models and mostly run one at a time.
  - **Why it helps:** at batch size 1, per-call overhead and reading the roughly 2.8 MB of weights from memory dominate. Under load the weights get evicted from the cache between calls, which made this the bottleneck.
  - **Measured on the M1** (six workers at once, v7 opponent mix): about 290 steps/s per worker with 1 game, 334 with 16, 575 with 32, and 603 with 64. Running alone, the gain is small (about 1.0–1.45×), because the weights stay in cache.
  - **Rollout layout:** each game is its own env for GAE (id `worker × games_per_worker + game`). Workers get longer collect messages (`16 × games_per_worker` steps), since each message ends by playing every game on to its next learner decision.
  - **Exactness:** with one game, the batched worker reproduces the unbatched one exactly. A test checks this.

  On an M1 (4 performance + 4 efficiency cores), 8 workers collected about 2,500 steps/s against random/greedy/snapshot opponents, against about 1,000 in one process. That was measured with a background system process using more than one core. Workers sit idle during the update, which is about 15% of a rollout.
- **Scripts that call `train()` with `num_envs > 1` need an `if __name__ == "__main__":` guard.** Workers are spawned, and spawning re-imports the main script. The `catan train` CLI is already guarded.
- **Curriculum:** start against `random`, move to `greedy` once the win rate is above ~60%, then raise `self_play_prob`. Warm-start each stage with `init_policy`.
- **Credit assignment:** a game is hundreds of decisions with a reward only at the end; keep `gamma` high. `vp_reward_scale` helps early, but too much of it trains a VP-grabber that ignores blocking.
- **Measure progress with tournaments,** not training win rate: `catan tournament` with a gauntlet of `policy:<id>` against greedy presets, and watch `catan ratings`.
