---
name: experiment-proposal
description: Set up the next catan-rl experiment - proposal, configs, experiment.yaml, smoke test - ready for the user to launch.
---

# Setting up an experiment

The runner refuses uncommitted changes. The user launches long runs; you set them up and smoke-test them.

## Proposal: `experiments/vN_<name>/vN-proposal.md`

Follow `experiments/v9_rank_reward/v9-proposal.md`: question, hypothesis with one bold prediction, design table, what it tests, comparison, outcomes → next (every row leads somewhere), cost. Fix the comparison before the run:
- **Primary:** VP margin vs 3× greedy, 400 games, seed 6000, paired by seed. Better = all paired diffs > 0 and mean ≥ +0.3; worse = the mirror; else no detectable difference.
- **Secondary:** rank, win rate, mean VP, reference rating, 2 random + greedy. Greedy itself scores −2.18 (finding 20).
- **Health:** explained variance > 0.3, KL ≤ 0.02, plus any stop rule specific to the arm.

## Configs

- Reuse an existing run as the control if its settings match (v9 used v7's runs). Copy its YAML, change only the arm, and show the `diff`.
- Seeds 50–52 as a sweep: `runs: [{config: arm.yaml, seeds: [50, 51, 52]}]`.
- Match the control's `games_per_worker` and `num_envs`. New controls use 64 (finding 24). Set `deterministic_collection: true` and `max_workers: 8`.
- Base `experiment.yaml` on v9's. For 5M screens, add `compare.steps`.

## Time

Use real runs, not profiler numbers. v9: about 900 steps/s per run at the start, about 450 by 10M as the snapshot pool fills; 3 × 20M took about 12 h of training plus about 40 min of analysis.

## Smoke test and hand over

- Copy v9's `smoke.yaml` and `smoke_experiment.yaml`, with smoke names that differ from the real runs' names.
- Validate each YAML with `load_config`. If code changed: add tests and run `make test lint typecheck`.
- Run the smoke test and check the report shows the arm's change.
- Add a README section and an unticked next-steps item.
- Commit when asked.
- Hand over: `catan experiment run experiments/vN_<name>/experiment.yaml`, then `catan experiment watch vN_<name>`.
