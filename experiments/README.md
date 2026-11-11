# Experiments

YAML configs for `catan train` (PPO) and `catan tournament`, grouped by version. Each version folder is one line of work.

- Once a version has been run, leave its configs as they were, so the results below stay reproducible.
- To change an approach, start a new folder with the next number rather than editing an old one.
- Results live in `data/` (gitignored), keyed by the run ids and policy ids recorded here.

```bash
catan train experiments/v1_examples/ppo_small.yaml
catan tournament experiments/v1_examples/baselines.yaml
```

Config fields and defaults are in `src/catan/config/models.py`. Unknown keys are rejected. See [docs/50-training.md](../docs/50-training.md) for the training loop.

## Versions

### v1_examples: starting examples

The configs the project shipped with. `baselines` checks that greedy beats random, `ppo_small` is a 20k-step smoke run of the training loop, `ppo_league` is a long self-play run, `strategies_round_robin` compares the greedy presets and bandits, and `search_gauntlet` runs ISMCTS against greedy variants.

### v2_first_curriculum: first real PPO run (Nov 2026)

Two stages, single process, run before greedy's trade acceptance was hardened.

| Stage | Opponents | Steps | Run → policy | Result |
|---|---|---|---|---|
| `stage1_random` | random, 20% self-play | 500k | `d1f641b555e0` → `8a19bcdc50` | 97.5% vs 3× random |
| `stage2_greedy` | random/greedy pool, 40% self-play | 1M | `0657e3d0ddb5` → `9a4a93bf0a` | 0% vs 3× greedy at every eval |

What it showed:
- The policy learned to exploit greedy's old acceptance rule with 1-for-2 offers.
- It almost never built cities.
- Its few wins came in very long games.

That led to hardening greedy's trade acceptance and to the plan for behaviour cloning. These configs ran against the old greedy; re-running them today plays the hardened one.
