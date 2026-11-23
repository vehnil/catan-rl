---
name: experiment-results
description: Review a finished catan-rl experiment against its pre-set comparison rule and write up the results.
---

# Experiment results

## Gather (from files)

- `data/experiments/<name>/report.md`.
- `compare.json`: `points` lists the policy ids per group and is in seed order, so pair by position. `rows` gives `vp_margin`, `mean_rank`, `win_rate` and `mean_vp` per policy and lineup.
- `ratings.json`. Earlier experiments' ratings of the same policies are on the same scale.
- The proposal's comparison rule. Apply it as written.

## Review

- Lead with the verdict under the rule, then a per-seed table (control, arm, diff, mean), then the secondary results and health.
- Say which outcome row of the proposal the result lands in, and the caveats (resumes, noise).
- To check whether the arm learned faster, use `compare.steps` points. If there are none, offer:
  `catan compare --group arm5M=<ids> --group ctl5M=<ids> --vs greedy3=greedy,greedy,greedy --vs mix=random,random,greedy --seed 6000`
- In-training evals are only a hint (finding 18).

## Write up (when the user agrees)

- **Proposal:** a `## Results` section and the actual cost.
- **`experiments/README.md`:** the results line.
- **notes:** use the `notes` skill (new findings, the header, the "Tried" line, corrections, the ticked next-steps item).
