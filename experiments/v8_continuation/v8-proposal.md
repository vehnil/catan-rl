# v8 proposal: greedy-heavy continuation vs control

Written before the run, 12 Nov 2026. Background: [findings](../../notes/findings.md) 14–19 and [drawing-board](../../notes/drawing-board.md).

## Question

Is v7 stuck against greedy because it rarely trains against greedy?

v7 beat v6 stage 2 on every pre-registered metric, but against 3× greedy it still trails by 5.6 VP and wins 1.8%. Its reference rating rose about 1.5 points per 1M steps, while its margin against 3× greedy rose only 0.08 VP per 1M: the gains came mostly from beating weaker players. Games against 3× greedy lasted 72–81 turns throughout training. Four greedy players also take about 82 turns (below), so game length is set by greedy and isn't a measure of the learner's pace. What did change: the learner reached more VP (3.2 → 4.4) by the time greedy won.

## Hypothesis

v7's training opponents were 50% the current policy, 30% past snapshots and 20% fixed (random / `greedy:epsilon=0.5` / greedy, drawn uniformly). Full greedy filled only about 7% of seats. The learner mostly played slow copies of itself, which never punish slow play, so nothing pushed it to race to 10 VP.

**If so, raising greedy-style seats to about 36% will improve the margin against 3× greedy more than the same number of steps under v7's mix.**

## Design

Two arms, three seeds each. Every run continues from the same seed's v7 final for 5M steps.

| | Control | Greedy-heavy |
|---|---|---|
| Starts from | v7 final, same seed | v7 final, same seed |
| Opponent seats | 50% current / 30% snapshots / 20% fixed (≈7% greedy, 7% `ε=0.5`, 7% random) | 40% current / 24% snapshots / 36% fixed (24% greedy, 12% `ε=0.5`, no random) |
| `fixed_opponents` | `[random, greedy:epsilon=0.5, greedy]` | `[greedy, greedy, greedy:epsilon=0.5]` |
| `latest_prob` / `self_play_prob` | 0.5 / 0.6 | 0.4 / 0.4 |

Shared settings:
- 5M steps, constant learning rate 1e-4 (v7 ended at 0), otherwise v7's PPO settings.
- Seeds 50–52 (the v7 seeds); `deterministic_collection: true`, so each run is reproducible.
- 200-game evaluations against 3× greedy every 1M steps, and a checkpoint every 1M.
- Run with `catan experiment run experiments/v8_continuation/experiment.yaml`.

Pairing each arm with the same seed's starting policy removes most of the seed-to-seed spread that made v6 and v7 hard to read. The control isn't just "v7 again": restarting the learning rate might help on its own, and the control measures that.

## What it tests

1. **Opponent mix:** greedy-heavy vs control, paired by seed.
2. **Steps with a live learning rate:** control vs the v7 finals. If the control gains clearly, v7 was cut off by its learning-rate decay, not by its recipe.
3. **The finding 2 risk:** training against greedy made earlier policies worse (v3, v4), because they lost every game. Here most seats are still the learner's own copies, so training games should stay winnable.

## Comparison, set before the run

**Primary:** VP margin against 3× greedy, 400 games per final policy (`compare`, seed 6000). For each seed, take greedy-heavy minus control.
- *Greedy-heavy is better* if all three paired differences are positive **and** their mean is at least +0.3 VP.
- *Worse* if all three are negative and the mean is at most −0.3 VP.
- Otherwise: no detectable difference.

With three seeds a t-interval is very wide, so this sign-and-size rule is fixed up front instead. 0.3 VP is roughly the per-policy noise of a 400-game evaluation.

**Secondary:**
- Mean rank and win rate against 3× greedy.
- The learner's mean VP against 3× greedy (its VP when the game ends, which is when greedy wins).
- The reference rating (`reference_v1`), every 1M steps.
- VP margin in the 2 random + greedy lineup.
- Control vs the v7 finals, on the same margin rule (question 2).

**Reference point, measured before the run:** greedy vs 3× greedy, 400 games (`catan sim`, seed 7000, run `d362692c0721`). An agent exactly as strong as greedy scores a **margin of −2.18**, mean rank 2.35, a 25% win rate and 7.26 VP, in games of 81.9 turns. Against that, v7's −5.61 is a gap of about **3.4 VP**, not 5.6.

**Health:** explained variance above about 0.3, KL at or below about 0.02. Stop the greedy-heavy arm if its training `win100` stays below 0.10 for 1M steps, or if value loss falls toward 0.001 (the "I lose" collapse of finding 13).

## Outcomes and what comes next

| Result | Reading | Next |
|---|---|---|
| Greedy-heavy better | The opponent mix was the bottleneck | Tune the mix: PFSP, or the epsilon ladder; then the rank reward |
| No difference, control gains over v7 | Steps still help once the learning rate is live | Longer runs; consider cloud compute |
| No difference, control flat | Neither steps nor opponents are the bottleneck | The board-aware network (H4) becomes the main line |
| Greedy-heavy worse | Too much greedy too soon (finding 2) | Raise greedy's share in smaller steps (the epsilon ladder) |

## Cost

6 runs at once (12 workers on the M1's 8 cores). Estimated beforehand at 8–10 hours plus about an hour of analysis. **Actual: 3 h 8 min of training (15:41–18:50) and 33 min of analysis**, with no crashes or resumes. Each run got 530–660 steps/s, faster than the estimate assumed, because a 5M-step run only fills about 10 snapshots.

## Results

Run 12 Nov 2026, code `aa62343`. Full report: `data/experiments/v8_continuation/report.md`.

**Verdict: no detectable difference between the arms, and neither arm improved on v7.** This is the "neither arm moves, control flat" row: neither more steps nor more greedy opponents improve this setup, so the main line moves to the network (H4).

VP margin against 3× greedy, 400 games per final, same games for every policy:

| Seed | v7 final (start) | Control | Greedy-heavy | Greedy − control | Control − v7 |
|---|---|---|---|---|---|
| s1 | −5.70 | −5.88 | −5.73 | +0.16 | −0.18 |
| s2 | −5.45 | −5.73 | −5.96 | −0.24 | −0.28 |
| s3 | −5.67 | −5.82 | −5.63 | +0.19 | −0.14 |
| **Mean** | **−5.61** | **−5.81** | **−5.77** | **+0.04** | **−0.20** |

- **Primary (greedy-heavy vs control):** the paired differences have mixed signs, with a mean of +0.04. That's no detectable difference by the rule set beforehand.
- **Control vs v7 finals:** all three seeds lost ground against 3× greedy (−0.20 on average, short of the −0.3 "worse" bar). In the 2 random + greedy lineup all three lost 0.44–0.58 VP (mean −0.51), which meets the "worse" rule. 5M more steps at a live learning rate made the policy slightly worse, not better.
- **Reference rating:** flat. The starts were 28.7–30.8; the finals range from 28.5 to 33.3 with no pattern by arm. Checkpoints in between wander within about ±3 points, which is the noise level (finding 16).
- **Rank, win rate, mean VP against 3× greedy:** no differences between arms (rank 3.51–3.52, wins 1.2%, VP 4.26–4.30), and all slightly below v7 (3.47, 1.8%, 4.44).
- **Health:** normal throughout in both arms (explained variance 0.84–0.87, KL about 0.005). The greedy arm's training win rate was 0.08 for the first 500k (before any snapshots existed), then 0.13–0.15. The stop rule never triggered, and value loss stayed at about 0.007, with no "I lose" collapse.

**What it answers:**
1. *Opponent mix:* tripling the share of greedy seats (7% → 24%, plus 12% `ε=0.5`) for 5M steps didn't improve play against greedy. Exposure to greedy isn't what limits the learner.
2. *Steps with a live learning rate:* no gain, and a small loss. v7 had already plateaued by its end. Its decay to a learning rate of 0 wasn't hiding further progress; if anything, the anneal settled the policy, and restarting at 1e-4 made it drift.
3. *Finding 2 risk:* didn't repeat. With most seats still the learner's own copies, training against greedy neither helped nor broke the policy.

**Caveat:** 5M steps is a quarter of v7. A slow effect, at v7's own rate of about 0.08 VP per 1M, would be about +0.4 VP over 5M and might not show yet. But the control should then have gained too, and it lost ground instead.
