# v8 proposal: greedy-heavy continuation vs control

Written before the run, 12 Nov 2026. Background: [findings](../../notes/findings.md) 14–19 and [drawing-board](../../notes/drawing-board.md).

## Question

Is v7 stuck against greedy because it rarely trains against greedy?

v7 beat v6 stage 2 on every pre-registered metric, but against 3× greedy it still trails by 5.6 VP and wins 1.8%. Its reference rating rose about 1.5 points per 1M steps, while its margin against 3× greedy rose only 0.08 VP per 1M: the gains came mostly from beating weaker players. Games against 3× greedy lasted 72–81 turns from start to end of training, so the learner never sped up; it just reached more VP (3.2 → 4.4) before greedy won.

## Hypothesis

v7's training opponents were 50% the current policy, 30% past snapshots and 20% fixed (random / `greedy:epsilon=0.5` / greedy, drawn uniformly). Full greedy filled only about 7% of seats. The learner mostly played slow copies of itself, which never punish slow play, so nothing pushed it to race to 10 VP.

**If so, raising greedy-style seats to about 36% will improve the margin against 3× greedy, and shorten games against it, more than the same number of steps under v7's mix.**

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
- Game length against 3× greedy (the pace hypothesis predicts it falls for greedy-heavy).
- The reference rating (`reference_v1`), every 1M steps.
- VP margin in the 2 random + greedy lineup.
- Control vs the v7 finals, on the same margin rule (question 2).

**Reference point:** greedy vs 3× greedy (400 games), measured before the run, gives the margin, rank and win rate of an agent exactly as strong as greedy.

**Health:** explained variance above about 0.3, KL at or below about 0.02. Stop the greedy-heavy arm if its training `win100` stays below 0.10 for 1M steps, or if value loss falls toward 0.001 (the "I lose" collapse of finding 13).

## Outcomes and what comes next

| Result | Reading | Next |
|---|---|---|
| Greedy-heavy better, games shorter | The opponent mix was the bottleneck | Tune the mix: PFSP, or the epsilon ladder; then the rank reward |
| Greedy-heavy better, games not shorter | Greedy seats help, but not through pace | Look at what changed (replays, build order) before tuning further |
| No difference, control gains over v7 | Steps still help once the learning rate is live | Longer runs; consider cloud compute |
| No difference, control flat | Neither steps nor opponents are the bottleneck | The board-aware network (H4) becomes the main line |
| Greedy-heavy worse | Too much greedy too soon (finding 2) | Raise greedy's share in smaller steps (the epsilon ladder) |

## Cost

6 runs at once (12 workers on the M1's 8 cores). Expect about 8–10 hours, plus about an hour of analysis; `catan experiment watch v8_continuation` shows the real ETA.

## Results

Not run yet.
