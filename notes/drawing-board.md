# Why we may need to go back to the drawing board

Written halfway through v7 (10M of 20M steps), revised at 11M. See [findings](findings.md) for the earlier experiments, [next-steps](next-steps.md) for the plan and [experiments/README.md](../experiments/README.md) for v7's comparison set before the run.

## The concern

Before v7 the question was whether our setup was even good RL practice. The critique found the main faults: runs far too short (0.5–1M steps), no plain self-play baseline, noisy small-sample conclusions, and no PPO health metrics. Phases 1–4 fixed the methodology, and v7 is the conventional baseline: self-play against the current policy plus a snapshot pool plus 20% fixed opponents, 20M steps × 3 seeds, with all three running at once.

The bar we set beforehand: **if v7 doesn't yield great benefits, we go back to the drawing board on both performance and training mechanisms.** v7 is the test of "is the problem just that we never trained long enough?"

## What v7 shows at 10M

PPO itself is healthy, so the weak learning is not a broken optimiser:
- Explained variance is 0.8–0.9.
- KL is about 0.005–0.008.
- Clip fraction is about 0.07–0.10.
- Entropy is falling slowly (about 1.2–1.3 now).
- No crashes since the resume at 1M.

The learner is **not** losing every training game. `win100` in the training logs is about 0.22–0.35, so the learner wins a quarter to a third of the games it trains on. The "always loses, so the reward says nothing" failure of v3/v4 ([findings](findings.md) 2, 13) doesn't apply here.

Learning against 3× greedy is real but slow. Results are VP margin (own VP minus the best opponent's), from the 100-game in-training evals:

| Seed | 1M | 2M | 3M | 4M | 5M | 6M | 7M | 8M | 9M | 10M |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | −7.16 | −6.89 | −6.81 | −6.72 | −6.62 | −6.25 | −6.38 | −6.46 | −6.75 | −6.56 |
| s2 | −7.22 | −7.00 | −6.87 | −6.86 | −6.76 | −6.60 | −6.35 | −5.85 | −6.05 | −5.95 |
| s3 | −6.90 | −6.88 | −6.76 | −6.60 | −6.83 | −6.32 | −6.23 | −6.50 | −6.38 | −6.26 |

- Mean rank has moved from about 3.8–3.9 to 3.5–3.7.
- Win rate is 0–1% at every checkpoint.
- The best score so far is about 3.9 VP, slightly above v6's best (3.5–3.6), but nowhere near beating greedy.
- Games against 3× greedy end at about **74–76 turns**, with the learner on about 3.5 VP. Greedy finishes first; the learner is too slow, the same pace problem as [findings](findings.md) 4.
- Each point carries about ±0.3 VP of noise. 100 games is half what the protocol asks for a curve checkpoint (200), so these evals are a progress check, not evidence. Trends matter more than single checkpoints.

## Compute against gain

The clearest number so far: **v6 stage 2 trained for about 1–2M steps in total; v7 has spent 5–10× that and is about +0.3 VP better.** This doesn't depend on the projection below or the learning-rate schedule. Unless the second half is much better than the first, v7 has shown that steps alone buy very little in this setup.

## The linear projection to 20M

A straight-line fit through the 1M checkpoints (fitting from 1M or from 3M gives the two ends of each range):

| Seed | Slope per 1M | Projected margin at 20M | Projected rank at 20M |
|---|---|---|---|
| s1 | +0.02 to +0.06 | −5.8 to −6.3 | 3.6–3.7 |
| s2 | +0.15 to +0.16 | −4.3 to −4.4 | 3.1–3.2 |
| s3 | +0.07 | −5.5 to −5.6 | 3.4 |
| **Mean** | **about +0.09** | **about −5.3** | **about 3.4** |

- At this rate, **1 VP of margin costs about 10M steps,** about 6 hours with three seeds at about 440 steps/s each.
- The projection is probably *optimistic*. The learning rate decays linearly to 0, so the second half usually learns more slowly than the first.
- **The end point isn't known.** We have no greedy-vs-3×-greedy baseline. An agent exactly as strong as greedy would still have a negative margin against the best of three opponents, and win about 25%. So "close the gap" doesn't mean reaching a margin of 0, and the earlier "60M+ steps" estimate is unanchored until that baseline is measured (see [next-steps](next-steps.md)).
- The seeds may disagree: s2 improves about twice as fast as the others. s1 "stalling since 6M" (−6.25 → −6.56) is within the ±0.3 noise, though, so this is *possibly* a fragile setup, not shown to be one. 200-game evals of the s1 checkpoints would settle it.

## Why that points to the drawing board

1. **More steps alone isn't the answer.** v7 was the "just train longer, conventionally" test. It helps, but the gain per step is too small (see "Compute against gain").
2. **Throughput is capped, and optimising our code can't fix that.** The logs show about 430–440 steps/s per seed (cumulative since the 1M resume), down from 650–870 early in the run, and well below the 550–610 we expected. More distinct snapshot opponents mean smaller inference batches. The full 20M run takes over 10 hours, not 7–9. Observation, belief and feature code is about half of each step, so even making it free gives at most 2×. A 5× or 10× increase in steps means more cores (cloud), not code work.
3. **The training opponents don't look like greedy.** 50% of seats are the current policy and 30% are past snapshots. The 20% fixed slot is split three ways (random / `greedy:epsilon=0.5` / greedy), so **full greedy is only about 7% of training seats**, and random is another 7%. The learner mostly plays slow copies of itself, which never punish slow play, so nothing pushes it to race to 10 VP the way greedy does. This fits both the slow progress against greedy and the 74–76-turn evals.
4. **Credit is spread thin.** The learner wins often enough in training, but the reward is ±1 at the end plus 0.05 per VP, over about 260 learner decisions per game (γ = 0.995). Which of those moves mattered is hard to tell from the outcome alone. This is the case for a denser reward (H1). It is about credit assignment, not about the learner always losing.
5. **The model has no knowledge of the board.** The flat MLP has to learn Catan's geometry (adjacency, pips, distance to build spots) from scratch, from a flat feature vector.

## How to read v7's final result

v7's verdict comes from the comparison in [experiments/README.md](../experiments/README.md), set before the run, not from the mid-run projection above:
- **Primary:** the reference rating (`catan rate`, 400 games) of the three finals, mean and 95% interval over seeds, plus a rating at each 1M checkpoint to see the curve.
- **Secondary:** VP margin and mean rank against 3× greedy (`catan compare`, 400 games) against v6 stage 2's four finals. *Better* means v7's three-seed interval lies entirely above v6's.
- **Health:** explained variance above about 0.3, KL at or below about 0.02, and the rating rising across checkpoints.

**This bar is lower than the goal.** The bar at the top of this note asks for "great benefits", but v6's best is already within about 0.4 VP of v7's. v7 could pass the margin test and still be nowhere near greedy. Decide now: **a narrow pass still means going back to the drawing board.** Passing only confirms the recipe is better than v6; it doesn't make it good enough.

**This note overrides one line of the plan.** The README says that if the rating plateaus, "the next step is longer runs or PFSP opponent sampling, not new settings". That was written assuming more steps were affordable. Point 2 shows they aren't on this machine, so the table below allows setting changes (opponent mix, reward, network) in row 2. Longer runs stay the answer only if row 1 holds *and* cloud compute is available.

**If it falls short, the curves say which drawing board to go back to.** Even a disappointing result isn't a dead end: it lands in one of these rows.

| What we see | What it means | Where to go next |
|---|---|---|
| Rating still rising at 20M, but not past v6 yet | Under-trained: the recipe works but needs more steps | **Performance.** Only more cores (cloud) gets 5–10× more steps; code work caps at 2× (point 2) |
| Rating flattens early, health normal (explained variance > 0.3, small KL) | PPO works but has learned all it can from this setup | **Training design:** opponent mix first (cheapest), then the reward (rank or margin), the network (board-aware features or a graph network), the action space (trading on or off) |
| Rating flattens, health bad (explained variance collapsing, entropy drifting) | The optimisation itself is struggling | **PPO fixes:** learning rate and entropy, `target_kl`, separate policy and value networks |
| Beats v6 against PPO opponents but not against greedy | Self-play is too insular | **Opponents:** more greedy-style seats, the epsilon ladder, PFSP |

At 10M, health is normal and the curves are rising slowly but unevenly across seeds. That points between rows 1 and 2. Point 3 makes row 4 more likely than it first looked; the reference rating's split between greedy-style and PPO reference opponents will show it.

## What to measure before the verdict

- **Greedy vs 3× greedy** (400 games): margin, rank and win rate for an agent of greedy's own strength. This is the end point for every "gap to greedy" number.
- **Pace:** game length against 3× greedy, and the learner's VP at the moment greedy wins, at each checkpoint. Pace was the bottleneck in [findings](findings.md) 4 and the evals suggest it still is.
- **Value head away from self-play:** explained variance on the greedy eval games. 0.8–0.9 in training may only mean it predicts outcomes against itself.
- **The s1 checkpoints at 200 games**, to see whether s1 really stalled.

## Training design options

Ordered roughly by cost. From the earlier [findings](findings.md):
- **Opponent mix (new, config only):** drop random from the fixed pool, raise greedy and `greedy:epsilon=0.5` to a larger share of seats (e.g. 30–40%). It tests point 3 directly and costs nothing to build.
- **H2, the epsilon-greedy ladder:** opponents `greedy:epsilon=0.8 → 0.5 → 0.2 → 0`, so difficulty rises in steps.
- **H1, a rank or margin reward** (Phase 5, v8): spreads credit more densely than ±1 at the end (point 4).
- **PFSP:** prioritised opponent sampling, favouring snapshots the learner loses to.
- **H6, a no-trading environment:** a smaller action space and simpler games.
- **H4, a board-aware network** (Phase 6, v9): per-vertex features first, then a graph or attention model. Possibly separate policy and value networks. The biggest build, and probably the biggest gain (point 5).
- **Cloning from greedy:** deprioritised, because the project prefers learning from the ground up.

**Screen cheaply before committing a night.** A full arm (3 seeds × 20M) takes more than 10 hours, so v8 as planned is two nights or more. Instead, run each new arm to 5M and compare it with v7's own 5M checkpoints, which already exist. Only arms that beat v7 at 5M get the full 20M. Config-only arms (opponent mix, epsilon ladder, reward) can be screened this way while H4 is being built.
