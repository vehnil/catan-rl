# Findings and hypotheses

What the experiments so far (v2–v7, see [experiments/README.md](../experiments/README.md)) have shown, what we think is going on, and what other Catan and multi-agent RL work suggests trying. Started after v5 stage 1, Nov 2026; findings 14–19 are from the v7 baseline, 20–23 from v8.

## Findings

1. **Beating random is easy.** Every learner reached 92–100% against 3× random within 200–500k steps (about 3–12 minutes with parallel workers). Games get shorter as it improves, from about 220 turns at 100k to 115 at 500k.
2. **Training against greedy makes the policy worse.** In v3 stage 2 (2 random + greedy) and v4 stage 3 (2 frozen PPO + greedy), the policy almost never won:
   - The win/loss reward was the same every game (a loss), so it carried no information.
   - The value head learned to predict "I lose".
   - Entropy drifted up (1.62 → 1.89 in v3, 0.86 → 1.3 in v4), meaning the policy was getting more random.
   - Play ended worse than where the stage started.
3. **The best policies against greedy never trained on it.** On the 200-game yardstick (plus 2 random + greedy), policies trained only on random scored 4.6–4.8 VP. That beats v4's PPO v2 at 4.3 VP (random, then self-play) and v3's stage 2 at 3.9 VP (trained against greedy). Wins are 2–3.5% for all of them, within each other's error bars.
4. **Pace is the bottleneck.** Greedy wins in about 65 turns in these lineups, while our best policies finish in about 95–120 even when they win. Until the policy can reach 10 VP about as fast as greedy, it can't beat greedy, and losing every game teaches it nothing (finding 2).
5. **Self-play against a single frozen copy overfits.** v4's PPO v2 beat frozen v1 92–100% of the time and won faster, but that didn't carry over to greedy. This matches the "competitive overfitting" reported elsewhere (see below).
6. **The speed bonus has no clear effect.** v5 A (with the bonus) and B (without) learned at the same pace against random and scored the same against greedy. B did finish last in the round-robin (2.8% vs A's 7.8%), so there may be a small effect against competent players.
7. **Double reward per VP looks strong against other PPO policies.** v5 C matched the others against random and greedy, but won 50.9% of the round-robin, nearly matching PPO v2 (56.6%), which trained three times as long. That's one seed; v6 tests it with four.
8. **Policies find exploits in hand-written opponents.** The first PPO round learned 1-for-2 trade offers that greedy accepted, because greedy only checked its own gain. Hardening greedy's acceptance closed it. Expect this again with any rule-based opponent.
9. **Training is limited by the game engine, not the network.** Python game logic dominates; the network is small. Parallel workers gave about 2.5× on the M1. Metal (MPS) is slower than CPU for both the update (0.48 s vs 0.31 s) and single-position inference (11×).

10. **Self-play against a population of peers, mixed with random, does carry over to greedy.** In v6 stage 2, each learner trained against its three frozen stage-1 peers plus about 25% random seats. All four improved against greedy (4.78–5.13 VP and 4–6.5% wins, the best so far), although greedy never appeared in training. v4's single-copy self-play didn't (finding 5). This supports H3.
11. **The overfitting curve shows only mild specialisation, starting around 400k steps.** Checkpoints every 200k were scored against the peers, held-out PPO policies, a greedy mix and 3× random on fixed seeds (`catan curve`). Held-out PPO scores rose from 2–12% to 25–48%, and 3× random stayed at 97–100%. Two of four learners (s1, s3) flattened against held-out PPO after about 400k while their peers score kept rising, which suggests stages of about 400–600k against a fixed population, then refreshing the population.
12. **Seed matters as much as settings.** In v6 stage 1, four seeds with identical settings spread as widely as v5's four different settings, and v5 C's round-robin lead (40%) didn't reproduce (8–19%). Compare settings over several seeds, never one.

13. **The failed greedy stages had healthy update sizes but almost nothing to learn from** (`catan health`, run after the fact). In v3 stage 2 and v4 stage 3:
    - KL (about 0.01) and clip fraction (0.10–0.14) were normal, the same as successful runs.
    - Value loss fell to 0.001–0.003, a third or less of a successful run's (about 0.008 in v6 stage 2), while returns sat near −0.8.

    The value head learned "I lose" accurately, so the advantages were tiny. Per-rollout advantage normalisation then scales that near-noise up to full-size updates, which fits the policy drifting rather than improving: v4 stage 3's entropy rose from 1.11 to 1.44. Explained variance wasn't logged then; it is now, so future runs can confirm this directly.

14. **The v7 baseline beats v6 stage 2 on the comparison set before the run.** v7 is continuous self-play: 50% the current policy, 30% a pool of up to 40 past snapshots, 20% fixed opponents (random, `greedy:epsilon=0.5`, greedy). It ran 20M steps × 3 seeds with linear learning-rate decay. Final policies, 400 games each, mean with 95% interval over seeds:

    | Lineup | Metric | v7 (3 seeds) | v6 stage 2 (4 seeds) |
    |---|---|---|---|
    | 3× greedy | VP margin | −5.61 [−5.95, −5.27] | −6.63 [−6.84, −6.43] |
    | 3× greedy | Mean rank | 3.47 [3.41, 3.53] | 3.75 [3.68, 3.83] |
    | 3× greedy | Win rate | 1.8% [1.1, 2.6] | 0.1% |
    | 3× greedy | Mean VP | 4.44 | 3.46 |
    | 2 random + greedy | VP margin | −4.28 [−4.83, −3.73] | −5.26 [−5.45, −5.07] |
    | 2 random + greedy | Win rate | 9.5% | 3.4% |

    Every interval is clear of v6's. It still trails the best greedy player by about 5.6 VP.

15. **Against 3× greedy, v7 now places second; v6 rarely did.** 400 games per final policy:

    | Policy | 1st | 2nd | 3rd | 4th |
    |---|---|---|---|---|
    | v7 s1 | 1.8% | 12.8% | 23.0% | 62.5% |
    | v7 s2 | 2.8% | 11.8% | 20.8% | 64.8% |
    | v7 s3 | 1.5% | 13.0% | 24.5% | 61.0% |
    | v6 stage 2 (mean of 4) | 0.3% | 5.8% | 15.6% | 78.4% |

    The gain is mostly from last place to second and third. Wins stay rare.

16. **The reference rating climbs steadily through the whole run.** It's computed against the fixed reference set (400 games per checkpoint, every 2M):
    - It rose about 1.5 points per 1M steps, from 16–20 at 2M to a peak of 31.7–34.5 at 16–18M. The finals are 28.7–30.8.
    - All three seeds passed v6 s4 (26.0) and `greedy:epsilon=0.5` (27.3) between 8M and 14M.
    - They stay well below `greedy:epsilon=0.2` (41.8) and greedy (50.3).
    - All three seeds dropped 2–3 points from peak to final, while the learning rate was near zero and KL was about 0.001, so the policies barely changed. Differences between ratings below about 3 points should therefore be treated as noise, even though the reported uncertainty is about ±1.8.

17. **PPO stayed healthy for 20M steps, unlike the failed greedy stages (finding 13).**
    - Explained variance held at 0.83–0.87 from start to finish.
    - Entropy fell smoothly from 1.57 to 1.19.
    - Value loss *rose* from 0.004 to 0.012 instead of collapsing, so the value head kept facing outcomes it couldn't predict yet.
    - KL and clip fraction shrank to about zero along with the learning rate.

    Mixing in the current policy and past snapshots keeps most games winnable, about a 25–28% training win rate, so the learning signal never died out.

18. **Against greedy, progress per step is slow, and 100-game evaluations are noisy.**
    - On the 1M-checkpoint evaluations (100 games against 3× greedy), the mean margin over three seeds improved about 0.08 VP per 1M steps: −7.09 at 1M, −6.26 at 10M, −5.54 at 20M.
    - Mean rank improved from 3.84 to 3.36.
    - Single checkpoints moved by ±0.3–0.5 VP. In one case all three seeds jumped together at 18M and fell back at 19M.
    - The reference rating (finding 16) rises much more clearly than the margin against 3× greedy over the same checkpoints.

19. **Speed fell by a third as the snapshot pool grew.**
    - Per-seed throughput was about 650 steps/s early in the run and about 450 by the end.
    - The machine load was unchanged, so the drop followed the pool filling toward 40 snapshots: more distinct opponent models means smaller batched-inference groups per worker.
    - Earlier, the encoding and batching work (perf A+B) gave 1.69× on a controlled re-run of v6 stage 2: 288 → 505 steps/s per learner.

20. **Greedy as an opponent: equal strength isn't a margin of 0.** Greedy against 3× greedy (400 games) scores a VP margin of −2.18, mean rank 2.35 (ties lower it from 2.5), a 25% win rate and 7.26 VP, in games of 81.9 turns. So v7's −5.61 is a gap of about 3.4 VP to greedy's level, not 5.6. Games with only greedy players also last as long as v7's games against 3× greedy (72–81 turns): game length is set by greedy and says nothing about the learner's pace.

21. **v7 had plateaued: 5M more steps at a live learning rate made it slightly worse** (v8 control). Continuing each v7 final under v7's own setup at a constant learning rate of 1e-4 lost ground in all three seeds: −0.20 VP on average against 3× greedy and −0.51 in the 2 random + greedy lineup (400 games, same games for every policy). The reference rating stayed flat within its ±3 noise. Health was normal throughout. v7's late gains had already stopped (finding 16), and its learning-rate decay to 0 wasn't hiding further progress; restarting the rate made the policy drift instead.

22. **Training against greedy more often doesn't help against greedy** (v8 greedy-heavy). Raising greedy seats from about 7% to 24% (plus 12% `greedy:epsilon=0.5`, no random) for 5M steps gave no detectable difference from the control: paired by seed, +0.16, −0.24 and +0.19 VP against 3× greedy (mean +0.04). Rank, win rate and mean VP matched the control too. The hypothesis that v7 was held back by rarely meeting greedy is not supported.

23. **With self-play in the mix, training against greedy no longer breaks the policy.** Unlike v3 and v4 (findings 2 and 13), the greedy-heavy arm kept a 13–15% training win rate once snapshots existed (8% before), value loss stayed at about 0.007, and the policy didn't get worse than the control. Having most seats still be the learner's own copies keeps the reward informative, but it doesn't make greedy teachable either.

24. **The M1 tops out at about 8 rollout workers, and Python game code dominates each step** (profiled 12 Nov 2026, one `_BatchActor` with v7's settings and 32 games, the v7 s1 final as the learner).
    - Aggregate throughput with N workers at once: 1 → 1,221 steps/s, 4 → 3,315, 6 → 3,708, **8 → 4,243, 12 → 4,216**. v8 ran 12 workers (6 runs × 2) and gained nothing from the last 4, while using most of the 16 GB of memory.
    - The snapshot pool costs less than finding 19 suggested: 14% alone (1,233 → 1,056 steps/s at pool 0 vs 40) and 20% with 8 workers at once (5,052 → 4,019). Bigger pools mean smaller batches: the learner's mean batch falls from 18 games to 4.
    - Where a worker's time goes (cProfile, pool 10): feature encoding about 19% (3.3 encodes per learner step, since each network opponent encodes its own view), observations and belief tracking about 17%, the rules engine 15–20%, greedy's move search about 10%, network inference about 12%. About 85% is Python game, feature and belief code.
    - **`games_per_worker: 64` is about 1.35–1.4× faster than v7/v8's 32** (8 workers at once): 4,303 → 5,731 steps/s at pool 10, and 3,816 → 5,436 at pool 40. The learner's mean batch jumps from about 9–10 games to 33. 128 games gives no more (5,701 / 5,552). At 64 the pool size barely matters (−5% from 10 to 40), so no cap is needed.
    - For long runs on this machine: keep the total at about 8 workers (e.g. 4 runs × 2) with 64 games per worker. Note that changing `games_per_worker` changes the collected data, so an arm compared against an earlier run should keep that run's setting.

## Hypotheses, and what other work suggests

**H1. Rewards that still carry information when losing.** Our reward is ±1 at the end plus a small amount per VP. Against stronger opponents nearly every game is −1, so learning stops (findings 2 and 4).
- Gendre & Kaneko, the only RL agent we found that beats a strong Catan bot, used ±0.75 for a win or loss plus 0.02 per VP of difference at the end.
- The UMD Catan project found dense rewards necessary for long games.
- For four players, a rank-based reward would let the policy learn to come second rather than fourth: +1, +⅓, −⅓, −1 for first to last, or VP margin over the leader.
- *Try:* a rank or margin reward, compared against plain win/loss.

**H2. Raise opponent strength in small steps.** Jumping from random to full greedy is too big a step (finding 2).
- The Pommerman work used a curriculum of rule-based opponents, then a population with Elo-based matchmaking.
- *Try:* opponents of `greedy:epsilon=0.8 → 0.5 → 0.2 → 0`, moving up a step once the win rate passes about 40%.

**H3. Mix fixed opponents into self-play, and play against many past selves.** This addresses finding 5.
- Territory Paint Wars (competitive PPO) found that self-play win rate stays near 50% while win rate against a fixed opponent collapses, from 73.5% to 21.6%. Playing 20% of episodes against a fixed random policy restored it to 77%.
- Gendre & Kaneko found that training against one fixed strong copy learned poorly and overfit. They used a rolling spread of past checkpoints instead (fictitious self-play).
- The AlphaStar-style league adds prioritised sampling, which favours opponents the learner struggles against (PFSP).
- *Tried* in v6 stage 2 (frozen peers plus ~25% random seats): it worked (findings 10 and 11). *Tried* in v7 (current policy plus up to 40 past snapshots plus 20% fixed opponents): it beat v6 stage 2 (findings 14–17).

**H4. A network that knows the board's layout.** Our flat MLP has to work out which hexes touch which vertex from index positions alone, and the board changes every game.
- Gendre & Kaneko's key result came from encoding the board for a CNN that follows the hex layout, combined with a vector branch for non-board information.
- *Try:* cheap per-vertex features first (pips per resource, legal spots, cards missing per build), then a graph or hex-layout network.

**H5. Evaluate against both fixed baselines and past selves.** The UMD project's self-play results looked fine until they evaluated against fixed baselines, which revealed overfitting. We already run the greedy yardstick and round-robins. Keep doing both, every stage.

**H6. Shrink the action space.** About 230 of our 488 actions are trade proposals. Neither Gendre & Kaneko's agent nor the UMD agent trades at all.
- *Try:* a no-trading environment (`rules.max_trade_proposals_per_turn: 0`) to see whether trading helps or just adds noise.

**H7. PPO details.**
- The "37 implementation details" survey normalises advantages per minibatch; we normalise per rollout.
- Gendre & Kaneko added an L2 penalty on the policy's raw action scores so rarely legal actions don't drift toward zero probability; without it, their training was unstable.
- Both are small, testable changes.

**Background:** Pfeiffer (2004) found that pure Q-learning in Catan failed to beat simple baselines. Every successful attempt we found uses domain structure (H4), dense rewards (H1), or simplified rules (H6).

## Sources

- Gendre & Kaneko, *Playing Catan with Cross-dimensional Neural Network*, 2020: <https://arxiv.org/abs/2008.07079>
- Aggarwal et al., *Learning to Settle: Reinforcement Learning in Catan* (UMD): <https://research.aniaggarwal.com/catan-rl>
- *Territory Paint Wars: Diagnosing and Mitigating Failure Modes in Competitive Multi-Agent PPO*, 2026: <https://arxiv.org/abs/2604.04983>
- *Multi-Agent Training for Pommerman: Curriculum Learning and Population-based Self-Play*, 2024: <https://arxiv.org/abs/2407.00662>
- Huang et al., *The 37 Implementation Details of Proximal Policy Optimization*, ICLR blog track 2022: <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/>
- Pfeiffer, *Reinforcement Learning of Strategies for Settlers of Catan*, 2004: <https://core.ac.uk/download/43714282.pdf>
