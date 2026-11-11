# Findings and hypotheses

What the experiments so far (v2–v5, see [experiments/README.md](../experiments/README.md)) have shown, what we think is going on, and what other Catan and multi-agent RL work suggests trying. Written after v5 stage 1, Nov 2026.

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
- *Try:* in stage 2 of v6, keep ~20% random seats alongside the frozen peers, and draw peers from several checkpoints.

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
