# v9 proposal: rank reward vs win/loss

Written before the run, 12 Nov 2026. Background: [findings](../../notes/findings.md) 14–24 and the [v8 results](../v8_continuation/v8-proposal.md#results).

## Question

Is the learner held back by its learning signal? Against strong opponents, the win/loss reward treats nearly every game as the same −1.

v8 ruled out the opponent mix: tripling greedy's share for 5M steps changed nothing, and 5M more steps under v7's own mix made the policy slightly worse. v7 had plateaued. The two remaining suspects are the learning signal and the model. The signal is far cheaper to test, so it goes first.

## Hypothesis

Against 3× greedy, v7 loses about 98% of games. With the ±1 reward, finishing 2nd at 9 VP and 4th at 3 VP both pay −1. Only the 0.05-per-VP shaping separates them, and that's at most 0.5 over a whole game. So in most strong-opponent games the end-of-game reward carries no information about which play was better.

**If that's the bottleneck, paying the end-of-game reward by finishing place will train a policy with a better margin against 3× greedy than v7, at the same number of steps.**

## Design

Three runs, one per v7 seed. Each is v7's config exactly, trained from scratch for 20M steps. The only change is the end-of-game reward:

| | v7 (control) | v9 rank |
|---|---|---|
| End-of-game reward | Win +1, otherwise −1 | 1st +1, 2nd +⅓, 3rd −⅓, 4th −1 |
| Finishing place | n/a | The winner is 1st, the rest by total VP; tied seats share the average place |
| Games cut off without a winner | 0 | 0 |
| VP shaping | 0.05 per VP | 0.05 per VP (unchanged) |

Shared settings:
- v7's PPO and league settings: learning rate 2.5e-4 decaying linearly to 0 at 20M, and opponent seats 50% current policy, 30% snapshots, 20% fixed (random / `greedy:epsilon=0.5` / greedy).
- Seeds 50–52, `games_per_worker: 32`, 2 workers per run.
- `deterministic_collection: true`. This fixes how the workers' shares are merged, so the run is reproducible. It doesn't change what is collected.
- New code: `ppo.terminal_reward: rank`. The default, `win`, leaves every earlier config unchanged.
- Run with `catan experiment run experiments/v9_rank_reward/experiment.yaml`.

**The control is v7 itself:** same seeds, settings and opponents. No control runs are needed. Its finals and its 1M-step checkpoints are already stored, so the comparison is paired by seed at every matched step.

## What it tests

1. **Learning signal:** does a reward that separates good losses from bad ones train a stronger policy? (Rank vs v7 finals at 20M.)
2. **Speed of learning:** does it get there sooner? (The greedy3 curve and reference rating against v7's at matched steps. The 5M point is the screen from [next-steps](../../notes/next-steps.md).)
3. **Side effects:** does valuing 2nd place make the policy play for second instead of winning? (Win rate against 3× greedy and in training.)

## Comparison, set before the run

**Primary:** VP margin against 3× greedy, 400 games per final policy (`compare`, seed 6000). For each seed, take rank minus the v7 final.
- *Rank is better* if all three paired differences are positive **and** their mean is at least +0.3 VP.
- *Worse* if all three are negative and the mean is at most −0.3 VP.
- Otherwise: no detectable difference.

This is the same rule as v8. 0.3 VP is roughly the per-policy noise of a 400-game evaluation.

**Secondary:**
- Mean rank against 3× greedy, which is what this reward targets, plus win rate and mean VP.
- The greedy3 curve (200 games, seed 5000) and the reference rating (`reference_v1`), every 2M steps, against v7's at the same steps. If v7's checkpoints weren't evaluated with this exact spec, I'll run them through it afterwards.
- VP margin in the 2 random + greedy lineup.

**Reference point:** an agent as strong as greedy scores a margin of −2.18 against 3× greedy (finding 20). The v7 finals score −5.61.

**Health:**
- Explained variance above about 0.3, KL at or below about 0.02.
- The rank reward's returns range ±1, like the win/loss reward's, so value losses stay comparable.
- Watch for a training win rate that falls well below v7's at the same steps while its VP rises. That would mean the policy is playing for place.

## Outcomes and what comes next

| Result | Reading | Next |
|---|---|---|
| Rank better | The learning signal was a bottleneck | Use rank reward from here on, including in the board-aware network |
| No difference, rank ahead early | It speeds learning up but doesn't change the ceiling | Keep it for faster screening; the network is the main line |
| No difference | The signal isn't the limit; the model is | The board-aware network (H4), with less doubt that it's the right target |
| Rank worse | Rewarding 2nd place hurts winning | Drop it; the network is the main line |

## Cost

3 runs × 2 workers = 6 workers. Estimated at 7.5–8.5 hours of training: 60M steps at about 2,300 steps/s, allowing for the snapshot pool filling to 40. Analysis adds about 30 minutes. Started around midnight, it should finish around 8:30–9:30 am. The dashboard ETA after the first hour is the better estimate.
