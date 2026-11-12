# Experiments

YAML configs for `catan train` (PPO) and `catan tournament`, grouped by version. Each version folder is one line of work.

- Once a version has been run, leave its configs as they were, so the results below stay reproducible.
- To change an approach, start a new folder with the next number rather than editing an old one.
- Results live in `data/` (gitignored), keyed by the run ids and policy ids recorded here.

```bash
catan train experiments/v1_examples/ppo_small.yaml
catan tournament experiments/v1_examples/baselines.yaml
```

What we've learned across versions, and hypotheses for training better: [notes/findings.md](../notes/findings.md).

Config fields and defaults are in `src/catan/config/models.py`. Unknown keys are rejected. See [docs/50-training.md](../docs/50-training.md) for the training loop.

## Evaluation protocol (from v7 on)

v2–v6 drew conclusions from 60–200 games and one to four seeds, which is often within the noise ([findings](../notes/findings.md) 12). From v7 on, every experiment follows these rules:

1. **Write down the comparison before running:** what is compared with what, the primary metric, and what counts as better.
2. **Run at least three seeds per arm,** and compare arms with `catan compare`. It reports the mean over seeds and a 95% t-interval. If the intervals overlap, the result is "no detectable difference".
3. **Primary metrics:**
   - **VP margin over the best opponent, and mean rank.** These carry information in every game, unlike wins.
   - **The reference rating** (`catan rate`, set in [reference.yaml](reference.yaml)).
   - Win rate is secondary.
4. **Headline lineup: 3× greedy.** The 2 random + greedy lineup is a secondary, beginner-level check, since random players inflate everyone's VP.
5. **Game counts:**
   - 400 games per final evaluation.
   - 200 per `catan curve` checkpoint.
   - A 400-game reference rating at each 1M-step checkpoint for long runs.
6. **Sampled play is the main result,** since that's the policy as trained. `--deterministic` (argmax) is a secondary view.
7. **Check training health** with `catan health <run>` before reading any results.

## Chaining stages

`init_policy` warm-starts from an earlier policy. It accepts a policy id, an exact policy name, or a training config's `name`. A config name picks the **newest** policy trained by that config, because trained policies are named `<config name>-<run id>`. So stage *n* can say `init_policy: <stage n−1 name>`, and it picks up whatever that stage last produced. Re-running a stage replaces what the next stage starts from.

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

### v3_greedy_ladder: four-stage greedy ladder

The opponents get one greedy player stronger at each stage. Each stage warm-starts from the previous one.

| Stage | Opponents (exact lineup) | Steps | Warm start |
|---|---|---|---|
| `stage1_3random` | random, random, random | 500k | from scratch |
| `stage2_2random_1greedy` | random, random, greedy | 1M | stage 1 |
| `stage3_1random_2greedy` | random, greedy, greedy | 1M | stage 2 |
| `stage4_3greedy` | greedy, greedy, greedy | 2M | stage 3 |

- **Exact lineups.** Every game has exactly these opponents (`league.lineup: true`), with seats shuffled each game. The older configs instead drew each seat at random from a pool, so the mix varied from game to game.
- **No self-play** (`self_play_prob: 0`), so each stage measures one thing.
- **8 parallel workers** (`num_envs: 8`).
- **Evaluation** uses the stage's own lineup, 40 games every 100k steps. A stage is "passed" when it clearly beats its share, which is 25% with four players. Compare stages on a common yardstick afterwards with a gauntlet tournament.

Run them in order (about 4.5M steps in total):

```bash
for s in stage1_3random stage2_2random_1greedy stage3_1random_2greedy stage4_3greedy; do
  catan train experiments/v3_greedy_ladder/$s.yaml || break
done
```

Results (8 workers, about 2,600 steps/s):

| Stage | Run → policy | Final eval vs its lineup |
|---|---|---|
| `stage1_3random` | `c2e62f657fc9` → `91194c26a7` | 95% wins, 10.0 VP (under 3 minutes) |
| `stage2_2random_1greedy` | `7feeb985b7ed` → `774c176a44` | 2.5% wins, 4.1 VP |
| `stage3_1random_2greedy` | `513e4781886c` | stopped by hand after about 12k steps |
| `stage4_3greedy` | | not run |

Stage 2 stalled. Wins stayed at 0–2.5% and VP at about 3.2–3.5 for all 1M steps, while entropy rose from 1.62 to 1.89. Greedy cashes in on the random players' mistakes and ends the game around turn 60, long before the policy reaches 10 VP. So the policy loses every game, the value head learns "I lose", and only the small VP reward is left. More greedy players would only make that worse, so the ladder was stopped.

### v4_ppo_ladder: PPO against frozen PPO

The policy learns to win against earlier versions of itself, then meets greedy:

| Stage | Opponents (exact lineup) | Steps | Warm start |
|---|---|---|---|
| `stage1_3random` | random × 3 | 500k | from scratch (produces PPO v1) |
| `stage2_3ppo1` | PPO v1 × 3 (frozen) | 1M | stage 1 |
| `stage3_2ppo1_1greedy` | PPO v1 × 2 (frozen), greedy | 1M | stage 2 |

- **Only the learner trains.** A `policy:<ref>` opponent is resolved to a policy id when the run starts (the log shows which one), loaded once, and kept frozen.
- **`policy:v4_stage1_3random`** means the newest policy trained by stage 1.
- **Faster wins pay more** (`fast_win_bonus: 0.5`, `fast_win_turns: 200`): +0.5 on top of +1 for a win at turn 0, shrinking linearly to nothing by turn 200. Losses get no bonus, so there's no reason to lose quickly.
- **Game length is tracked:** `turns_100` in training and `eval_mean_turns` in evals.

The question to answer: does stage 3 (2 frozen v1 + greedy) do better than v3's stage 2 (2 random + greedy: 2.5% wins, 4.1 VP)? If it does, build the full ladder.

```bash
for s in stage1_3random stage2_3ppo1 stage3_2ppo1_1greedy; do
  catan train experiments/v4_ppo_ladder/$s.yaml || break
done
```

Results (8 workers; 26 minutes for all three stages):

| Stage | Run → policy | Final eval vs its lineup |
|---|---|---|
| `stage1_3random` | → `7d21686065` (PPO v1) | 100% wins, games down from 213 to 121 turns |
| `stage2_3ppo1` | → `9413f93f8b` (PPO v2) | 80% at the end (92–100% from 300k on), games about 95–110 turns |
| `stage3_2ppo1_1greedy` | → `cac7b6904c` | 2.5% wins, 3.5 VP |

Stage 1 learned much faster than v3's stage 1 (97.5% at 200k vs 40%), and games got shorter as it went. That fits the speed bonus, though the seeds also differed. Stage 2 beat frozen v1 easily and kept getting faster.

The same comparison for every policy (200 games each, same seeds and seat rotations; 95% interval in brackets):

| Policy | + 2 random + greedy | + 3 greedy |
|---|---|---|
| v3 stage 2 (`774c176a44`) | 2.5% [1–6%], 3.91 VP | 0.5%, 3.38 VP |
| v4 stage 2, PPO v2 (`9413f93f8b`) | 1.5% [1–4%], **4.30 VP** | 0.5%, 3.01 VP |
| v4 stage 3 (`cac7b6904c`) | 0% [0–2%], 3.19 VP | 0%, 2.76 VP |

There was no gain against greedy:
- Self-play made the policy faster than v1 (about 95 turns vs 120), but greedy ends these games around turn 65.
- Training against greedy (stage 3) made it worse, as in v3. It almost never wins, entropy drifts up (0.86 → 1.3), and play degrades.
- The best policy against greedy is PPO v2, which never played greedy.

### v5_population: four learners, then each against the others

Several learners with different settings, so they learn different things, then each one trains against the others. Greedy is only a yardstick, never a training opponent.

| Learner | What's different (otherwise v4's settings) | Seed (stage 1 / 2) |
|---|---|---|
| A `A_base` | nothing: speed bonus 0.5, 0.05 per VP, entropy 0.01 | 10 / 20 |
| B `B_nobonus` | no speed bonus (plain win/loss reward; also tests whether the bonus helped in v4) | 11 / 21 |
| C `C_vp` | double the reward per VP (`vp_reward_scale: 0.1`) | 12 / 22 |
| D `D_explore` | double the exploration bonus (`entropy_coef: 0.02`) | 13 / 23 |

| Stage | Opponents | Steps per learner | Warm start |
|---|---|---|---|
| `stage1_<learner>` | random × 3 | 500k | from scratch |
| `stage2_<learner>` | the other three learners' stage-1 policies (frozen) | 1M | own stage 1 |

- **Parallel.** The four learners of a stage run at the same time, with 2 workers each, using all 8 cores.
- **One stage at a time,** with a review of the results before the next. A third round against the peers' stage-2 policies is possible if stage 2 looks promising.
- **After each stage, every policy gets the same yardstick:** 200 games against 2 random + greedy and 200 against 3× greedy (same seeds and seat rotations as the v4 comparison), plus a round-robin of the population.

Run a stage:

```bash
for l in A_base B_nobonus C_vp D_explore; do
  catan train experiments/v5_population/stage1_$l.yaml > data/v5_stage1_$l.log 2>&1 &
done; wait
```

As built, v5 compares *settings*: the learners differ in their own settings but all face the same environment. The intended experiment, identical learners compared across seeds, is v6.

#### Stage 1 results (four learners in parallel, about 12 minutes)

Evals against 3× random (40 games), as wins / turns per game:

| Learner | 100k | 200k | 300k | 400k | 500k | Policy |
|---|---|---|---|---|---|---|
| A base | 52.5% / 222 | 90% / 183 | 95% / 138 | 100% / 113 | 97.5% / 115 | `0111f9d125` |
| B no bonus | 42.5% / 249 | 87.5% / 166 | 97.5% / 140 | 95% / 124 | 97.5% / 123 | `60d4a7d393` |
| C 2× VP | 42.5% / 241 | 97.5% / 170 | 100% / 119 | 97.5% / 137 | 100% / 114 | `836dbaee25` |
| D 2× explore | 37.5% / 296 | 82.5% / 209 | 92.5% / 173 | 95% / 159 | 92.5% / 128 | `c3c961eee6` |

The yardstick: the same 200-game comparison as for v3 and v4, with the same seeds and seat rotations.

| Policy | + 2 random + greedy | + 3 greedy |
|---|---|---|
| A | 2.5%, **4.79 VP** | 0%, **3.55 VP** |
| B | 3.5%, 4.75 VP | 0%, 3.37 VP |
| C | 2.5%, 4.60 VP | 0%, 3.04 VP |
| D | 2.0%, 4.06 VP | 0%, 3.16 VP |
| *v4 PPO v2* | 1.5%, 4.30 VP | 0.5%, 3.01 VP |
| *v3 stage 2* | 2.5%, 3.91 VP | 0.5%, 3.38 VP |

Round-robin of the four plus v4's PPO v2 (400 games, every 4-of-5 lineup, seats rotated):

| Policy | Win rate [95% interval] | VP |
|---|---|---|
| v4 PPO v2 | 56.6% [51–62%] | 8.2 |
| C | 50.9% [45–56%] | 7.8 |
| A | 7.8% | 4.6 |
| D | 6.9% | 4.7 |
| B | 2.8% | 4.1 |

- **Training only against random gave the best yardstick scores yet** (A, B, C: 4.6–4.8 VP against greedy). Training against greedy hasn't helped so far.
- **The speed bonus made no clear difference** against random or greedy (A vs B), though B finished last in the round-robin.
- **C (2× reward per VP) is strong against other PPO policies,** almost matching PPO v2, which trained three times as long. That's one seed per setting, so v6 repeats C's settings with four seeds.

### v6_seeds: one learner, four seeds

v5 as intended. Four **separate** learners, each its own network, with **identical settings**: v5 learner C's (`vp_reward_scale: 0.1`, speed bonus 0.5 over 200 turns, `entropy_coef: 0.01`). Only the seed differs, so each sees its own boards, dice and games. They don't share weights; each one's workers pool their own experience into one update every 4,096 decisions (about 10–30 games).

| Stage | Opponents | Steps per learner | Seeds |
|---|---|---|---|
| `stage1_s1..s4` | random × 3 | 500k | 30–33 |
| `stage2_s1..s4` (draft) | the other three learners' stage-1 policies, frozen | 1M | 40–43 |

Questions for stage 1:
- **How much do identical learners differ by seed alone?** v5's four settings ended up close against random and greedy but far apart in the round-robin. If v6's seeds spread just as widely, v5's differences were mostly luck.
- **Is C's round-robin strength real?**

In stage 2, each seat is drawn from a pool of the three peers plus `random`, so about 25% of seats are random. This mixing is meant to avoid overfitting to the peers ([findings](../notes/findings.md) H3). Stage 2 also saves a checkpoint every 200k steps.

Run one stage at a time, then compare the four (the greedy yardstick plus a round-robin with v5 C and v4 PPO v2) before the next:

```bash
for l in s1 s2 s3 s4; do
  catan train experiments/v6_seeds/stage1_$l.yaml > data/v6_stage1_$l.log 2>&1 &
done; wait
```

#### Stage 1 results (about 12 minutes)

Evals against 3× random, as wins / turns. Identical settings; only the seed differs:

| Learner | 100k | 200k | 300k | 400k | 500k | Policy |
|---|---|---|---|---|---|---|
| s1 | 47.5% / 252 | 72.5% / 233 | 90% / 173 | 97.5% / 144 | 95% / 130 | `7d8befad98` |
| s2 | 85% / 189 | 97.5% / 149 | 92.5% / 143 | 95% / 127 | 100% / 124 | `caa237bc46` |
| s3 | 70% / 213 | 97.5% / 155 | 97.5% / 138 | 95% / 140 | 97.5% / 111 | `89ab7f7aa7` |
| s4 | 80% / 243 | 100% / 164 | 100% / 142 | 97.5% / 148 | 100% / 134 | `15b6f5bc65` |

The yardstick (200 games, same seeds as v3–v5):

| Policy | + 2 random + greedy | + 3 greedy |
|---|---|---|
| s1 | 3.0%, 4.40 VP | 0%, 3.18 VP |
| s2 | 3.0%, 4.54 VP | 0%, 3.34 VP |
| s3 | 3.0%, 4.74 VP | 0.5%, 3.27 VP |
| s4 | 3.5%, 4.42 VP | 0%, 3.39 VP |

Round-robin of the four, v5 C and v4 PPO v2 (480 games): PPO v2 58.4% [53–64%], v5 C 40.3% [35–46%], s3 19.4%, s2 13.1%, s4 10.3%, s1 8.4%.

- **Seed alone spreads early learning widely** (47.5–85% at 100k, compared with v5's 37.5–52.5% across different settings), but all four end close together.
- **Against greedy, seeds span 4.40–4.74 VP.** That covers v5 A, B and C, so v5's setting changes made no measurable difference there.
- **C's round-robin lead didn't reproduce.** Four seeds with exactly C's settings won 8–19%, against v5 C's 40%. C's seed was lucky; one seed per setting isn't enough.
- **Self-play is still strongest against PPO opponents:** PPO v2 tops the round-robin. It doesn't help against greedy.

#### Measuring overfitting in stage 2

Run `catan curve v6_stage2_sX` on each learner, playing every policy against four fixed lineups:
- **peers:** its three stage-1 peers, the opponents it trains on.
- **unseen PPO:** v5 A, v5 C and v4 PPO v2, which it never trains against.
- **greedy mix:** 2 random + greedy.
- **3× random.**

The policies are its stage-1 starting point ("pre"), each 200k checkpoint, and the final policy ("post"). Overfitting shows up as the peers score rising while the held-out scores flatten or fall.

#### Stage 2 results (1M steps each)

Overfitting curves from `catan curve`, 60 games per cell, seed 5000. Win rate (VP for the greedy mix) at the start, 400k, 800k and the end:

| Learner | Peers (trained on) | Unseen PPO (held out) | Greedy mix VP (held out) | 3× random |
|---|---|---|---|---|
| s1 | 17 → 55 → 68 → 75% | 2 → 25 → 22 → 25% | 4.32 → 4.37 → 4.85 → 5.32 | 97 → 100% |
| s2 | 37 → 60 → 78 → 70% | 12 → 27 → 32 → 48% | 4.25 → 4.60 → 5.13 → 4.88 | 97 → 98% |
| s3 | 37 → 67 → 90 → 85% | 5 → 25 → 35 → 30% | 4.02 → 4.88 → 4.53 → 4.67 | 97 → 98% |
| s4 | 13 → 30 → 55 → 80% | 7 → 12 → 32 → 38% | 4.22 → 4.18 → 4.67 → 4.27 | 97% |

Full tables (including 200k and 600k) are in `data/curves/v6_stage2_sX.json`. With 60 games, a single point carries about ±12 points, so read the trends.

The yardstick (200 games, same seeds as v3–v5):

| Policy | + 2 random + greedy | + 3 greedy |
|---|---|---|
| s1 `37c99e2fac` | 4.0%, 4.78 VP | 0.5%, 3.50 VP |
| s2 `e2f768b3ab` | 6.5% [4–11%], 4.92 VP | 0%, 3.53 VP |
| s3 `cb5cb1c5ef` | 5.0%, 4.98 VP | 1.0%, 3.50 VP |
| s4 `cb4614a1f1` | 5.5%, 5.13 VP | 0%, 3.62 VP |

Checkpoints (200k / 400k / 600k / 800k): s1 `7cbee1e5c5` `a79e3e27cf` `426387e952` `1ed72852eb`; s2 `535f4ece5c` `b50ae85cca` `d7aef17750` `23bb0a13e1`; s3 `969141c1b9` `0a36b31021` `8d70bd1e5e` `ef253a8e67`; s4 `5ec61ad89c` `a7b186b421` `ec1a6ad758` `c6e08bf595`.

- **No strong overfitting.** Scores against held-out PPO rose from 2–12% to 25–48% (25% is even), so the learners didn't just learn to beat their peers.
- **Mild late specialisation in s1 and s3:** their held-out PPO score flattens after about 400k while the peers score keeps rising. s2 and s4 kept improving on both.
- **No forgetting:** all four still beat 3× random 97–100% of the time.
- **Best results against greedy so far.** Every learner gained 0.24–0.71 VP over its stage-1 self, and all four beat the previous best (v5 A, 4.79 VP and 2.5% wins). Unlike v4, self-play here helped against greedy, even though greedy never appeared in training. The differences from v4 are a population of peers instead of one frozen copy, and about 25% random seats.

### v7_baseline: the conventional self-play baseline

The plain recipe nearly every self-play project starts from, and that we had never run: one continuous run from scratch, no stages, with the opponent drawn from the policy's own history ([findings](../notes/findings.md), critique of v2–v6).

- **Opponent seats, each drawn independently:**
  - the current policy, 50%;
  - a uniformly chosen past snapshot, 30% (one every 500k steps, last 40 kept, so the whole run);
  - a fixed opponent, 20% (random, `greedy:epsilon=0.5` or greedy).
- **Settings, frozen for v7:**
  - 20M steps, linear learning-rate decay from 2.5e-4.
  - The plain reward: ±1 plus 0.05 per VP, no speed bonus.
  - `[512, 256]` MLP, `num_envs: 2`.
  - Checkpoints and resumable state every 1M.
- **Three seeds,** s1–s3 (seeds 50–52), run at once overnight.
- **`smoke.yaml`:** a 200k-step check of the same setup.

**Comparison, set before the run:**
- *Primary:* the reference rating (`catan rate`, [reference.yaml](reference.yaml)) of the three finals, mean and 95% interval over seeds.
- *Secondary:* VP margin and mean rank against 3× greedy (`catan compare`, 400 games), against v6 stage 2's four finals.
- *Better* means the three-seed interval for VP margin against 3× greedy lies entirely above v6's.
- *Training health:* explained variance stays above about 0.3, KL at or below about 0.02, and the reference rating rises across the 1M checkpoints. If it plateaus, the next step is longer runs or PFSP opponent sampling, not new settings.

```bash
for s in s1 s2 s3; do
  catan train experiments/v7_baseline/baseline_$s.yaml > data/v7_baseline_$s.log 2>&1 &
done; wait
# after a crash or reboot:
catan train experiments/v7_baseline/baseline_s1.yaml --resume v7_baseline_s1
```

Results: not run yet.
