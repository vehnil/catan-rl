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

Stage 2 is a draft, to revisit after stage 1. One option from [findings](../notes/findings.md) H3 is to keep about 20% random seats alongside the frozen peers, to avoid overfitting to them.

Run one stage at a time, then compare the four (the greedy yardstick plus a round-robin with v5 C and v4 PPO v2) before the next:

```bash
for l in s1 s2 s3 s4; do
  catan train experiments/v6_seeds/stage1_$l.yaml > data/v6_stage1_$l.log 2>&1 &
done; wait
```

Results: not run yet.
