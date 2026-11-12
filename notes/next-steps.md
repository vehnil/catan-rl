# Next steps

Roughly in order. See [goals](goals.md) for the why.

## Up next: put training on a conventional footing

A critique against standard RL practice found that v2–v6 trained far too briefly (0.5–1M steps), never ran the plain self-play baseline, and drew conclusions from noisy, small samples. Fix in this order (plan: phases 1–6):

- [x] **PPO health:** explained variance, a better KL estimator, grad norm, `target_kl`, linear LR decay, `catan health`. The retroactive check is [findings.md](findings.md) 13.
- [x] **Resumable long runs:** `train_state.pt` at each checkpoint, `catan train --resume`.
- [x] **Evaluation protocol:** VP margin and rank as primary metrics, the `catan rate` reference rating, `catan compare` across seeds, `#argmax` policies. The protocol is in `experiments/README.md`.
- [ ] **v7 baseline:** continuous self-play against the current policy plus a 40-snapshot history plus 20% fixed opponents, 20M steps × 3 seeds, overnight. Discuss before anything else.
- [ ] **Before v7's verdict** (see [drawing-board](drawing-board.md)):
  - Greedy vs 3× greedy, 400 games: the margin, rank and win rate that "as good as greedy" actually means.
  - Pace at each checkpoint: game length against 3× greedy and the learner's VP when greedy wins.
  - Explained variance on the greedy eval games, not just in training.
  - The s1 checkpoints at 200 games, to check whether s1 stalled.
- [ ] **Opponent-mix arm:** v7 with random dropped from the fixed pool and greedy plus `greedy:epsilon=0.5` raised to about 30–40% of seats (full greedy is about 7% in v7). Config only.
- [ ] **Screen new arms at 5M** against v7's own 5M checkpoints; only arms that beat it get 20M × 3 seeds.
- [ ] **v8 reward:** rank reward against win/loss on the v7 setup, 3 seeds each.
- [ ] **v9 network:** per-vertex features, then a board-aware (graph) network, against the v7 baseline.
- [ ] Maybe: PFSP opponent sampling (favour snapshots the learner loses to), if v7's uniform pool plateaus.

## Earlier: ground-up ladders (no imitation)

The project prefers learning from the ground up, so cloning waits (see below). The v3 ladder (3 random → 2 random + 1 greedy → …) stalled at stage 2: 0–2.5% wins and about 3.5 VP for all of its 1M steps. Greedy cashes in on the random players' mistakes and finishes around turn 60, long before the policy does. The policy loses every game, so the win/loss reward never tells it which moves were better.

- [x] **v4 PPO ladder** (`experiments/v4_ppo_ladder`): 3 random → 3× frozen PPO v1 → 2× PPO v1 + 1 greedy, with the speed bonus. No gain against greedy. Self-play made it faster (about 95 turns vs 120 against v1) but greedy finishes around 65, and the greedy stage made it worse, as in v3. Best against greedy: PPO v2 (`9413f93f8b`, never trained on greedy), 4.3 VP vs 3.9 for v3. Details in `experiments/README.md`.
- [ ] Variant, probably too ambitious but worth a look: 3 random → 3 PPO v1 → 2 PPO v2 + 1 greedy, where v2 is the stage-2 learner.
- [ ] **v5 population** (`experiments/v5_population`): four learners train in parallel against random, then each trains against the other three (frozen). Greedy is only a yardstick, not a training opponent. Run one stage at a time and discuss before the next. *As built, this is a settings comparison, not what was intended:* the learners differ in their own settings (baseline, no speed bonus, 2× VP reward, 2× exploration) but all face the same environment. That still shows which settings to use for the experiment below.
- [x] **v6: one learner, four seeds (what v5 was meant to be)** (`experiments/v6_seeds`): four separate learners, each its own network, with identical settings (v5 C's) and only different seeds. Each sees its own boards and dice. Compare them after each stage, then train each against the other three (frozen) before anything faces greedy. This is also the seed check on v5 C's round-robin lead. Findings and hypotheses so far: [findings.md](findings.md).
- [ ] Later: identical learners in *different* environments. Candidate environments, mixed and matched:
  - Opponents: 3× random; 3× `greedy:epsilon=0.8` (mostly random with some greedy play); 3× frozen PPO v1; 3× greedy.
  - Board: random boards vs the fixed standard board (no board variety).
  - Rules: trading on vs off; a lower VP target (e.g. 8) for shorter games and different strategies.
- [ ] Epsilon-greedy ladder: opponents `greedy:epsilon=0.8 → 0.5 → 0.2 → 0`, so difficulty rises in steps instead of jumping from random to greedy.
- [ ] Reward that still teaches when losing, e.g. finishing rank or VP margin over the leader instead of only win/loss.

## Later: behaviour cloning from greedy

The first PPO round (500k steps vs random, then 1M vs greedy) beat random but never beat greedy. It learned to exploit greedy's trade acceptance (greedy only checks its own gain) and almost never built cities. Cloning gives it greedy's play as a starting point, and a direct training signal for the opening, which the end-of-game reward barely reaches (γ = 0.995 over ~200–380 decisions per game).

- [x] Harden greedy's trade acceptance (`max_partner_edge`, plus "never give more cards than you get"). The 1-for-2 exploit is gone. The mirror stays at 25% per seat, greedy still beats random 100/100, and mirror trades drop from 4.9 to 3.4 per game.
- [ ] Performance first. Training is collection-bound: in a profile of 8k steps, `env.step` took 9.9 s and the PPO update 0.7 s.
  - [x] Parallel rollout workers (`ppo.num_envs`): one env and policy copy per process, chunked collection. About 2.5× steps/s with 8 workers on the M1. Less than hoped: the efficiency cores are slow, and per-worker speed drops as more run.
  - [x] ~~Batched inference across envs~~: tried first, but the main process became the bottleneck (about 1 ms per small forward pass plus pipe traffic), capping out at about 1,750 steps/s. Per-worker policy copies replaced it.
  - [x] `torch.set_num_threads(1)` in each worker.
  - [x] Vectorise greedy's `_best_offer` (`evaluate_hands`, bit-for-bit equal to `evaluate`): greedy games about 30% faster, identical moves.
  - [ ] Maybe: overlap the update with the next rollout's collection (workers idle about 15% of the time; costs one update of policy lag).
  - [x] **Cheaper network players (Nov 2026):** v7's opponent mix makes about 80% of seats networks, so a learner step costs about 3.8 network decisions. Three changes made a v7 worker 1.23× faster (about 845 → 1,035 steps/s alone, 250 → 300 with six at once):
    - Features fill one preallocated vector, with the board blocks cached per board. Bit-identical, checked against a frozen copy of the old encoder.
    - The learner samples without building a `Categorical`. Bit-identical, same RNG stream.
    - Opponents run the policy in NumPy and sample with the agent's NumPy RNG. Same distribution, different random stream.
  - Scaling on this M1 is the bigger limit. Six workers at once give only about 2.3× one worker against random opponents, and about 1.75× with network opponents (each of those networks is about 2.8 MB, and many snapshots compete for the cache). The machine has 4 performance and 4 efficiency cores, plus background load. More cores (option D) beats further code work here.
  - [x] **Batch network decisions across games (`ppo.games_per_worker`, Nov 2026):** about 2× per worker with six workers at once (290 → 575 steps/s at 32 games), but only about 1.0–1.45× running alone.
    - It's mainly a memory effect: one read of the weights per batch instead of one per decision.
    - Batches average about 8 games for the learner and 7 for current-policy opponents. Past snapshots don't batch (about 1.2), being spread over many models.
    - A first version that kept every game in lockstep didn't help at all: opponents move in bursts, so batches averaged 1.9.
  - [ ] Maybe: lower `max_trade_proposals_per_turn` to cut trade-spam decisions (a rules change).
- [ ] Generate 2–5k greedy-only games in parallel (tournament runner, replays saved).
- [ ] Dataset: re-run the replays and record each seat's (features with its own beliefs, mask, action, final outcome). Four trajectories per game.
- [ ] Train the same `ActorCritic`: cross-entropy on greedy's actions for the policy head, the final outcome for the value head. Report held-out accuracy.
- [ ] Save it as a normal PPO policy, then warm-start PPO from it (`init_policy`) with a lower learning rate.
- [ ] Capacity check: if imitation accuracy plateaus low, compare a bigger network against richer features to see which one is missing.

## What already exists

- **Belief tracker** (`player/belief.py`): forward filter and Viterbi over opponents' hands. It rules out impossible hands on builds, offers, accepts, discards and monopoly payouts. It also gives `determinize()` for search.
- **PPO** (`agents/learned/ppo/`): actor-critic with a scalar value head. Reward is +1 for the winner and −1 for everyone else at the end of the game, plus 0.05 per VP gained (`vp_reward_scale`) and an optional speed bonus for wins (`fast_win_bonus`). Training runs in parallel workers (`num_envs`), and opponents can be fixed lineups including frozen stored policies. None of the trained policies beats greedy yet; see `experiments/README.md` for every run.
- **Greedy trading:** proposes at most one offer per turn, never asking for more cards than it gives, picked by its own evaluation gain. It accepts only if its evaluation rises by more than 2, the offer doesn't ask for more cards than it gives, and the proposer's estimated gain isn't bigger than its own. The older self-only rule, which PPO exploited, is `max_partner_edge=inf`. Search agents never consider trade proposals.
- **Replays** store only `seed + actions`; `catan view` replays them in the browser, with seat views and beliefs.

## 1. Measurement

- [ ] Belief accuracy: compare the tracker to the true hands during simulated games (log-likelihood of the true hand, how often the most likely hand is right, how many hypotheses are tracked).
- [ ] Trade stats in the store: proposals, acceptances, and the win rate of traders vs non-traders.
- [ ] A tournament with trading on vs off, to see whether trading matters for the current agents.

## 2. Game export and the GUI

- [x] Timeline export: replay → JSON with the board spelled out plus per-step state and public events. This is also a readable game log.
- [x] `catan view <replay>`: a small local server (stdlib only) and one HTML page with an SVG board, step slider, play/pause, player panels and an event log.
- [x] A seat point-of-view mode that hides opponents' hands and shows the belief tracker's guesses next to the truth.
- [ ] Later: better graphics. Today the board is flat-colored hexes and simple piece shapes, and seat colors are the main cue, which is enough for now. Possible upgrades: terrain textures or icons, real piece art, resource-card icons, and animations for rolls, steals and trades.

## 3. Better hand inference

- [ ] Smoothing (forward-backward): revise past unseen steals once later evidence comes in. Today only the current estimate is updated.
- [ ] Joint tracking for steals between two opponents. Today each opponent's hand is tracked separately, which drops the link between thief and victim.
- [ ] Dev card inference from behaviour, e.g. a player who doesn't play a knight while the robber blocks them probably doesn't hold one.
- [ ] More belief inputs for the network: uncertainty (entropy) per opponent, and P(can afford) for each build type.

## 4. The value model

- [ ] Value head → 4-way win probability (softmax over seats), trained on game outcomes. A move's value = change in my win probability.
- [ ] Dataset builder: re-run replays to recover each decision's (features, mask, action, outcome).
- [ ] Behavior cloning from greedy (see "Later" above), later from ISMCTS games.
- [ ] Expert iteration: ISMCTS guided by the network's policy and value, with the network trained on the search results; repeat.

## 5. Trading agents

- [ ] Belief-aware offers: pick offers by P(accept) × (my gain − partner's gain), using the tracker's estimate of each opponent's hand.
- [ ] Let search consider trade proposals (a small beam of offers with modelled responses).
- [ ] Use what trades reveal: an offer or a refusal is evidence about the other player's hand.

## 6. Performance (when training gets slow)

- [ ] Fix device handling in `trainer.py`: the single-process loop's truncation bootstrap and last-value calls ignore `device`. MPS isn't worth it for now, though: measured on the M1, an update takes 0.48 s on MPS vs 0.31 s on CPU, and single-position inference is 11× slower.
- [x] Rollout workers and a faster greedy: done (see the top of this file).
