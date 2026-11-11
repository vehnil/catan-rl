# Next steps

Roughly in order. See [goals](goals.md) for the why.

## Up next: ground-up ladders (no imitation)

The project prefers learning from the ground up, so cloning waits (see below). The v3 ladder (3 random → 2 random + 1 greedy → …) stalled at stage 2: 0–2.5% wins and about 3.5 VP for all of its 1M steps. Greedy cashes in on the random players' mistakes and finishes around turn 60, long before the policy does. The policy loses every game, so the win/loss reward never tells it which moves were better.

- [x] **v4 PPO ladder** (`experiments/v4_ppo_ladder`): 3 random → 3× frozen PPO v1 → 2× PPO v1 + 1 greedy, with the speed bonus. No gain against greedy. Self-play made it faster (about 95 turns vs 120 against v1) but greedy finishes around 65, and the greedy stage made it worse, as in v3. Best against greedy: PPO v2 (`9413f93f8b`, never trained on greedy), 4.3 VP vs 3.9 for v3. Details in `experiments/README.md`.
- [ ] Variant, probably too ambitious but worth a look: 3 random → 3 PPO v1 → 2 PPO v2 + 1 greedy, where v2 is the stage-2 learner.
- [ ] **v5 population** (`experiments/v5_population`): four learners with different settings train in parallel against random, then each trains against the other three (frozen). Greedy is only a yardstick, not a training opponent. Run one stage at a time and discuss before the next.
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
