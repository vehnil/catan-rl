# Next steps

Roughly in order. See [goals](goals.md) for the why.

## Up next: behaviour cloning from greedy

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
- **PPO** (`agents/learned/ppo/`): actor-critic with a scalar value head. Reward is +1 for the winner and −1 for everyone else at the end of the game, plus 0.05 per VP gained (`vp_reward_scale`). Two trained policies from the first round: `8a19bcdc50` (vs random) and `9a4a93bf0a` (vs greedy); neither beats greedy.
- **Greedy trading:** proposes at most one offer per turn, giving at least as many cards as it asks for, picked by its own evaluation gain. It accepts any offer that raises its own evaluation by more than 2, without looking at the proposer's gain, which PPO learned to exploit. Search agents never consider trade proposals.
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
- [ ] Behavior cloning from greedy (see the top of this file), later from ISMCTS games.
- [ ] Expert iteration: ISMCTS guided by the network's policy and value, with the network trained on the search results; repeat.

## 5. Trading agents

- [ ] Belief-aware offers: pick offers by P(accept) × (my gain − partner's gain), using the tracker's estimate of each opponent's hand.
- [ ] Let search consider trade proposals (a small beam of offers with modelled responses).
- [ ] Use what trades reveal: an offer or a refusal is evidence about the other player's hand.

## 6. Performance (when training gets slow)

- [ ] Fix device handling in `trainer.py` (lines 133 and 154 ignore `device`), then run rollouts on CPU and updates on MPS.
- [x] Rollout workers and a faster greedy: done (see the top of this file).
