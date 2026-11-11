# Next steps

Roughly in order. See [goals](goals.md) for the why.

## What already exists

- **Belief tracker** (`player/belief.py`): forward filter and Viterbi over opponents' hands. It rules out impossible hands on builds, offers, accepts, discards and monopoly payouts. It also gives `determinize()` for search.
- **PPO** (`agents/learned/ppo/`): actor-critic with a scalar value head. Reward is +1 for the winner and −1 for everyone else at the end of the game. The only saved policy (`ppo_small`, 20k steps) is essentially untrained.
- **Greedy trading:** one-for-one offers that almost always get rejected. Search agents never consider trade proposals.
- **Replays** store only `seed + actions`; no GUI exists.

## 1. Measurement

- [ ] Belief accuracy: compare the tracker to the true hands during simulated games (log-likelihood of the true hand, how often the most likely hand is right, how many hypotheses are tracked).
- [ ] Trade stats in the store: proposals, acceptances, and the win rate of traders vs non-traders.
- [ ] A tournament with trading on vs off, to see whether trading matters for the current agents.

## 2. Game export and the GUI

- [ ] Timeline export: replay → JSON with the board spelled out plus per-step state and public events. This is also a readable game log.
- [ ] `catan view <replay>`: a small local server (stdlib only) and one HTML page with an SVG board, step slider, play/pause, player panels and an event log.
- [ ] A seat point-of-view mode that hides opponents' hands and shows the belief tracker's guesses next to the truth.

## 3. Better hand inference

- [ ] Smoothing (forward-backward): revise past unseen steals once later evidence comes in. Today only the current estimate is updated.
- [ ] Joint tracking for steals between two opponents. Today each opponent's hand is tracked separately, which drops the link between thief and victim.
- [ ] Dev card inference from behaviour, e.g. a player who doesn't play a knight while the robber blocks them probably doesn't hold one.
- [ ] More belief inputs for the network: uncertainty (entropy) per opponent, and P(can afford) for each build type.

## 4. The value model

- [ ] Value head → 4-way win probability (softmax over seats), trained on game outcomes. A move's value = change in my win probability.
- [ ] Dataset builder: re-run replays to recover each decision's (features, mask, action, outcome).
- [ ] Behavior cloning from greedy/ISMCTS games, then warm-start PPO from it (`init_policy`).
- [ ] Expert iteration: ISMCTS guided by the network's policy and value, with the network trained on the search results; repeat.

## 5. Trading agents

- [ ] Belief-aware offers: pick offers by P(accept) × (my gain − partner's gain), using the tracker's estimate of each opponent's hand.
- [ ] Let search consider trade proposals (a small beam of offers with modelled responses).
- [ ] Use what trades reveal: an offer or a refusal is evidence about the other player's hand.

## 6. Performance (when training gets slow)

- [ ] Fix device handling in `trainer.py` (lines 133 and 154 ignore `device`), then run rollouts on CPU and updates on MPS.
- [ ] Vectorized environments: N games in worker processes, with their positions batched into one network call.
- [ ] Speed up `greedy` / `evaluate`, which dominate the cost of opponents during training.
