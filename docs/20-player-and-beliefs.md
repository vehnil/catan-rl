# Player layer: observations and beliefs

`src/catan/player/`: what seat i knows, and what it can infer.

## Observation (`observation.py`)

`observe(state, seat)` returns a frozen `Observation`:

- **Public:** board layout, buildings, roads, robber, bank, dev deck size, phase, whose turn and who acts, last roll, each player's hand size, unplayed dev card count, knights, public VP, road length, pieces left, title holders, any trade in progress.
- **Private to `seat`:** its own hand, playable dev cards, dev cards bought this turn.
- `legal_mask`: the legal actions if `seat` is the actor, otherwise all False.
- `redacted`: a `GameState` copy with opponents' hands and dev cards zeroed and the deck emptied. It exists so search agents can fill the hidden parts with sampled guesses. Don't read opponents' hands from it.

`tests/player/test_observation.py` changes only hidden information (opponent hand composition, opponent dev card types, deck order) and checks that every observation field and the feature vector stay identical.

## Why card counting works in Catan

Almost every card movement is public (production, setup payouts, build costs, bank and domestic trades, discards, monopoly, year of plenty). A perfect counter knows every hand exactly except for:

1. **Robber steals between two other players**: you don't see which card moved.
2. **Development cards**: you see that someone bought one, not which.

So uncertainty starts at zero, grows only through unseen steals, and public evidence keeps cutting it back down.

## BeliefTracker (`belief.py`)

One tracker per seat. Feed it `event.view(seat)` for every event; the base `Agent` class does this automatically.

### Hand model

Each opponent's hand is a **hidden Markov process**. `HandDistribution` stores a weighted set of possible hands (5-resource count vectors):

| Event | Update |
|---|---|
| Known gain or loss (production, trade, build cost…) | **shift** every hypothesis; drop any that go negative |
| Unseen steal from this opponent | **branch** each hypothesis by which card was taken, weighted by count / hand size |
| Unseen steal by this opponent | **mix**: gain one card, distributed like the victim's expected card |
| Evidence (paid a cost, offered a trade, accepted one, discarded a type) | **filter** out hypotheses that couldn't have done it |
| Monopoly payout | **condition** on the exact amount taken, then set that resource to 0 |

Duplicate hypotheses are merged. The support is capped at the top 4,096 (in random-play tests it never exceeded about 30).

Opponents are tracked **independently**, a factored approximation: when one opponent steals from another, the correlation between their hands is dropped. The exact per-resource totals (known from the bank) are enforced later, when sampling.

### Two filters, two answers

The same events drive two copies of each distribution:

- **Forward filter (sum-product).** Merging duplicates sums their weights. This is the true posterior marginal: `hand_marginals(q)` (expected counts), `map_hand(q)` (most probable hand now), `prob_can_afford(q, cost)`.
- **Viterbi filter (max-product).** Merging takes the max. This tracks the single most likely *history* of hidden steals: `viterbi_hand(q)`.

They can differ. A hand reachable through two medium-probability histories can be the most probable hand overall, while a different hand lies at the end of the single most likely history. `tests/player/test_belief.py::test_viterbi_differs_from_marginal_mode` builds exactly that case.

### Dev cards

- `unseen_dev_pool()`: the 25-card deck minus publicly played cards minus your own.
- `dev_type_probs()`: P(type) for any unseen card, assuming every unseen card is equally likely to be any of them.
- `expected_hidden_vp(q)`: number of cards `q` holds × P(VP card).

### Determinization

`determinize(redacted, rng)` produces a full `GameState` consistent with everything seen:

- Opponent hands are sampled from the forward filter, subject to the bank constraint (opponents' totals per resource = 19 − bank − your hand). The last opponent's hand is implied by the other two, with rejection sampling and a fallback to the most likely hands.
- Opponents' dev cards are dealt from the unseen pool, including which were bought this turn; the rest becomes a shuffled deck.

Search agents (ISMCTS, Max^n, Paranoid, the opening bandit) search over these samples instead of the true hidden state.

## Features (`features.py`)

`encode(obs, belief)` → `float32[1388]`, with every seat-indexed quantity rotated so index 0 is "me":

| Block | Size |
|---|---|
| Hexes: terrain one-hot, pips, robber | 19 × 8 |
| Vertices: (settlement, city) × relative owner, port type | 54 × 15 |
| Edges: relative road owner | 72 × 4 |
| Players: hand size, dev count, knights, VP, road length, pieces left, titles, discards owed, expected hidden VP, **expected hand from the belief tracker** | 4 × 17 |
| Globals: own hand and dev cards, bank, deck size, phase, whose move, turn counters, last roll, trade offer, proposer, acceptors | 70 |

Changing this layout invalidates trained PPO weights.

## Agent base class (`base.py`)

`Agent` owns a `BeliefTracker` and feeds it every event. Its API is documented in [agents](30-agents.md#the-agent-api).
