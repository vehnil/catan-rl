# Engine

`src/catan/engine/`: the complete rules of 4-player base-game Catan, with no RL dependencies.

## Board (`board.py`)

- 19 hexes in pointy-top axial coordinates `(q, r)` within radius 2. Corners are computed geometrically, which gives **54 vertices** and **72 edges** with stable integer ids (sorted top to bottom, left to right).
- `TOPOLOGY` holds static adjacency tables: hex→vertices, vertex→vertices, vertex→edges, vertex→hexes, edge→vertices, and the 30 coastal edges in clockwise order.
- Padded tables use a **sentinel** index (`54` for vertices, `72` for edges). State arrays have one extra always-empty slot at that index, so the rules can do vectorized lookups like `vertex_owner[vertex_vertices]` without masking.
- `Board` is a concrete layout: terrain, number tokens, 9 ports (4× 3:1 and one 2:1 per resource) on fixed coastal slots.
  - `Board.standard()` is a fixed reference layout.
  - `Board.random(rng, avoid_adjacent_red=True)` shuffles terrain, tokens and ports, and rejects layouts with adjacent 6/8s.

Resources are `WOOD=0, BRICK=1, SHEEP=2, WHEAT=3, ORE=4` (`constants.py`). Dev cards are `KNIGHT, VICTORY_POINT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY` (14/5/2/2/2).

## State (`state.py`)

`GameState` is a dataclass of small numpy arrays and scalars: hands `(4,5)`, bank `(5,)`, dev deck (hidden order), dev cards held and bought-this-turn `(4,5)`, buildings and roads, robber, remaining pieces, cached road lengths, title holders, discard obligations, trade-in-progress, turn counters and the winner.

- `clone()` copies arrays explicitly (no `deepcopy`). Search agents call it constantly.
- `actor` is the seat that must act next. During `DISCARD` it's the next seat owing cards; during `TRADE_RESPOND` it's the next responder; otherwise it's `current`.
- `total_vp(p)` includes hidden VP cards. `public_vp(p)` doesn't.

`RulesConfig` (frozen dataclass) holds the tunable rules:

| Field | Default | Meaning |
|---|---|---|
| `vp_to_win` | 10 | |
| `discard_limit` | 7 | More cards than this on a 7 → discard half |
| `board` | `"standard"` | or `"random"` |
| `avoid_adjacent_red` | true | for random boards |
| `max_trade_proposals_per_turn` | 3 | domestic proposals per turn |
| `max_actions_per_turn` | 50 | after this, only END_TURN is legal |
| `max_turns_per_game` | 500 | player-turns; then the game is **truncated** |

## Turn flow (`phases.py`)

```
SETUP_SETTLEMENT → SETUP_ROAD → … snake order 0 1 2 3 3 2 1 0 … → ROLL
ROLL ──7──▶ DISCARD (if anyone holds > 7) ─▶ MOVE_ROBBER ─▶ STEAL (if ≥ 2 victims) ─▶ MAIN
ROLL ──else──▶ produce resources ─▶ MAIN
ROLL / MAIN ──knight──▶ MOVE_ROBBER ─▶ STEAL? ─▶ back to ROLL / MAIN
ROLL / MAIN ──road building──▶ ROAD_BUILDING (up to 2 free roads) ─▶ back
MAIN ──propose──▶ TRADE_RESPOND (each opponent in seat order) ─▶ TRADE_CONFIRM? ─▶ MAIN
MAIN ──end turn──▶ ROLL (next seat), or GAME_OVER at the turn cap
any ──current player reaches 10 VP──▶ GAME_OVER
```

The engine resolves some steps automatically:
- A steal with exactly one possible victim happens on its own.
- Trade responders who can't afford the offer are skipped.
- Road building ends early if no legal road remains.

## Rules implemented

- Distance rule. Settlements must connect to your road after setup. Roads can't extend through an opponent's settlement.
- The second setup settlement pays one card per adjacent non-desert hex.
- Production with the robber blocking. Bank shortage rule: if a resource runs short and several players are owed it, nobody gets any; if only one is owed, they get what's left.
- Discard half on 7. The robber must move to a different hex. Stolen cards are drawn uniformly from the victim's hand.
- Dev cards: you can't play one on the turn you bought it, at most one per turn, and you can play it before rolling. Knight, road building, year of plenty and monopoly are implemented. VP cards stay hidden.
- Longest road (≥ 5, trail search that stops at opponent buildings). The holder keeps it on ties; if the holder is overtaken by a tie, nobody holds it. Largest army (≥ 3 knights, strictly more than the holder).
- Bank trades 4:1, generic ports 3:1, resource ports 2:1.
- You can win only on your own turn. VP gained off-turn is checked at the start of your turn.

## Action space (`actions.py`)

Every move is an integer in `[0, 488)`. `encode(type, arg)`, `decode(index)` and `describe(index)` convert between forms; `block(type)` gives a type's slice of the mask.

| Type | Offset | Size | Argument |
|---|---:|---:|---|
| ROLL | 0 | 1 | |
| END_TURN | 1 | 1 | |
| BUILD_SETTLEMENT | 2 | 54 | vertex (also setup placement) |
| BUILD_CITY | 56 | 54 | vertex |
| BUILD_ROAD | 110 | 72 | edge (also setup and free roads) |
| BUY_DEV | 182 | 1 | |
| PLAY_KNIGHT | 183 | 1 | |
| PLAY_ROAD_BUILDING | 184 | 1 | |
| PLAY_YEAR_OF_PLENTY | 185 | 15 | unordered resource pair |
| PLAY_MONOPOLY | 200 | 5 | resource |
| MOVE_ROBBER | 205 | 19 | hex |
| STEAL | 224 | 3 | victim as seat offset +1..+3 |
| DISCARD | 227 | 5 | one card at a time |
| MARITIME_TRADE | 232 | 20 | (give, get) pair; the rate comes from your ports |
| PROPOSE_TRADE | 252 | 230 | offer index into `TRADE_OFFERS` |
| ACCEPT_TRADE / REJECT_TRADE | 482 / 483 | 1 / 1 | |
| CONFIRM_TRADE | 484 | 3 | partner as seat offset |
| CANCEL_TRADE | 487 | 1 | |

**Changing block order or sizes breaks trained policies and stored replays.**

## Domestic trading (`trading.py`, `rules.py`)

1. In `MAIN`, the current player proposes one of 230 offers: give 1–2 cards (at most 2 types) for 1–2 cards of *different* types. You can only offer cards you hold.
2. Each opponent, in seat order, accepts or rejects. Only those who can afford it may accept. Others are auto-rejected, and that looks like an ordinary rejection to everyone else.
3. If anyone accepted, the proposer confirms one of them or cancels. If nobody accepted, the offer can't be proposed again that turn.

`max_trade_proposals_per_turn` and `max_actions_per_turn` guarantee every turn ends.

## Events (`events.py`)

`apply` returns `Event(kind, player, data, secret, visible_to)`. `event.view(seat)` merges `secret` into the view only for seats in `visible_to` (`None` means an all-seeing spectator).

Hidden parts:
- `steal`: the resource, visible only to thief and victim.
- `buy_dev`: the card, visible only to the buyer.
- An auto-rejected trade: the "auto" flag, visible only to the responder.

Everything else (production, discards, monopoly amounts, trades) is public, as at a real table.

## Determinism and replay (`game.py`)

`Game(seed, config)` owns the state, the RNG, the action history and the event log. `Game.replay(seed, config, actions)` rebuilds any game. `rules.apply_roll(state, d1, d2)` resolves a roll with chosen dice, for search chance nodes and tests.

## Tests

- `tests/engine/test_rules.py` and `test_trading.py` check individual rules on scripted positions built with `tests/conftest.py` helpers.
- `test_properties.py` (hypothesis) plays random games on both board types and checks at every step: resource and piece conservation, cached road length, incremental VP, masked-out actions rejected, identical replays.
