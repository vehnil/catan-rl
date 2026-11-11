# Agents

All agents live in `src/catan/agents/` and are built from **spec strings** (`registry.py`):

```
name                         greedy
name:key=value,key=value     ismcts:iterations=300,max_depth=6
policy:<id | name | path>    policy:08ceaf5856
```

Values are parsed as int, float, bool or string. `catan agents` lists the names. In CLI `--agents` lists, separate specs with `;` when they contain commas.

## The `Agent` API

Every agent subclasses `player.base.Agent`:

```python
class Agent(ABC):
    name: str                     # registry name
    track_beliefs: bool = True    # base class maintains self.belief
    def reset(self, seat, rng): ...
    def observe(self, event_view): ...         # updates self.belief
    @abstractmethod
    def act(self, obs: Observation) -> int: ...
    def config(self) -> dict: ...              # constructor kwargs, stored in policy manifests
    def save_files(self, directory) -> dict: ...   # weights, statistics
    def load_files(self, directory, manifest): ...
```

## Where each algorithm fits

- **Viterbi** isn't a control algorithm: it decodes the most likely hidden sequence in an HMM, which is exactly what opponents' hands are. It lives in the belief tracker's max-product filter ([player & beliefs](20-player-and-beliefs.md)).
- **Bandits** suit one-shot choices with noisy payoffs: `opening_bandit`, `meta_bandit`, and UCB inside ISMCTS.
- **Minimax** assumes 2-player zero-sum perfect information; Catan is 4-player, non-zero-sum, stochastic and partly hidden. `maxn` and `paranoid` (with expectimax dice) are kept mainly as comparison points.
- **ISMCTS** is the strongest classical approach to hidden-information games.
- `random` and `greedy` are baselines; `ppo` is the learned policy.

## Shared heuristics (`heuristics.py`)

`evaluate(state, p, weights)` scores a position for seat `p`. It's used by greedy, bandit rollouts and search leaves:

```
100·VP + 3·production_pips + 6·#resource_types + 25·progress_to_next_build
− 6·cards_over_7 + 18·unplayed_dev_cards + 8·knights + 3·min(road_length, 8)
+ 1.5·pips_of_best_reachable_settlement_spot
```

(The coefficients shown are the `balanced` defaults.) `Weights` is a dataclass, so these coefficients are tunable. `PRESETS` holds four strategy styles: `balanced`, `expansion`, `cities`, `development`.

Other helpers: `vertex_pips` (per-vertex resource pips, cached per board and robber position), `vertex_score` (settlement desirability, including diversity and port value), `robber_hex_score`.

## Baselines

### `random`
Uniform over legal actions. Domestic proposals get 5% of the total probability (`trade_weight`); otherwise their 230 actions would swamp everything else. No belief tracking. Random games take about 1,000 steps.

### `greedy[:preset=…]`
One-step lookahead on its best guess of the state (opponents' hands filled in from the belief tracker's most likely values):

- **Builds, bank trades, year of plenty, free roads:** apply each to a copy, keep the best strict improvement in `evaluate`. Requiring strict improvement rules out trade loops.
- **Buying a dev card:** cost plus a fixed bonus.
- **Setup:** best `vertex_score`; roads toward the best open spot.
- **Robber:** maximize damage to opponents' production (weighted by their VP), never block yourself. Steal from the VP leader.
- **Discard:** the card whose loss hurts `evaluate` least.
- **Dev plays:** knight if the robber blocks you or largest army is one knight away; monopoly when opponents are expected to hold ≥ 3 of a resource (from the belief tracker); road building when no settlement spot is reachable.
- **Trading:** propose the offer that most improves your position (at most `max_proposals` per turn). Accept if your evaluation improves by more than `accept_margin` and the proposer isn't within 2 VP of winning. Confirm with the lowest-VP acceptor.

Beats 3× random in 100/100 games, averaging about 62 turns.

## Bandits (`bandits/`)

- `UCB1` and `BetaThompson`: standalone multi-armed bandits with `select`, `update`, `best` and JSON round-trip.
- **`opening_bandit[:budget=48,top_k=6,rollout_turns=8,algo=ucb|thompson]`**: treats each candidate initial settlement (the top-k by `vertex_score`) as an arm. Pulling an arm means sampling hidden info, placing the settlement, playing everyone greedily (ε = 0.1) for `rollout_turns` turns after setup, and scoring `sigmoid((my_eval − best_rival_eval) / 60)`. It commits to the most-visited arm, then plays greedy for the rest of the game.
- **`meta_bandit[:arms=a|b|c,algo=thompson|ucb,stats_path=…]`**: picks a strategy preset per game and updates on win/loss from the `GAME_OVER` event. Set `stats_path` to keep learning across games (each tournament game builds a fresh agent). Use `workers=1` when the learning curve matters.

## Search (`search/`)

All search agents extend `SearchAgent` (`common.py`), which handles three things:

- **Where to branch.** Only at strategic phases (setup, roll/knight, robber, main, road building). Routine decisions (discards, trade responses, steal targets, confirmations) use the greedy policy and don't consume depth.
- **What to consider.** A beam of the `beam` best actions by one-step evaluation, plus non-rankable moves (end turn, dev plays, buy dev), plus the greedy policy's choice. Domestic proposals are excluded from search.
- **Hidden information.** Search runs over `belief.determinize(...)` samples.

### `maxn[:depth=2,beam=5,determinizations=1]`
Max^n: each node returns a utility vector and the player to move maximizes their own component. A ROLL is a chance node averaged over the 11 dice sums. Leaves use the per-seat `sigmoid(lead over best rival)`.

### `paranoid[:depth=3,beam=5]`
Assumes every opponent minimizes *your* utility, which turns the game into me vs. a coalition and makes alpha-beta pruning valid. It's pessimistic, but it's a useful contrast to Max^n.

### `ismcts[:iterations=150,c=0.7,max_depth=8,rollout_turns=0,beam=6]`
Single-observer information-set MCTS (Cowling et al. 2012):
- Each iteration samples a new determinization and walks one shared tree keyed by action history.
- UCB uses **availability counts** (how often an action was legal at that node), not parent visits, because legal moves differ between samples.
- Nodes store a per-seat reward vector, and the player to move selects by its own component.
- Dice and steals are sampled inside each iteration ("open loop").
- Leaves: an optional greedy rollout of `rollout_turns`, then the heuristic.
- `root_stats()` shows visits and mean rewards per root action.

## Learned (`learned/`)

### `ppo[:hidden=[512,256],deterministic=false]`
An actor-critic MLP over the 1,388 features. Illegal logits are masked to −1e9 before the softmax. It samples by default, or takes the argmax with `deterministic=true`. Saved as `weights.safetensors`; `export_onnx` for use outside Python. Training is covered in [training](50-training.md).

## Cost per decision (rough)

| Agent | One game vs 3× greedy |
|---|---|
| greedy | ~0.3 s |
| opening_bandit (budget 48) | ~2 s |
| maxn depth 2 | ~1 s |
| paranoid depth 3 | ~3 s |
| ismcts 60 iterations | ~13 s |

## Writing a new agent

Subclass `Agent` (see [above](#the-agent-api)):

```python
from catan.player.base import Agent

class MyAgent(Agent):
    name = "mine"
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    def act(self, obs):
        legal = obs.legal_mask.nonzero()[0]
        ...                                   # self.belief is up to date
        return int(choice)
    def config(self):
        return {"temperature": self.temperature}   # must round-trip through __init__
```

Then add `"mine": "catan.agents.my_module:MyAgent"` to `_REGISTRY` in `registry.py`, or call `registry.register("mine", MyAgent)` at runtime. Add it to `FAST_SPECS` in `tests/agents/test_agents.py`; that test plays a full game and fails on any illegal action.
