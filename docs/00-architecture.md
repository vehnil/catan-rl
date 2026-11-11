# Architecture

## The layers

```
            ┌──────────────────────────────────────────────────────────┐
            │  sim/  runner · tournaments · ratings · store · replays  │  ← experiments
            │  cli.py  config/                                         │
            └──────────────▲───────────────────────────▲───────────────┘
                           │ plays games between       │ trains through
            ┌──────────────┴──────────────┐   ┌────────┴───────────────┐
            │  agents/                    │   │  env/                  │
            │  random · greedy · bandits  │   │  PettingZoo AEC        │
            │  maxn · paranoid · ismcts   │   │  Gymnasium (1 learner) │
            │  ppo                        │   └────────▲───────────────┘
            └──────────────▲──────────────┘            │
                           │ act(observation)          │
            ┌──────────────┴───────────────────────────┴───────────────┐
            │  player/  observation · belief tracker · features · Agent│  ← what a seat may know
            └──────────────────────────▲───────────────────────────────┘
                                       │ events, state
            ┌──────────────────────────┴───────────────────────────────┐
            │  engine/  board · state · actions · rules · events       │  ← the game itself
            └──────────────────────────────────────────────────────────┘
```

Each layer imports only the layers below it (exceptions: [docs/README](README.md#where-the-layering-bends)). The engine has no RL dependencies, so it can be tested, profiled or rewritten on its own.

## One game step

`sim/runner.py:play_game` is the reference loop; the envs and search rollouts are variations of it:

```python
game = Game(seed, rules)                          # engine: board, shuffled dev deck, RNG
for seat, agent in enumerate(agents):
    agent.reset(seat, rng)                        # agent gets a fresh BeliefTracker
while not game.done:
    p = game.actor                                # who must act now (not always whose turn it is)
    action = agents[p].act(observe(game.state, p))   # player layer: redacted view + legal mask
    for event in game.step(action):               # engine: validate, mutate state, emit events
        for seat, agent in enumerate(agents):
            agent.observe(event.view(seat))       # each seat sees only its share of each event
```

- **State** is owned by the engine. Agents only ever get an `Observation` (`player/observation.py`).
- **Events** come from `rules.apply`: public data plus optional secret data visible to some seats. They drive belief tracking, replays and logs.
- **Actions** are integers in `[0, 488)`, with legality always available as a boolean mask.

## Design principles

- **Determinism.** All randomness (board layout, dev deck shuffle, dice, which card a steal takes) comes from one seeded `numpy` generator, so `(seed, rules, action list)` replays a game exactly and replay files are tiny. Every budget (turn caps, action caps, MCTS iterations, rollouts) is a count, never seconds, so results don't depend on machine speed.
- **No information leaks.** Hidden data exists only in `GameState` and `Event.secret`. `observe()` copies public fields and the seat's own cards. A test perturbs hidden data and checks that observations don't change.
- **One flat action space with masks.** Simple for neural policies (one softmax), search (enumerate the mask) and storage (ints). Seat-relative arguments (steal target, trade partner) let one policy play any seat.
- **Agents are spec strings**, e.g. `ismcts:iterations=200`. Worker processes rebuild agents from specs; saved policies rebuild them from `config()`.

## Where things live

| Want to change... | Look in |
|---|---|
| A rule | `engine/rules.py` (+ `tests/engine/test_rules.py`; the property tests guard invariants) |
| What players can see / card counting | `player/observation.py`, `player/belief.py` |
| Neural-net inputs | `player/features.py` (changing it invalidates trained PPO policies) |
| Position evaluation used by greedy and search | `agents/heuristics.py` |
| How agents are named and built | `agents/registry.py` |
| Tournament formats, ratings | `sim/tournament.py`, `sim/ratings.py` |
| Database schema | `sim/store.py` |
| Config fields | `config/models.py` |
