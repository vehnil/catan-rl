# catan-rl documentation

Each page depends only on the pages above it in the graph. Pages with the same tens digit (`30-`, `31-`) can be read in either order.

```
                 00 architecture            overview, read first
                        │
                 10 engine                  engine/
                        │
                 20 player & beliefs        player/
                   ┌────┴────┐
            30 agents     31 environments   agents/   env/
                   │         │
            40 simulation    │              sim/  config/  cli.py
                   └────┬────┘
                 50 training                agents/learned/ppo/{trainer,league}.py
```

| Doc | Covers |
|---|---|
| [00 architecture](00-architecture.md) | Layers, one game step, design principles |
| [10 engine](10-engine.md) | Board, state, turn phases, the 488-action space, trading, limits, events, determinism |
| [20 player & beliefs](20-player-and-beliefs.md) | What a seat can see, card counting, forward filter and Viterbi, determinization, NN features |
| [30 agents](30-agents.md) | Every agent and how it decides, spec strings, the `Agent` API, writing a new agent |
| [31 environments](31-environments.md) | PettingZoo AEC and single-agent Gymnasium wrappers |
| [40 simulation](40-simulation.md) | Runner, tournaments, ratings, the SQLite store, replays, saved policies, CLI |
| [50 training](50-training.md) | PPO loop, self-play league, configs, metrics, tuning tips |

## Short paths

- **Changing a rule:** 00 → 10
- **Writing an agent:** 00 → 10 → 20 → 30
- **Plugging in an outside RL library:** 00 → 10 → 20 → 31
- **Running tournaments:** 00 → 30 (skim) → 40
- **Training PPO:** all of them, in order

## Where the layering bends

The graph matches the real imports, with two exceptions:

- `agents/registry.py` imports `sim.artifacts` lazily, inside `make_agent`, only to resolve `policy:<id>` specs. Every other spec builds without the simulation layer.
- `agents/learned/ppo/trainer.py` and `league.py` live in `agents/` but import the env, config and sim layers, so they're documented in 50.

`config/` depends only on the engine (for `RulesConfig`). Tournament configs are described in 40, training configs in 50.
