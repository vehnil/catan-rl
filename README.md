# catan-rl

A research platform for training and comparing agents that play 4-player base-game Catan:

- a complete, deterministic **rules engine**, including domestic trading;
- a **player layer** that sees only what a real player would, with card counting and a belief tracker (forward filter + Viterbi);
- **agents**: random and greedy baselines, bandits, Max^n / Paranoid search, ISMCTS, and PPO self-play;
- a **simulation platform**: parallel runner, tournaments, OpenSkill ratings, a SQLite results store, replays, policy checkpoints, and a browser game viewer.

## Quick start

```bash
uv sync
make test                                   # full test suite (~20 s)
uv run --no-editable catan play -v          # one greedy-vs-random game, every action printed
make baselines                              # 100-game tournament, results into data/catan.db
make ppo-small                              # ~30 s PPO run; saves a policy under data/policies/
uv run --no-editable catan ratings          # leaderboard
uv run --no-editable catan view             # browse stored games in the browser
```

Always run through `uv run --no-editable` (the Makefile does). macOS keeps marking `.venv` files hidden, and Python skips hidden `.pth` files, so the editable install breaks with `No module named 'catan'`. Alternatives: `export UV_NO_EDITABLE=1`, or `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`.

## Layout

```
src/catan/
  engine/    rules only, no RL deps: board, state, actions, rules, phases, trading, scoring, events, render, game
  player/    observation (no hidden info), belief tracker, NN features, Agent base class
  agents/    baselines/  bandits/  search/  learned/ppo/   + heuristics.py, registry.py
  env/       PettingZoo AEC env, single-agent Gymnasium env
  sim/       runner, tournament, ratings, store (SQLite), replay, timeline, artifacts (policies)
  viewer/    `catan view`: local server + one-page browser game viewer
  config/    pydantic models for YAML experiment files
  cli.py     `catan` command
docs/        in-depth documentation (start at docs/README.md)
notes/       project goals and next steps
experiments/ versioned YAML configs (see experiments/README.md)
data/        generated output (gitignored): catan.db, policies/, replays/, runs/
```

## Docs

In reading order: [architecture](docs/00-architecture.md) → [engine](docs/10-engine.md) → [player & beliefs](docs/20-player-and-beliefs.md) → [agents](docs/30-agents.md) / [environments](docs/31-environments.md) → [simulation](docs/40-simulation.md) → [training](docs/50-training.md). See [docs/README.md](docs/README.md) for shortcuts by task.
