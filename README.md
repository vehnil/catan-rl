# catan-rl

A research platform for training and comparing agents that play 4-player base-game Catan. It has:

- a complete, deterministic **rules engine**, including domestic trading;
- a **player layer** that sees only what a real player would, with card counting and a belief tracker (forward filter + Viterbi);
- **agents**: random and greedy baselines, bandits, Max^n / Paranoid search, ISMCTS, and PPO self-play;
- a **simulation platform**: parallel runner, tournaments, OpenSkill ratings, a SQLite results store, replays, and policy checkpoints.

## Quick start

```bash
uv sync
make test                                   # full test suite (~20 s)
uv run --no-editable catan play -v          # one greedy-vs-random game, every action printed
make baselines                              # 100-game tournament, results into data/catan.db
make ppo-small                              # ~30 s PPO run; saves a policy under data/policies/
uv run --no-editable catan ratings          # leaderboard
```

> **Why `--no-editable`?** On this Mac, files inside `.venv` keep getting the macOS "hidden" flag. Python ignores hidden `.pth` files, which silently breaks uv's editable install (`ModuleNotFoundError: No module named 'catan'`). `--no-editable` installs a normal copy, and `[tool.uv] cache-keys` rebuilds it whenever a source file changes. The Makefile uses it everywhere. Alternatives: `export UV_NO_EDITABLE=1`, or `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`.

## Layout

```
src/catan/
  engine/    rules only, no RL deps: board, state, actions, rules, phases, trading, scoring, events, render, game
  player/    observation (no hidden info), belief tracker, NN features, Agent base class
  agents/    baselines/  bandits/  search/  learned/ppo/   + heuristics.py, registry.py
  env/       PettingZoo AEC env, single-agent Gymnasium env
  sim/       runner, tournament, ratings, store (SQLite), replay, artifacts (policies)
  config/    pydantic models for YAML experiment files
  cli.py     `catan` command
experiments/ example YAML configs
data/        generated output (gitignored): catan.db, policies/, replays/, runs/
```

## Engine

- **Actions** are a flat discrete space of 488 indices (`engine/actions.py`). Every phase exposes a legal-action mask. Seat-relative arguments (steal target, trade partner) are offsets, so one policy can play any seat.
- **Turn flow** (`engine/phases.py`): setup snake order → roll → (7: discard, robber, steal) → main → end turn. The player who must act (`state.actor`) isn't always the player whose turn it is: discards and trade responses happen out of turn.
- **Domestic trading**: propose one of 230 enumerated offers (give 1–2 cards for 1–2 cards) → each opponent accepts or rejects in seat order → the proposer confirms one acceptor or cancels. You can only offer cards you hold. A rejected offer can't be repeated that turn. Responders who can't afford the trade are skipped, and to everyone else that looks like an ordinary rejection.
- **Limits** count actions, never wall-clock time, so games stay reproducible (`RulesConfig`): `max_trade_proposals_per_turn=3`, `max_actions_per_turn=50` (after that only END_TURN is legal), `max_turns_per_game=500` (the game is truncated and players are ranked by VP).
- **Determinism**: all randomness flows through one seeded generator, so `(seed, rules, actions)` reproduces a game exactly. That's all a replay stores.

## Player layer

- `observe(state, seat)` copies public information plus the seat's own cards. Tests check that changing hidden information never changes an observation.
- `BeliefTracker` treats each opponent's hand as a hidden Markov process. Hands are fully known except after **unseen robber steals** between other players. Every public event that requires holding cards (builds, offers, accepts, discards, monopoly payouts) filters out impossible hands.
  - Forward filter → posterior over hands (`hand_marginals`, `map_hand`, `prob_can_afford`)
  - Viterbi (max-product) → the hand at the end of the most likely steal history (`viterbi_hand`)
  - Dev cards: unseen pool and per-type probabilities, expected hidden VP
  - `determinize()` samples a full game state that's consistent with everything seen, for search agents.

## Agents

Build any agent from a spec string: `name` or `name:key=value,...`, or `policy:<id|name|path>` for a stored policy. Run `catan agents` to list them.

| Spec | What it is |
|---|---|
| `random` | Uniform legal play (trade proposals down-weighted) |
| `greedy[:preset=expansion\|cities\|development]` | One-step lookahead over a hand-tuned evaluation function, plus heuristics for the robber, trades and dev cards |
| `opening_bandit[:budget=48,top_k=6,algo=ucb\|thompson]` | Initial placements chosen as a best-arm bandit over rollouts, then greedy |
| `meta_bandit[:stats_path=...]` | Picks a strategy preset each game (Thompson or UCB) and learns across games |
| `maxn[:depth=2,beam=5]` | Max^n search with expectimax dice nodes on a sampled hidden state |
| `paranoid[:depth=3]` | Assumes opponents minimize your score; alpha-beta pruning |
| `ismcts[:iterations=150,max_depth=8,rollout_turns=0]` | Single-observer information-set MCTS with availability-count UCB |
| `ppo[:hidden=[512,256]]` | PyTorch actor-critic with action masking |

All search agents use the greedy policy for routine decisions (discards, trade responses) and search only strategic ones, over a beam of top-ranked candidate moves.

## Simulation

```bash
catan sim -a "greedy,random,random,random" --games 200 --workers 8
catan tournament experiments/strategies_round_robin.yaml   # fixed | round_robin | gauntlet
catan train experiments/ppo_league.yaml                    # PPO vs random/greedy + its own snapshots
catan replay data/replays/<run>/<game>.json.gz --step      # re-simulate and verify
catan export <policy> --format json|onnx
catan runs | catan policies | catan ratings | catan benchmark
```

- Every tournament seat is rotated, because placing first is an advantage.
- **SQLite** (`data/catan.db`) tables: `runs`, `matches`, `match_players`, `policies`, `ratings`. Query it directly for analysis.
- **Policies** are saved as `data/policies/<id>/manifest.json` (kind, constructor config, parent policy, run, git SHA, metrics), plus `weights.safetensors` for neural agents or `state.json` for bandit statistics. Non-neural policies are plain text. `export --format json` dumps weights as lists (only sensible for small networks). ONNX needs `uv add onnx onnxscript`.

## Measured so far (on this machine)

- Engine: about 550 random games/min on one core, about 2,200/min with 8 workers.
- Greedy vs 3× random: 100/100 wins.
- `ppo_small` (20k steps, ~30 s): training win rate 0.33 → 0.46 against random opponents; final eval 40% over 20 games (chance is 25%).
