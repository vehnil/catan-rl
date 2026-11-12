# Simulation, storage and the CLI

`src/catan/sim/` runs games, schedules tournaments, rates agents and stores everything.

## Runner (`runner.py`)

- `play_game(agents, seed, rules) -> GameRecord` plays one game (agents[i] sits in seat i, and seat 0 places first).
- `GameRecord` holds the seed, rules, agent spec per seat, winner, turns, truncated flag, VP and rank per seat, step count, duration, and the full action list.
- Ranks: the winner is 1, the rest are ordered by VP, and ties share a rank.
- `run_matchups(matchups, rules, workers)` runs `Matchup(specs, seed)` jobs, sequentially or across a process pool. Workers rebuild agents from spec strings, so nothing heavy is pickled. Results are identical either way (tested).

## Tournaments (`tournament.py`)

Every format rotates seats, because placing first is an advantage.

| `format` | Lineups |
|---|---|
| `fixed` | Exactly 4 `agents`, the same every game |
| `round_robin` | Every 4-agent combination of `agents` (with repetition if fewer than 4) |
| `gauntlet` | `candidate` in every game; the other 3 seats drawn from `agents` |

`run_tournament(cfg)` creates a run, plays the games, saves replays, writes each match to the store, updates ratings, and records a summary with per-agent stats and 95% Wilson confidence intervals.

```yaml
# experiments/v1_examples/strategies_round_robin.yaml
name: strategies
format: round_robin
agents: [greedy, "greedy:preset=expansion", "greedy:preset=cities", "greedy:preset=development", "opening_bandit:budget=32"]
games: 200
workers: 8
seed: 1
rules: { board: random }
```

## Ratings (`ratings.py`)

**OpenSkill Plackett-Luce**, a TrueSkill-like Bayesian rating that handles 4-player rankings directly. Each agent has `mu` (skill) and `sigma` (uncertainty); the leaderboard sorts by `mu − 3·sigma`. When one agent fills several seats (self-play), each seat is rated separately and the results averaged. `wilson_interval(wins, games)` gives win-rate confidence intervals.

## Store (`store.py`), at `data/catan.db`

Plain SQLite (stdlib `sqlite3`, WAL mode), so you can query it from anything.

| Table | Rows |
|---|---|
| `runs` | id, kind (`tournament` / `train`), name, config JSON, status, metrics JSON, timestamps |
| `matches` | id, run_id, seed, rules JSON, winner seat, turns, steps, truncated, duration, replay path |
| `match_players` | match_id, seat, agent spec, vp, rank, is_winner |
| `policies` | id, name, kind, path, parent_id, run_id, manifest JSON |
| `ratings` | agent, mu, sigma, games |

Example: win rate by seat for greedy:

```sql
SELECT seat, AVG(is_winner) FROM match_players WHERE agent = 'greedy' GROUP BY seat;
```

## Replays (`replay.py`)

A replay is a gzipped `GameRecord` JSON in `data/replays/<run_id>/`. Since the engine is deterministic, `(seed, rules, actions)` is enough to rebuild any position.

- `rebuild(rec, upto=n)`: the game after n actions
- `step_through(rec, seat)`: yields (step, actor, action description, rendered board), optionally from one seat's point of view
- `verify(rec)`: re-simulates and checks the recorded outcome

## Timeline and viewer (`timeline.py`, `catan/viewer/`)

`build_timeline(rec)` re-simulates a replay into plain JSON (about 1 MB for a 300-step game):

- `meta`: seed, agents, result, `verified`, plus name tables (resources, dev cards, trade offers).
- `board`: hex centers, corner positions and edges in unit-hex coordinates (pointy top, circumradius 1), plus ports.
- `steps[i]`: the action that led to position i (`None` for i = 0), its events, a state snapshot, and every seat's belief tracker summary. Each event holds the spectator view; events with secrets add `views[seat]` for the seats that don't see them.

`catan view [replay]` starts a stdlib HTTP server on 127.0.0.1 and opens the viewer page (one HTML file, no build step). With no replay given, it opens a picker listing every match in the store that has a replay file. The page has:

- an SVG board with pieces and the robber, with this step's changes highlighted;
- step and turn navigation, play/pause, and keyboard shortcuts (← → step, Shift+← → turn, Space play/pause);
- a "view as" seat selector. In seat view, opponents' hands become the tracker's guesses (expected count, MAP, Viterbi, number of hypotheses), and "show truth" puts the real hand underneath with mismatches in red.

The URL keeps the replay, step and seat, so a position can be linked to.

## Policies (`artifacts.py`)

```
data/policies/<id>/
  manifest.json          kind, constructor config, files, parent_id, run_id, metrics, git SHA
  weights.safetensors    neural agents
  state.json             extra state (e.g. meta-bandit arm statistics)
```

- `save_policy(agent, data_dir, store=...)` writes the folder and registers it in the store.
- `load_policy(path)` rebuilds the agent from `kind` + `config` and loads its files.
- `resolve_policy_path(ref)` accepts a path, an id or a name.
- `export_policy(path, "json" | "onnx")`. JSON dumps the manifest plus weights as nested lists (only sensible for small networks). ONNX needs `uv add onnx onnxscript`.

Any stored policy can play in a tournament as `policy:<id>`.

## CLI (`cli.py`)

Run with `uv run --no-editable catan …`.

| Command | What it does |
|---|---|
| `catan agents` | List agent names |
| `catan play -a "greedy,random,random,random" -v` | One game, optionally printing every action |
| `catan sim -a … --games 200 --workers 8` | Fixed-lineup tournament |
| `catan tournament experiments/<version>/x.yaml` | Tournament from a config |
| `catan train experiments/v1_examples/ppo_small.yaml` | PPO training |
| `catan curve <run> --vs name=a,b,c [--vs …]` | Evaluate a run's starting policy, checkpoints and final policy against fixed lineups (same games for each), to spot overfitting |
| `catan health <run> [--windows N]` | Summarize a training run's PPO health metrics (entropy, explained variance, KL, clip fraction, value loss, grad norm) over windows, flagging warning signs |
| `catan replay <file> [--step] [--seat N]` | Re-simulate, render, verify |
| `catan view [file] [--port] [--no-open]` | Browser viewer for stored games |
| `catan timeline <file> [-o out.json]` | Export a replay as per-step JSON |
| `catan export <policy> -f json\|onnx` | Export a policy |
| `catan ratings` · `catan policies` · `catan runs` | Inspect the store |
| `catan benchmark --games 200 --workers 8` | Throughput |

All commands accept `--data-dir` (default `data/`).

Measured throughput: about 550 random games/min on one core, about 2,200/min with 8 workers.
