# CLAUDE.md

Catan RL platform: rules engine → player layer (observation + beliefs) → agents → envs → simulation/storage. Python 3.12, uv, PyTorch.

## Commands

Always run through `uv run --no-editable` (or the Makefile). macOS keeps marking `.venv` files hidden, and Python skips hidden `.pth` files, so the editable install breaks with `No module named 'catan'`. Tests also work without the install (`pythonpath = ["src", "."]`).

```bash
make test          # uv run --no-editable pytest   (~20 s; `-m "not slow"` to skip slow tests)
make lint          # ruff check;  make fmt  to format + autofix
make typecheck     # mypy src (pydantic plugin)
uv run --no-editable catan --help
```

## Architecture rules

- `engine/` is pure: no torch/gym/pettingzoo imports. `rules.apply(state, action, rng)` mutates in place and returns `Event`s. All randomness goes through `rng`, so `(seed, rules, actions)` must reproduce a game. Don't add hidden RNG.
- State arrays indexed by vertex/edge have a trailing **sentinel** slot (`NUM_VERTICES`/`NUM_EDGES`) that stays empty. Padded adjacency tables in `board.TOPOLOGY` point at it. Keep it empty.
- Hidden information is carried only by `Event.secret` + `visible_to`. Anything agents see must come from `observe()` or `event.view(seat)`. `tests/player/test_observation.py` checks for leaks. When adding an event, decide what is public, and update `player/belief.py` if the event moves cards.
- Action indices are a flat space (`engine/actions.py`, 488 actions). Changing block sizes or order invalidates trained PPO policies and stored replays.
- Limits are action counts (`RulesConfig`), never wall-clock time.
- Agents subclass `player.base.Agent` (`reset`/`observe`/`act`). `config()` must return the constructor kwargs: `sim/artifacts.load_policy` rebuilds agents from it. Register new agents in `agents/registry.py`.
- `agents/heuristics.py` is shared by greedy, bandit rollouts and search leaves.
- Search agents (`agents/search/common.py`) branch only in `BRANCH_PHASES`. Routine decisions use the greedy default policy.

## Testing conventions

- `tests/conftest.py` helpers (`main_phase_game`, `place`, `road`, `give`, `path_edges`) build scripted positions. `give` moves cards from the bank so resource conservation holds.
- `tests/engine/test_properties.py` (hypothesis) checks invariants at every step of random games: conservation, piece counts, cached road length, VP, masks vs `apply`, replay determinism. Keep it passing after any rules change.
- Mark long tests `@pytest.mark.slow`.

## Data

`data/` is gitignored: `catan.db` (SQLite: runs, matches, match_players, policies, ratings), `policies/<id>/`, `replays/<run>/`, `runs/<run>/metrics.jsonl`.
