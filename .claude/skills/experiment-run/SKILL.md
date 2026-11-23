---
name: experiment-run
description: Check on, estimate the ETA of, stop, resume or re-analyse a running catan-rl experiment, including after sleep or a dead battery.
---

# Monitoring experiments

Code: `src/catan/sim/experiment.py`, `dashboard.py`; docs: `docs/40-simulation.md`. State is in `data/experiments/<name>/` (`state.json`, `driver.log`, `<run>.log`, `report.md`); per-run metrics are in `data/runs/<id>/metrics.jsonl`.

## Checking

- Start with `catan experiment status <name>`, then `tail driver.log` and `pgrep -fl "catan|caffeinate"`.
- Measure speed as wall-clock steps over the last 15–30 min of `metrics.jsonl`, not one `sps_recent` (it dips at eval rows).
- After the first hour, trust the dashboard ETA. Before that, remember the speed roughly halves by about 10M as the snapshot pool fills, and add about 40 min for analysis.

## The machine (M1 laptop)

- `caffeinate -i` only stops idle sleep. A closed lid or a dead battery (`pmset -g batt`, `pmset -g log | grep -E "Sleep|Wake"`) freezes the runs; they continue on wake with nothing lost.
- About 8 rollout workers is the ceiling. Check what else is using the CPU (`ps -Ao pcpu,comm -r | head`) and new swapouts (`vm_stat`). Ignore the load average.

## Stop, resume, re-analyse

- Ask before stopping, killing or relaunching anything. `catan experiment stop` loses the work since each run's last `train_state.pt`, so say how much that is.
- Run the same yaml again to resume. Only crashes count toward `max_resumes`. A resumed run isn't bit-identical; note that in the write-up.
- `catan experiment analyse <name>` reruns the whole analysis with the current code. For one extra comparison, `catan compare` is quicker.
- Never edit `data/experiments/<name>/code/` or `configs/`.
