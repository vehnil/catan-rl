---
name: notes
description: Review or edit catan-rl's notes/ (findings, next-steps, drawing-board, goals) - record findings, fix stale or contradictory notes.
---

# notes/

- `findings.md`: numbered findings (claim + evidence), then hypotheses H1–H7, then sources. No plans.
- `next-steps.md`: the plan as checkboxes. Done items stay ticked with a one-line result and links.
- `drawing-board.md`: the mid-v7 decision note. At most a short "where this ended up" line.
- `goals.md`: long-term goals; rarely changes.
- Experiment detail lives in `experiments/vN_*/vN-proposal.md` (Results section) and `experiments/README.md` (results line).

## Conventions

- Findings are append-only. Never renumber or delete; if a later result changes one, edit it in place: "v9 contradicts this: see 26".
- A finding is a bold one-sentence claim, then the experiment, games, seeds, numbers and noise. Say whether it's a comparison set before the run, a post-hoc look, or a profiler number (profiler ≠ real run).
- Noise: 400-game margin ±0.15–0.2 VP per policy; 100-game evals ±0.3–0.5 (finding 18); rating gaps < ~3 (finding 16).
- Keep the findings.md header's per-experiment ranges current. When a hypothesis is tested, add a `*Tried* in vN: …, findings X–Y.` line.
- Relative links, dates like "12 Nov 2026", real numbers.

## Review

Check: every finished experiment (`data/experiments/*/report.md`) is recorded in all four places; new numbers contradict no older finding; no stale "running/next/vN" labels or undone-but-done items; a "Tried" line on each tested hypothesis; patterns across findings that deserve their own finding. Report the issues before editing unless asked to edit. Check every number against its source file, not memory.
