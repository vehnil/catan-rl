# Goals

Focus: **agents that reason under hidden information and trade with each other.**

## 1. A policy that estimates the value of any move

- For any legal move, estimate how it changes my chance of winning. Moves that help an opponent win count against me.
- The value is a per-seat win probability (4 numbers that sum to 1), not a single score, so "P2 gets closer to winning" shows up directly as a loss for me.
- It has to work in every phase, including trades (what I gain minus what the partner gains), robber placement and dev card plays.

## 2. Accurate hand inference

- Keep a probability distribution over every opponent's hand and dev cards.
- Rule out hands that public actions prove impossible: someone who builds a city held at least 2 wheat and 3 ore; offering a trade proves you hold the cards; so does a discard or a monopoly payout.
- Backtrack: when later evidence rules out a hand, also revise what earlier unseen steals must have taken (smoothing), not just the current estimate.
- Feed these beliefs (expected hands, uncertainty, P(can afford X)) into the policy, and into which trades it offers and accepts.

## 3. A GUI to view games

- Replay any stored game step by step on a drawn board.
- View it as a spectator (everything visible) or as one seat (opponents' hands hidden), with the belief tracker's guesses shown next to the true hands.
