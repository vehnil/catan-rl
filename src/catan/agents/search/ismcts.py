"""Single-observer Information-Set MCTS (Cowling, Powley & Whitehouse, 2012).

Every iteration samples a fresh determinization of the hidden information
from the belief tracker, then descends one shared tree keyed by action
history. Because the legal actions differ between determinizations, UCB uses
*availability* counts (how often an action was legal when its parent was
visited) instead of parent visits. Each node keeps a per-seat reward vector,
and the player to move selects by their own component (Max^n backup), so
opponents are modelled as self-interested rather than adversarial.

Chance (dice, steals) is resolved by sampling inside each iteration ("open
loop"). Leaves are scored by an optional short default-policy rollout followed
by the heuristic ``leaf_values``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from catan.agents.search.common import SearchAgent
from catan.engine.constants import NUM_PLAYERS
from catan.engine.rules import apply
from catan.player.observation import Observation


class Node:
    __slots__ = ("children", "visits", "avail", "totals")

    def __init__(self) -> None:
        self.children: dict[int, Node] = {}
        self.visits = 0
        self.avail = 0
        self.totals = np.zeros(NUM_PLAYERS)

    def score(self, actor: int, c: float) -> float:
        if self.visits == 0:
            return math.inf
        return self.totals[actor] / self.visits + c * math.sqrt(
            math.log(max(self.avail, 1)) / self.visits
        )


class ISMCTSAgent(SearchAgent):
    name = "ismcts"

    def __init__(
        self,
        iterations: int = 150,
        c: float = 0.7,
        max_depth: int = 8,
        rollout_turns: int = 0,
        beam: int = 6,
        **kw: Any,
    ) -> None:
        super().__init__(beam=beam, **kw)
        self.iterations = iterations
        self.c = c
        self.max_depth = max_depth
        self.rollout_turns = rollout_turns
        self.last_root: Node | None = None

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "iterations": self.iterations,
            "c": self.c,
            "max_depth": self.max_depth,
            "rollout_turns": self.rollout_turns,
        }

    def search(self, obs: Observation) -> int:
        root = Node()
        for _ in range(self.iterations):
            self._iterate(root, obs)
        self.last_root = root
        return max(root.children, key=lambda a: root.children[a].visits)

    def _iterate(self, root: Node, obs: Observation) -> None:
        s = self.sample_state(obs)
        node, path, depth = root, [root], 0
        first = True
        while not s.done and depth < self.max_depth:
            if not first:
                self.advance_default(s, self.rng)
                if s.done:
                    break
            actions = self.candidates(s, obs.legal_mask if first else None)
            first = False
            for a in actions:
                child = node.children.get(a)
                if child is not None:
                    child.avail += 1
            untried = [a for a in actions if a not in node.children]
            actor = s.actor
            if untried:
                a = int(self.rng.choice(untried))
                child = node.children[a] = Node()
                child.avail = 1
                apply(s, a, self.rng, validate=False)
                path.append(child)
                break
            a = max(actions, key=lambda x: node.children[x].score(actor, self.c))
            apply(s, a, self.rng, validate=False)
            node = node.children[a]
            path.append(node)
            depth += 1
        self.rollout(s, self.rollout_turns, self.rng)
        reward = self.leaf_values(s)
        for n in path:
            n.visits += 1
            n.totals += reward

    def root_stats(self) -> dict[str, Any]:
        """Visit counts and mean reward per root action (for debugging / analysis)."""
        from catan.engine.actions import describe

        if self.last_root is None:
            return {}
        return {
            describe(a): {
                "visits": n.visits,
                "mean": (n.totals / max(n.visits, 1)).round(3).tolist(),
            }
            for a, n in sorted(self.last_root.children.items(), key=lambda kv: -kv[1].visits)
        }
