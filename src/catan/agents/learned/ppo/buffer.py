"""On-policy rollout storage with GAE(lambda) advantages.

Rows from several envs can be interleaved (parallel workers send chunks), so
each row records which env it came from. GAE runs over each env's rows in
order; minibatches draw from all filled rows.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

SEGMENT_FIELDS = ("obs", "masks", "actions", "logprobs", "values", "rewards", "dones")


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, n_actions: int) -> None:
        self.size = size
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.masks = np.zeros((size, n_actions), bool)
        self.actions = np.zeros(size, np.int64)
        self.logprobs = np.zeros(size, np.float32)
        self.values = np.zeros(size, np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)  # episode ended after this step
        self.envs = np.zeros(size, np.int64)
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def add(self, obs, mask, action, logprob, value, reward, done, env: int = 0) -> None:  # noqa: ANN001
        i = self.ptr
        self.obs[i], self.masks[i], self.actions[i] = obs, mask, action
        self.logprobs[i], self.values[i] = logprob, value
        self.rewards[i], self.dones[i] = reward, done
        self.envs[i] = env
        self.ptr += 1

    def add_segment(self, seg: dict[str, np.ndarray], env: int) -> None:
        """Append consecutive rows from one env (arrays keyed like the buffer's fields)."""
        i, k = self.ptr, len(seg["actions"])
        for name in SEGMENT_FIELDS:
            getattr(self, name)[i : i + k] = seg[name]
        self.envs[i : i + k] = env
        self.ptr += k

    def compute_returns(self, last_values: dict[int, float], gamma: float, lam: float) -> None:
        """``last_values[e]``: value of env ``e``'s current observation (after its last row)."""
        adv = np.zeros(self.ptr, np.float32)
        for e in np.unique(self.envs[: self.ptr]):
            rows = np.flatnonzero(self.envs[: self.ptr] == e)
            last_value = last_values[int(e)]
            gae = 0.0
            for k in reversed(range(len(rows))):
                t = rows[k]
                next_value = last_value if k == len(rows) - 1 else self.values[rows[k + 1]]
                nonterminal = 1.0 - self.dones[t]
                delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
                gae = delta + gamma * lam * nonterminal * gae
                adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values[: self.ptr]

    def minibatches(
        self, batch_size: int, rng: np.random.Generator, device: str
    ) -> Iterator[dict[str, torch.Tensor]]:
        idx = rng.permutation(self.ptr)
        adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        for start in range(0, self.ptr, batch_size):
            b = idx[start : start + batch_size]
            yield {
                "obs": torch.as_tensor(self.obs[b], device=device),
                "masks": torch.as_tensor(self.masks[b], device=device),
                "actions": torch.as_tensor(self.actions[b], device=device),
                "logprobs": torch.as_tensor(self.logprobs[b], device=device),
                "advantages": torch.as_tensor(adv[b], device=device),
                "returns": torch.as_tensor(self.returns[b], device=device),
            }

    def reset(self) -> None:
        self.ptr = 0
