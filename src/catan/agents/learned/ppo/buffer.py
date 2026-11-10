"""On-policy rollout storage with GAE(lambda) advantages."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch


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
        self.ptr = 0

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def add(self, obs, mask, action, logprob, value, reward, done) -> None:  # noqa: ANN001
        i = self.ptr
        self.obs[i], self.masks[i], self.actions[i] = obs, mask, action
        self.logprobs[i], self.values[i] = logprob, value
        self.rewards[i], self.dones[i] = reward, done
        self.ptr += 1

    def compute_returns(self, last_value: float, gamma: float, lam: float) -> None:
        adv = np.zeros(self.size, np.float32)
        gae = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values

    def minibatches(
        self, batch_size: int, rng: np.random.Generator, device: str
    ) -> Iterator[dict[str, torch.Tensor]]:
        idx = rng.permutation(self.size)
        adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        for start in range(0, self.size, batch_size):
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
