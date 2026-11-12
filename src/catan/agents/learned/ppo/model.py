"""Actor-critic MLP over the flat feature vector."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from catan.agents.learned.common.distributions import masked_categorical, masked_logits
from catan.engine.actions import NUM_ACTIONS
from catan.player.features import FEATURE_DIM


class ActorCritic(nn.Module):
    def __init__(
        self,
        hidden: list[int] | tuple[int, ...] = (512, 256),
        obs_dim: int = FEATURE_DIM,
        n_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Linear(last, n_actions)
        self.value = nn.Linear(last, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        self.hidden = list(hidden)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy(z), self.value(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        # The same steps (and RNG draw) as Categorical(logits=...).sample() / .log_prob(),
        # without building a distribution object, which costs more than the network here.
        logits, value = self(obs)
        z = masked_logits(logits, mask)
        z = z - z.logsumexp(-1, keepdim=True)
        probs = torch.softmax(z, -1)
        if deterministic:
            action = probs.argmax(-1)
        else:
            action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, True).T
            action = action.reshape(probs.shape[:-1])
        return action, z.gather(-1, action.unsqueeze(-1)).squeeze(-1), value

    def policy_logits_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Policy logits in NumPy (no value head, no autograd), for frozen opponents.

        ``obs`` is one feature vector, or a ``(batch, features)`` stack: one matrix
        product per layer for the whole batch, which also reads each weight matrix from
        memory once instead of once per game. At batch size 1 this is about twice as fast
        as torch. Reads the parameters in place, so it follows in-place weight updates.
        Matches the torch forward up to float32 rounding (~1e-6)."""
        x = obs
        for kind, w, b in self._numpy_layers():
            if kind == "tanh":
                x = np.tanh(x)
                continue
            x = w @ x + b if x.ndim == 1 else x @ w.T + b
        return x

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = masked_categorical(logits, mask)
        return dist.log_prob(actions), dist.entropy(), value

    def _numpy_layers(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """NumPy views of the trunk and policy head, rebuilt if a parameter is reallocated."""
        linears = [m for m in self.trunk if isinstance(m, nn.Linear)] + [self.policy]
        key = tuple(t.data_ptr() for m in linears for t in (m.weight, m.bias))
        cache = self.__dict__.get("_np_cache")
        if cache is not None and cache[0] == key:
            return cache[1]
        layers: list[tuple[str, np.ndarray, np.ndarray]] = []
        for m in [*self.trunk, self.policy]:
            if isinstance(m, nn.Linear):
                layers.append(("linear", m.weight.detach().numpy(), m.bias.detach().numpy()))
            elif isinstance(m, nn.Tanh):
                layers.append(("tanh", np.empty(0), np.empty(0)))
            else:
                raise TypeError(f"no NumPy path for {type(m).__name__}")
        self.__dict__["_np_cache"] = (key, layers)
        return layers

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_np_cache", None)  # views of this process's tensors; rebuilt on use
        return state
