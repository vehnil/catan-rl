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
        """Policy logits for one feature vector, in NumPy (no value head, no autograd).

        For frozen opponents: at batch size 1 this is about twice as fast as torch. Reads
        the parameters in place (shared memory), so it follows in-place weight updates.
        Matches the torch forward up to float32 rounding (~1e-6)."""
        x = obs
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                x = layer.weight.detach().numpy() @ x + layer.bias.detach().numpy()
            elif isinstance(layer, nn.Tanh):
                x = np.tanh(x)
            else:
                raise TypeError(f"no NumPy path for {type(layer).__name__}")
        return self.policy.weight.detach().numpy() @ x + self.policy.bias.detach().numpy()

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = masked_categorical(logits, mask)
        return dist.log_prob(actions), dist.entropy(), value
