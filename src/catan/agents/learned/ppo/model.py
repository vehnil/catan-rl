"""Actor-critic MLP over the flat feature vector."""

from __future__ import annotations

import torch
from torch import nn

from catan.agents.learned.common.distributions import masked_categorical
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
        logits, value = self(obs)
        dist = masked_categorical(logits, mask)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = masked_categorical(logits, mask)
        return dist.log_prob(actions), dist.entropy(), value
