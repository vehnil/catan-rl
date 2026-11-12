"""PPO policy wrapped as a playable ``Agent`` (for tournaments, evaluation and self-play)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from catan.agents.learned.ppo.model import ActorCritic
from catan.player.base import Agent
from catan.player.features import FEATURE_DIM, encode
from catan.player.observation import Observation


class PPOAgent(Agent):
    name = "ppo"

    def __init__(
        self,
        hidden: list[int] | None = None,
        deterministic: bool = False,
        model: ActorCritic | None = None,
    ) -> None:
        super().__init__()
        self.hidden = list(hidden or [512, 256])
        self.deterministic = deterministic
        self.model = model or ActorCritic(self.hidden)
        self.model.eval()

    def config(self) -> dict[str, Any]:
        return {"hidden": self.hidden, "deterministic": self.deterministic}

    def act(self, obs: Observation) -> int:
        # NumPy forward and sampling: at batch size 1 torch's per-call overhead dominates.
        # Same distribution (softmax over the legal actions' logits) as the torch path,
        # but a different random stream, so games differ from runs before this change.
        return self.choose(obs, self.model.policy_logits_numpy(self.features(obs)))

    # act() in two halves, so a driver can batch the forward pass across games:
    # features() for each pending agent, one policy_logits_numpy() on the stack, choose().

    def features(self, obs: Observation) -> np.ndarray:
        return encode(obs, self.belief)

    def choose(self, obs: Observation, logits: np.ndarray) -> int:
        legal = np.flatnonzero(obs.legal_mask)
        z = logits[legal]
        if self.deterministic:
            return int(legal[int(np.argmax(z))])
        cdf = np.cumsum(np.exp(z - z.max()))
        i = int(np.searchsorted(cdf, self.rng.random() * cdf[-1], side="right"))
        return int(legal[min(i, len(legal) - 1)])

    # ------------------------------------------------------------ persistence

    def save_files(self, directory: Path) -> dict[str, str]:
        from safetensors.torch import save_file

        save_file(
            {k: v.contiguous() for k, v in self.model.state_dict().items()},
            str(directory / "weights.safetensors"),
        )
        return {"weights": "weights.safetensors"}

    def load_files(self, directory: Path, manifest: dict[str, Any]) -> None:
        from safetensors.torch import load_file

        self.model.load_state_dict(load_file(str(directory / "weights.safetensors")))
        self.model.eval()

    def export_onnx(self, dest: Path) -> None:
        try:
            torch.onnx.export(
                self.model,
                (torch.zeros(1, FEATURE_DIM),),
                str(dest),
                input_names=["features"],
                output_names=["logits", "value"],
                dynamic_axes={"features": {0: "batch"}},
            )
        except ImportError as e:  # newer torch exporters need onnx / onnxscript
            raise RuntimeError("ONNX export needs extra packages: `uv add onnx onnxscript`") from e
