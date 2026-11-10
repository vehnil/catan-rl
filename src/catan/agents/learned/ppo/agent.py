"""PPO policy wrapped as a playable ``Agent`` (for tournaments, evaluation and self-play)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from catan.agents.learned.common.distributions import masked_logits
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

    @torch.no_grad()
    def act(self, obs: Observation) -> int:
        x = torch.as_tensor(encode(obs, self.belief)).unsqueeze(0)
        logits, _ = self.model(x)
        logits = masked_logits(logits, torch.as_tensor(obs.legal_mask).unsqueeze(0))
        if self.deterministic:
            return int(logits.argmax())
        g = torch.Generator().manual_seed(int(self.rng.integers(2**31)))
        return int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))

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
