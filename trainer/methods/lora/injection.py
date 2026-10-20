import math
from typing import Optional, List, Dict
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Add a rank-limited update to a linear layer, scaled by alpha / rank."""
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # A zero up-projection leaves the base layer's output unchanged initially.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.lora_up(self.lora_down(x)) * self.scaling


class LoRAInjectionManager:
    """Replace matching linear layers and track their adapter weights."""
    def __init__(
        self,
        rank: int = 64,
        alpha: float = 32.0,
        unet_targets: Optional[List[str]] = None,
        text_encoder_1_targets: Optional[List[str]] = None,
        text_encoder_2_targets: Optional[List[str]] = None,
    ):
        self.rank = rank
        self.alpha = alpha

        self.unet_targets = unet_targets if unet_targets is not None else ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
        self.te1_targets = text_encoder_1_targets if text_encoder_1_targets is not None else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        self.te2_targets = text_encoder_2_targets if text_encoder_2_targets is not None else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

        self.injected_modules: Dict[str, LoRALinear] = {}

    def _should_inject(self, name: str, targets: List[str]) -> bool:
        """Targets are substrings of module paths, not regular expressions."""
        return any(target in name for target in targets)

    def inject_lora(self, model: nn.Module, prefix: str, targets: List[str]) -> None:
        """Wrap matching layers after traversal, so replacements aren't visited again."""
        replacements = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and self._should_inject(name, targets)
        ]
        for name, layer in replacements:
            parent_path, _, attribute = name.rpartition(".")
            parent = model.get_submodule(parent_path)
            wrapper = LoRALinear(layer, rank=self.rank, alpha=self.alpha)
            setattr(parent, attribute, wrapper)
            full_name = f"{prefix}.{name}" if prefix else name
            self.injected_modules[full_name] = wrapper

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Return adapter parameters only; base weights stay out of the optimizer."""
        return [
            weight
            for wrapper in self.injected_modules.values()
            for weight in (wrapper.lora_down.weight, wrapper.lora_up.weight)
        ]

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return detached adapter weights, keyed by model and module path."""
        state_dict = {}
        for name, wrapper in self.injected_modules.items():
            state_dict[f"{name}.lora_down.weight"] = wrapper.lora_down.weight.detach()
            state_dict[f"{name}.lora_up.weight"] = wrapper.lora_up.weight.detach()
        return state_dict

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Loads a given LoRA state dict back into the injected modules."""
        with torch.no_grad():
            for name, wrapper in self.injected_modules.items():
                down_key = f"{name}.lora_down.weight"
                up_key = f"{name}.lora_up.weight"
                if down_key in state_dict:
                    wrapper.lora_down.weight.copy_(state_dict[down_key])
                if up_key in state_dict:
                    wrapper.lora_up.weight.copy_(state_dict[up_key])
