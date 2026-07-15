import math
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    A lightweight, generic LoRA wrapper for a torch.nn.Linear layer.
    """
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

        # Initialize down and up projections
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Reset parameters following standard LoRA initialization
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        lora_output = self.lora_up(self.lora_down(x)) * self.scaling
        return original_output + lora_output


class LoRAInjectionManager:
    """
    Manages search, injection, tracking, and restoration of LoRA layers
    within target model components (UNet, Text Encoders).
    """
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

        # Defaults to target general linear modules or cross attention modules if not specified.
        # This matches the expected ecosystem-compatible target behaviors.
        self.unet_targets = unet_targets if unet_targets is not None else ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
        self.te1_targets = text_encoder_1_targets if text_encoder_1_targets is not None else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        self.te2_targets = text_encoder_2_targets if text_encoder_2_targets is not None else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

        # Track our injected wrappers
        self.injected_modules: Dict[str, LoRALinear] = {}

    def _should_inject(self, name: str, targets: List[str]) -> bool:
        """Determines if the module name matches any target submodule pattern."""
        return any(t in name for t in targets)

    def inject_lora(self, model: nn.Module, prefix: str, targets: List[str]) -> None:
        """
        Recursively walks the module tree, finding nn.Linear layers matching the targets,
        and wraps them in a LoRALinear wrapper.
        """
        # We need to construct a list of replacements first to avoid modifying state during traversal
        replacements = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Construct the absolute path prefix key
                full_name = f"{prefix}.{name}" if prefix else name
                if self._should_inject(name, targets):
                    replacements.append((name, module, full_name))

        for name, original_layer, full_name in replacements:
            # We must traverse to the parent module to set the attribute
            parts = name.split(".")
            sub_model = model
            for part in parts[:-1]:
                sub_model = getattr(sub_model, part)

            last_part = parts[-1]
            # Wrap the original layer
            lora_wrapper = LoRALinear(original_layer, rank=self.rank, alpha=self.alpha)
            setattr(sub_model, last_part, lora_wrapper)
            self.injected_modules[full_name] = lora_wrapper

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Returns a list of all parameter groups that we want to optimize (the LoRA weights)."""
        params = []
        for wrapper in self.injected_modules.values():
            params.extend(list(wrapper.lora_down.parameters()))
            params.extend(list(wrapper.lora_up.parameters()))
        return params

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns the state dict containing ONLY the LoRA parameters."""
        state_dict = {}
        for name, wrapper in self.injected_modules.items():
            # Standardize naming representation of down and up weight tensors
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
