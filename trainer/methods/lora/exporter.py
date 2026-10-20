from typing import Dict, Optional

import torch
from safetensors.torch import save_file


def convert_trainer_to_kohya_format(
    lora_state_dict: Dict[str, torch.Tensor], alpha: float
) -> Dict[str, torch.Tensor]:
    """Map model paths to Kohya keys and add each layer's alpha tensor.

    For example, unet.attn.to_q.lora_down.weight becomes
    lora_unet_attn_to_q.lora_down.weight.
    """
    converted = {}
    prefixes = {"unet": "lora_unet_", "te1": "lora_te1_", "te2": "lora_te2_"}

    for key, tensor in lora_state_dict.items():
        component, _, path = key.partition(".")
        prefix = prefixes.get(component, "lora_custom_")
        if component not in prefixes:
            path = key

        for suffix in (".lora_down.weight", ".lora_up.weight"):
            if path.endswith(suffix):
                layer_name = prefix + path[:-len(suffix)].replace(".", "_")
                converted[layer_name + suffix] = tensor
                converted[layer_name + ".alpha"] = torch.tensor(alpha, dtype=torch.float32)
                break

    return converted


def export_kohya_safetensors(
    lora_state_dict: Dict[str, torch.Tensor],
    alpha: float,
    output_filepath: str,
    metadata: Optional[Dict[str, str]] = None,
    model_name: Optional[str] = None,
) -> None:
    """Write adapter tensors with SDXL LoRA metadata. Caller metadata takes precedence."""
    tensors = convert_trainer_to_kohya_format(lora_state_dict, alpha)
    tensors = {key: value.contiguous() for key, value in tensors.items()}
    rank = next(
        (value.shape[0] for key, value in lora_state_dict.items()
         if key.endswith(".lora_down.weight")),
        None,
    )
    export_metadata = {
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_network_module": "networks.lora",
        "ss_network_type": "lora",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
    }
    if model_name:
        export_metadata["ss_output_name"] = model_name
    if metadata:
        export_metadata.update(metadata)

    save_file(tensors, output_filepath, metadata=export_metadata)
