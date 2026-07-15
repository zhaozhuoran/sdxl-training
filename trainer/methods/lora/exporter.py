import torch
from typing import Dict, Any, Optional
from safetensors.torch import save_file


def convert_trainer_to_kohya_format(lora_state_dict: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    """
    Converts our internal LoRA model paths and state dict keys to the standard,
    ecosystem-compatible Kohya-ss format keys. This allows direct loading into
    AUTOMATIC1111, ComfyUI, etc.

    Internal keys format:
      - unet.model.name...to_q.lora_down.weight
      - te1.text_model.encoder...q_proj.lora_down.weight
      - te2.text_model.encoder...q_proj.lora_down.weight

    Kohya format keys format:
      - lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
      - lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight
      - lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight

    Additionally, Kohya expects an alpha parameter stored for each key, e.g.:
      - lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha
    """
    converted_dict = {}

    for full_key, tensor in lora_state_dict.items():
        # full_key is like: "unet.down_blocks.0.attentions.0...to_q.lora_down.weight"
        # We need to translate this key into Kohya format
        # Standardize the target name.

        # 1. Determine model prefix
        prefix = ""
        remaining = ""
        if full_key.startswith("unet."):
            prefix = "lora_unet_"
            remaining = full_key[len("unet."):]
        elif full_key.startswith("te1."):
            prefix = "lora_te1_"
            remaining = full_key[len("te1."):]
        elif full_key.startswith("te2."):
            prefix = "lora_te2_"
            remaining = full_key[len("te2."):]
        else:
            # Fallback or custom module mapping
            prefix = "lora_custom_"
            remaining = full_key

        # 2. Extract direction (down vs up, or weight type)
        # remaining is like: "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight"
        suffix = ""
        if ".lora_down.weight" in remaining:
            suffix = ".lora_down.weight"
            remaining = remaining.replace(".lora_down.weight", "")
        elif ".lora_up.weight" in remaining:
            suffix = ".lora_up.weight"
            remaining = remaining.replace(".lora_up.weight", "")
        else:
            # Unexpected suffix
            continue

        # 3. Clean and convert standard diffusers/transformers module path parts to Kohya-style underscores
        # For UNet, replace dots with underscores
        # For Text Encoders, we want "text_model_encoder_layers_..."
        kohya_layer_name = remaining.replace(".", "_")

        # Construct final Kohya format keys
        kohya_key = f"{prefix}{kohya_layer_name}{suffix}"
        converted_dict[kohya_key] = tensor

        # Always inject the corresponding constant alpha value for this key
        alpha_key = f"{prefix}{kohya_layer_name}.alpha"
        converted_dict[alpha_key] = torch.tensor(alpha, dtype=torch.float32)

    return converted_dict


def export_kohya_safetensors(
    lora_state_dict: Dict[str, torch.Tensor],
    alpha: float,
    output_filepath: str,
    metadata: Optional[Dict[str, str]] = None
) -> None:
    """
    Exports standard, ecosystem-compatible Kohya .safetensors LoRA checkpoint to disk.
    Include optional metadata info.
    """
    converted_dict = convert_trainer_to_kohya_format(lora_state_dict, alpha)

    # Standardize precision to float32 or bfloat16/float16 for ecosystem compatibility
    output_dict = {}
    for k, v in converted_dict.items():
        output_dict[k] = v.contiguous()

    # Save to disk using safetensors library
    save_file(output_dict, output_filepath, metadata=metadata)
