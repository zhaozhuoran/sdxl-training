import sys
import os
import torch
from pathlib import Path
from trainer.config import load_config
from trainer.methods.lora.exporter import export_kohya_safetensors, convert_trainer_to_kohya_format
from safetensors import safe_open

# Append scripts folder to sys.path to load migrate_metadata
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from migrate_metadata import migrate_file

def test_config_parsing_with_model_name(tmp_path):
    yaml_content = """
model:
  model_name: "custom_beautiful_model"
  pretrained_model_name_or_path: "mock-model"
dataset:
  path: "mock-dataset"
training:
  steps: 500
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = load_config(config_file)
    assert config.model.model_name == "custom_beautiful_model"


def test_export_kohya_safetensors_with_model_name(tmp_path):
    # Setup some mock weights
    lora_state_dict = {
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight": torch.randn(16, 128),
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight": torch.randn(128, 16)
    }

    output_filepath = tmp_path / "model.safetensors"

    # Export with model_name
    export_kohya_safetensors(
        lora_state_dict=lora_state_dict,
        alpha=16.0,
        output_filepath=str(output_filepath),
        model_name="test_model_output_name"
    )

    # Read back and verify using safe_open
    with safe_open(str(output_filepath), framework="pt") as f:
        metadata = f.metadata()
        assert metadata is not None
        assert metadata.get("ss_base_model_version") == "sdxl_base_v1-0"
        assert metadata.get("ss_output_name") == "test_model_output_name"
        assert metadata.get("ss_network_dim") == "16"
        assert metadata.get("ss_network_alpha") == "16.0"


def test_migrate_metadata_script(tmp_path):
    # Setup standard safetensors with older base version or no output name
    lora_state_dict = {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": torch.randn(16, 128),
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": torch.randn(128, 16)
    }

    # Save standard/old style file
    old_metadata = {
        "ss_base_model_version": "sdxl"
    }
    from safetensors.torch import save_file
    old_filepath = tmp_path / "old_model.safetensors"
    save_file(lora_state_dict, str(old_filepath), metadata=old_metadata)

    # Run migration on the file (no overwrite option)
    migrated_filepath = tmp_path / "old_model_migrated.safetensors"
    success = migrate_file(
        file_path=old_filepath,
        model_name="migrated_output_name",
        overwrite=False
    )

    assert success is True
    assert migrated_filepath.exists()

    # Verify migrated file metadata
    with safe_open(str(migrated_filepath), framework="pt") as f:
        metadata = f.metadata()
        assert metadata is not None
        assert metadata.get("ss_base_model_version") == "sdxl_base_v1-0"
        assert metadata.get("ss_output_name") == "migrated_output_name"

    # Run migration with overwrite
    success_overwrite = migrate_file(
        file_path=old_filepath,
        model_name="overwrite_name",
        overwrite=True
    )
    assert success_overwrite is True

    # Verify old file got overwritten and has the new metadata
    with safe_open(str(old_filepath), framework="pt") as f:
        metadata = f.metadata()
        assert metadata is not None
        assert metadata.get("ss_base_model_version") == "sdxl_base_v1-0"
        assert metadata.get("ss_output_name") == "overwrite_name"
