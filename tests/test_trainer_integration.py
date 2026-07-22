import os
import pytest
import torch
from PIL import Image
from trainer.engine.trainer import SDXLTrainer


def test_gpu_first_failure(tmp_path):
    config_yaml = """
model:
  pretrained_model_name_or_path: "mock"
dataset:
  path: "mock"
training:
  steps: 10
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml)

    # This test verifies the GPU-first guard, which only fires when no CUDA GPU
    # is present. On a CUDA-enabled machine the guard is intentionally never
    # triggered, so skip rather than fail.
    if torch.cuda.is_available():
        pytest.skip("Requires a CUDA-less environment to verify the GPU-first guard.")

    # In regular production execution, if no GPU is available, it must raise RuntimeError immediately.
    with pytest.raises(RuntimeError) as exc_info:
        SDXLTrainer(str(config_file), is_test_mode=False)

    assert "CUDA-compatible GPU is unavailable" in str(exc_info.value)


def test_test_mode_cpu_run(tmp_path):
    # Setup standard layout folder
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Generate image and caption
    img = Image.new("RGB", (128, 128), color="green")
    img.save(dataset_dir / "00001.png")
    (dataset_dir / "00001.txt").write_text("a green placeholder image", encoding="utf-8")

    config_yaml = f"""
model:
  pretrained_model_name_or_path: "mock-model"
dataset:
  path: "{dataset_dir}"
  batch_size: 1
  resolution: 64
training:
  steps: 5
  mixed_precision: "no"
network:
  type: "lora"
  rank: 8
  alpha: 4.0
optimizer:
  type: "adamw"
  learning_rate: 1e-4
scheduler:
  type: "constant"
checkpoint:
  save_every_steps: 2
  keep_last_recovery: 2
output:
  directory: "{tmp_path}/outputs"
  experiment_name: "cpu_test_run"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml)

    # In test mode, it should run fully and successfully on CPU without errors
    trainer = SDXLTrainer(str(config_file), is_test_mode=True)
    trainer.run()

    # Check output folder structure was created and managed successfully
    exp_dir = tmp_path / "outputs" / "cpu_test_run"
    assert exp_dir.exists()
    assert (exp_dir / "config.yaml").exists()
    assert (exp_dir / "logs" / "train.log").exists()

    # Check recovery checkpoints limit (only latest 2 should exist)
    chk_dir = exp_dir / "checkpoints"
    recovery_files = list((chk_dir / "recovery").glob("*.pt"))
    assert len(recovery_files) == 2 # step 2 and step 4 should exist, step 2 is deleted or step 2/4 remain

    # Check exported LoRA file exist
    lora_files = list((chk_dir / "lora").glob("*.safetensors"))
    assert len(lora_files) == 1
    assert "step-000005.safetensors" in lora_files[0].name


def test_trainer_unet_only_and_caching(tmp_path):
    # Setup dataset
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Generate image and caption
    img = Image.new("RGB", (128, 128), color="blue")
    img.save(dataset_dir / "00001.png")
    (dataset_dir / "00001.txt").write_text("a blue placeholder image", encoding="utf-8")

    config_yaml = f"""
model:
  pretrained_model_name_or_path: "mock-model"
dataset:
  path: "{dataset_dir}"
  batch_size: 1
  resolution: 64
  cache_latents: true
  cache_text_encoder_outputs: true
  cache_destination: "ram"
training:
  steps: 3
  mixed_precision: "no"
  train_text_encoder: false
network:
  type: "lora"
  rank: 8
  alpha: 4.0
optimizer:
  type: "adamw"
  learning_rate: 1e-4
scheduler:
  type: "constant"
output:
  directory: "{tmp_path}/outputs"
  experiment_name: "unet_only_cache_run"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml)

    # Initialize trainer
    trainer = SDXLTrainer(str(config_file), is_test_mode=True)
    trainer.run()

    # Verify that only unet has LoRA injected (no te1 or te2 keys)
    injected_keys = list(trainer.lora_manager.injected_modules.keys())
    assert any(k.startswith("unet.") for k in injected_keys)
    assert not any(k.startswith("te1.") for k in injected_keys)
    assert not any(k.startswith("te2.") for k in injected_keys)

    # Verify that caching populated the dataset ram_cache
    dataset = trainer.dataloader.dataset
    assert len(dataset.ram_cache) == 1
    for k, v in dataset.ram_cache.items():
        assert "latents" in v
        assert "prompt_embeds" in v
        assert "pooled_prompt_embeds" in v
