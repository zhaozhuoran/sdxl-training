import os
import pytest
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
