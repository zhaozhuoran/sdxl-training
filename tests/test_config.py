from trainer.config import load_config, TrainingPipelineConfig, NetworkConfig

def test_config_parsing(tmp_path):
    yaml_content = """
model:
  pretrained_model_name_or_path: "mock-model"
dataset:
  path: "mock-dataset"
  batch_size: 2
training:
  steps: 500
  mixed_precision: "fp16"
network:
  type: lora
  rank: 32
optimizer:
  type: adamw
scheduler:
  type: constant
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = load_config(config_file)
    assert isinstance(config, TrainingPipelineConfig)
    assert config.model.pretrained_model_name_or_path == "mock-model"
    assert config.dataset.path == "mock-dataset"
    assert config.dataset.batch_size == 2
    assert config.training.steps == 500
    assert config.training.mixed_precision == "fp16"
    assert config.network.rank == 32
    assert config.optimizer.type == "adamw"
    assert config.scheduler.type == "constant"
