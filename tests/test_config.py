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
    # Verify new default configuration values
    assert config.dataset.cache_latents is True
    assert config.dataset.cache_text_encoder_outputs is True
    assert config.dataset.cache_destination == "ram"
    assert config.dataset.cache_dir is None
    assert config.training.train_text_encoder is False


def test_custom_caching_config(tmp_path):
    yaml_content = """
model:
  pretrained_model_name_or_path: "mock-model"
dataset:
  path: "mock-dataset"
  batch_size: 1
  cache_latents: false
  cache_text_encoder_outputs: false
  cache_destination: "disk"
  cache_dir: "/tmp/custom_cache"
training:
  steps: 100
  train_text_encoder: true
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = load_config(config_file)
    assert config.dataset.cache_latents is False
    assert config.dataset.cache_text_encoder_outputs is False
    assert config.dataset.cache_destination == "disk"
    assert config.dataset.cache_dir == "/tmp/custom_cache"
    assert config.training.train_text_encoder is True
