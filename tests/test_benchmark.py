import pytest
from PIL import Image

from trainer.engine.trainer import SDXLTrainer, AutoOOMError
from trainer.benchmark import BenchmarkSuite


@pytest.fixture
def mock_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    img = Image.new("RGB", (64, 64), color="red")
    img.save(dataset_dir / "00001.png")
    (dataset_dir / "00001.txt").write_text("a placeholder caption", encoding="utf-8")
    return dataset_dir


@pytest.fixture
def base_config_yaml(mock_dataset, tmp_path):
    config_yaml = f"""
model:
  pretrained_model_name_or_path: "mock-model"
dataset:
  path: "{mock_dataset}"
  batch_size: 1
  resolution: 64
training:
  steps: 5
  mixed_precision: "no"
  min_free_vram_gb: 0.5
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
  experiment_name: "bench_base"
"""
    config_file = tmp_path / "base_config.yaml"
    config_file.write_text(config_yaml)
    return str(config_file)


def test_min_free_vram_gb_validation(base_config_yaml):
    from trainer.config import load_config
    import yaml
    from pydantic import ValidationError

    with open(base_config_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Test negative value
    data["training"]["min_free_vram_gb"] = -0.1
    with pytest.raises(ValidationError) as exc_info:
        from trainer.config import TrainingPipelineConfig
        TrainingPipelineConfig(**data)
    assert "min_free_vram_gb must be a non-negative finite float" in str(exc_info.value)

    # Test infinite value
    data["training"]["min_free_vram_gb"] = float("inf")
    with pytest.raises(ValidationError) as exc_info:
        TrainingPipelineConfig(**data)
    assert "min_free_vram_gb must be a non-negative finite float" in str(exc_info.value)


def test_vram_auto_oom_mock_check(base_config_yaml):
    # This verifies that AutoOOMError is raised when the mock free vram drops below threshold
    trainer = SDXLTrainer(base_config_yaml, is_test_mode=True)
    # 0.4 GB is less than the threshold 0.5 GB, should trigger OOM
    trainer._mock_free_vram_bytes = int(0.4 * (1024 ** 3))

    with pytest.raises(AutoOOMError) as exc_info:
        trainer._check_vram_limit("test_phase")

    assert "Auto OOM detected in phase: test_phase" in str(exc_info.value)
    assert "Free VRAM: 0.400 GB is below the threshold of 0.500 GB" in str(exc_info.value)

    # 0.6 GB should pass
    trainer._mock_free_vram_bytes = int(0.6 * (1024 ** 3))
    trainer._check_vram_limit("test_phase")


def test_benchmark_suite_execution(base_config_yaml, tmp_path):
    # Setup multiple trial configs
    trials = [
        {
            "name": "trial_1_pass",
            "overrides": {
                "training": {
                    "steps": 2
                }
            }
        },
        {
            "name": "trial_2_oom",
            "overrides": {
                "training": {
                    "steps": 2,
                    "min_free_vram_gb": 1.0
                }
            },
            # This triggers AutoOOMError in BenchmarkSuite
            "mock_free_vram_bytes": int(0.5 * (1024 ** 3))
        }
    ]

    state_file = tmp_path / "bench_state.json"

    suite = BenchmarkSuite(
        base_config_path=base_config_yaml,
        trials_configs=trials,
        state_file_path=str(state_file),
        is_test_mode=True
    )

    results = suite.run()

    assert len(results) == 2
    assert results[0]["name"] == "trial_1_pass"
    assert results[0]["status"] == "completed"

    assert results[1]["name"] == "trial_2_oom"
    assert results[1]["status"] == "canceled (Auto OOM detected)"
    assert "Auto OOM detected" in results[1]["error"]


def test_benchmark_suite_pause_and_resume(base_config_yaml, tmp_path):
    trials = [
        {
            "name": "trial_first",
            "overrides": {
                "training": {"steps": 1}
            }
        },
        {
            "name": "trial_second",
            "overrides": {
                "training": {"steps": 1}
            }
        }
    ]

    state_file = tmp_path / "pause_resume_state.json"

    # Instantiate first suite
    suite = BenchmarkSuite(
        base_config_path=base_config_yaml,
        trials_configs=trials,
        state_file_path=str(state_file),
        is_test_mode=True
    )

    # Pause after first trial by overriding the loop behavior or setting paused to True inside
    # Let's run with paused = True to prevent any runs initially
    suite.paused = True
    results = suite.run()
    assert len(results) == 0
    assert suite.current_trial_idx == 0

    # Unpause and run only 1 trial then pause manually
    suite.paused = False
    # To simulate pausing after 1 step, we can run a single trial and set paused=True in between
    # or just let it run. Let's do it by manipulating current_trial_idx or mocking.
    # Alternatively we run the first trial, then we save state and reload.
    # Let's run a custom run method or simply run the first trial and manually save the state:
    first_res = suite._run_single_trial(trials[0])
    suite.results.append(first_res)
    suite.current_trial_idx = 1
    suite.save_state()

    assert state_file.exists()

    # Create a second BenchmarkSuite loading the same state_file
    suite2 = BenchmarkSuite(
        base_config_path=base_config_yaml,
        trials_configs=trials,
        state_file_path=str(state_file),
        is_test_mode=True
    )

    assert suite2.current_trial_idx == 1
    assert len(suite2.results) == 1
    assert suite2.results[0]["name"] == "trial_first"

    # Resume/run remaining trials
    final_results = suite2.run()
    assert len(final_results) == 2
    assert final_results[1]["name"] == "trial_second"
    assert final_results[1]["status"] == "completed"
