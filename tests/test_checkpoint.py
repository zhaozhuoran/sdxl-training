import torch
import shutil
import pytest
from trainer.config import TrainingPipelineConfig, load_config
from trainer.checkpoint import CheckpointManager
from trainer.methods.lora.injection import LoRAInjectionManager

@pytest.fixture
def test_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_content = """
model:
  pretrained_model_name_or_path: "mock-model"
dataset:
  path: "mock-dataset"
training:
  steps: 1000
checkpoint:
  keep_last_recovery: 2
"""
    cfg_path.write_text(cfg_content)
    return load_config(cfg_path)

def test_checkpoint_saving_and_rolling_cleanup(tmp_path, test_config):
    # Initialize components
    manager = CheckpointManager(test_config, output_dir=str(tmp_path))

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

    model = MockModel()
    lora_manager = LoRAInjectionManager(unet_targets=["linear"])
    lora_manager.inject_lora(model, prefix="unet", targets=lora_manager.unet_targets)

    optimizer = torch.optim.AdamW(lora_manager.get_lora_parameters(), lr=1e-4)

    class MockScheduler:
        def state_dict(self):
            return {"last_epoch": 1}
    scheduler = MockScheduler()

    # 1. Save step 1 recovery checkpoint
    manager.save_checkpoint(
        step=1,
        epoch=0,
        lora_manager=lora_manager,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # 2. Save step 2 recovery checkpoint
    manager.save_checkpoint(
        step=2,
        epoch=0,
        lora_manager=lora_manager,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Check both recovery checkpoints exist
    assert (manager.recovery_dir / "recovery-000001.pt").exists()
    assert (manager.recovery_dir / "recovery-000002.pt").exists()

    # 3. Save step 3 recovery checkpoint (keep_last_recovery is 2, so 000001 should be deleted)
    manager.save_checkpoint(
        step=3,
        epoch=0,
        lora_manager=lora_manager,
        optimizer=optimizer,
        scheduler=scheduler
    )

    assert not (manager.recovery_dir / "recovery-000001.pt").exists()
    assert (manager.recovery_dir / "recovery-000002.pt").exists()
    assert (manager.recovery_dir / "recovery-000003.pt").exists()


def test_checkpoint_validation_and_fallback(tmp_path, test_config):
    manager = CheckpointManager(test_config, output_dir=str(tmp_path))

    # Empty or missing file
    valid, err = manager.validate_checkpoint(tmp_path / "nonexistent.pt")
    assert not valid
    assert "exist" in err

    # Non-dictionary structure (corrupted/invalid file)
    bad_file = tmp_path / "bad.pt"
    torch.save([1, 2, 3], bad_file)
    valid, err = manager.validate_checkpoint(bad_file)
    assert not valid
    assert "dictionary" in err

    # Missing metadata
    no_meta_file = tmp_path / "no_meta.pt"
    torch.save({"step": 1, "epoch": 0}, no_meta_file)
    valid, err = manager.validate_checkpoint(no_meta_file)
    assert not valid
    assert "metadata" in err

    # Incompatible higher major version check
    higher_ver_file = tmp_path / "high_ver.pt"
    state = {
        "metadata": {
            "checkpoint_version": "99.0.0",
            "framework": "sdxl-training"
        },
        "step": 1,
        "epoch": 0,
        "lora_state_dict": {},
        "optimizer_state_dict": {}
    }
    torch.save(state, higher_ver_file)
    valid, err = manager.validate_checkpoint(higher_ver_file)
    assert not valid
    assert "Incompatible format version" in err


def test_auto_resume_selection(tmp_path, test_config):
    manager = CheckpointManager(test_config, output_dir=str(tmp_path))

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

    model = MockModel()
    lora_manager = LoRAInjectionManager(unet_targets=["linear"])
    lora_manager.inject_lora(model, prefix="unet", targets=lora_manager.unet_targets)

    optimizer = torch.optim.AdamW(lora_manager.get_lora_parameters(), lr=1e-4)

    class MockScheduler:
        def state_dict(self):
            return {}
    scheduler = MockScheduler()

    # Save snapshot first (step 5)
    manager.save_checkpoint(
        step=5,
        epoch=0,
        lora_manager=lora_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        is_snapshot=True
    )

    # Save recovery (step 10)
    manager.save_checkpoint(
        step=10,
        epoch=0,
        lora_manager=lora_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        is_snapshot=False
    )

    # Recovery is written synchronously, but the snapshot is queued to the
    # background worker; flush so the auto-resume scan sees a deterministic state.
    manager.flush()

    # The newest valid checkpoint should be the recovery at step 10
    best_chk = manager.get_auto_resume_checkpoint()
    assert best_chk is not None
    assert "recovery-000010.pt" in best_chk.name

    # Now simulate corruption of step 10 recovery checkpoint by overwriting it
    torch.save([1, 2], manager.recovery_dir / "recovery-000010.pt")

    # It should automatically fall back to the snapshot at step 5
    best_chk_after_corruption = manager.get_auto_resume_checkpoint()
    assert best_chk_after_corruption is not None
    assert "snapshot-000005.pt" in best_chk_after_corruption.name
