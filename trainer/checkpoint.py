import os
import time
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from trainer.config import TrainingPipelineConfig, load_config
from trainer.methods.lora.injection import LoRAInjectionManager


class CheckpointManager:
    """
    Manages robust training checkpoint saving, validation, retention policies,
    and automatic or manual recovery processes.

    Structure of outputs folder:
    outputs/
      <experiment_name>/
        checkpoints/
          latest/
            trainer_state.pt
            training_state.json
          recovery/
            recovery-<step>.pt
          snapshots/
            snapshot-<step>.pt
          lora/
            step-<step>.safetensors
    """
    FORMAT_VERSION = "1.0.0"

    def __init__(
        self,
        config: TrainingPipelineConfig,
        output_dir: str,
        git_commit: Optional[str] = None,
        config_hash: Optional[str] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.git_commit = git_commit or "unknown"
        self.config_hash = config_hash or "unknown"

        # Setup structured folder paths
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.latest_dir = self.checkpoints_dir / "latest"
        self.recovery_dir = self.checkpoints_dir / "recovery"
        self.snapshots_dir = self.checkpoints_dir / "snapshots"
        self.lora_dir = self.checkpoints_dir / "lora"

        self._create_directories()

    def _create_directories(self) -> None:
        """Create all sub-directories needed for organized checkpoints."""
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)

    def build_metadata(self, step: int, epoch: int) -> Dict[str, str]:
        """Construct descriptive, searchable metadata for checkpoints."""
        return {
            "checkpoint_version": self.FORMAT_VERSION,
            "framework": "sdxl-training",
            "framework_version": "0.1.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "training_step": str(step),
            "training_epoch": str(epoch),
            "model_name": self.config.model.pretrained_model_name_or_path,
            "training_method": self.config.network.type
        }

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        lora_manager: LoRAInjectionManager,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        grad_scaler: Optional[Any] = None,
        is_snapshot: bool = False,
        rng_states: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves full trainer state (internal) and exported lora weights (ecosystem compatible).
        Also manages rolling recovery checkpoint limits.
        """
        metadata = self.build_metadata(step, epoch)

        # Prepare the state object for serialization
        state = {
            "metadata": metadata,
            "step": step,
            "epoch": epoch,
            "lora_state_dict": lora_manager.get_lora_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler is not None else None,
            "rng_states": rng_states or self._get_current_rng_states()
        }

        # Save to recovery or snapshot directory
        if is_snapshot:
            checkpoint_path = self.snapshots_dir / f"snapshot-{step:06d}.pt"
        else:
            checkpoint_path = self.recovery_dir / f"recovery-{step:06d}.pt"

        # Atomically write to avoid partial writes (corruptions)
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temp_path)
        os.replace(temp_path, checkpoint_path)

        # Update the 'latest' trainer symlinks/files as a lightweight index pointer
        latest_pt_path = self.latest_dir / "trainer_state.pt"
        # We can copy or reference. For robustness, let's copy the file.
        shutil.copy2(checkpoint_path, latest_pt_path)

        # Write latest metadata index
        training_state = {
            "latest_step": step,
            "latest_epoch": epoch,
            "latest_checkpoint_file": str(checkpoint_path.relative_to(self.output_dir)),
            "metadata": metadata
        }
        latest_json_path = self.latest_dir / "training_state.json"
        with open(latest_json_path, "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=4)

        # Enforce rolling history for recovery checkpoints
        if not is_snapshot:
            self._cleanup_old_recovery_checkpoints()

    def _get_current_rng_states(self) -> Dict[str, Any]:
        """Captures the standard random number generators states."""
        states = {
            "python": None, # For safety/simplicity
            "numpy": None,
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
        return states

    def _cleanup_old_recovery_checkpoints(self) -> None:
        """Removes older recovery checkpoints to stay within configured limits."""
        keep_last = self.config.checkpoint.keep_last_recovery
        if keep_last <= 0:
            return

        # Find and sort all recovery checkpoints
        files = list(self.recovery_dir.glob("recovery-*.pt"))
        # Sort ascending by step number extracted from filename
        files_with_steps = []
        for f in files:
            try:
                step_num = int(f.stem.split("-")[-1])
                files_with_steps.append((step_num, f))
            except ValueError:
                pass

        files_with_steps.sort(key=lambda x: x[0])

        # Delete if we exceed the limit
        if len(files_with_steps) > keep_last:
            to_delete = files_with_steps[:-keep_last]
            for _, f in to_delete:
                try:
                    f.unlink()
                except OSError:
                    pass

    def validate_checkpoint(self, path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validates a checkpoint file by inspecting:
          - successful deserialization
          - metadata structure
          - format version compatibility
          - existence of training step info and key state dicts
        """
        if not path.exists():
            return False, "File does not exist"

        try:
            # Load with weights on CPU to avoid allocating GPU memory during check
            state = torch.load(path, map_location="cpu")
        except Exception as e:
            return False, f"Deserialization failed: {e}"

        if not isinstance(state, dict):
            return False, "Checkpoint state is not a dictionary"

        # 1. Check metadata
        metadata = state.get("metadata")
        if not metadata:
            return False, "Missing metadata"

        # 2. Check format version
        version = metadata.get("checkpoint_version")
        if not version:
            return False, "Missing checkpoint format version"

        # Major version comparison to ensure long term design consideration compatibility
        try:
            chk_major = int(version.split(".")[0])
            self_major = int(self.FORMAT_VERSION.split(".")[0])
            if chk_major > self_major:
                return False, f"Incompatible format version: {version}. Expected <= {self_major}.x.x"
        except Exception:
            return False, f"Invalid format version string: {version}"

        # 3. Check training steps and epochs
        if "step" not in state or "epoch" not in state:
            return False, "Missing step or epoch info"

        # 4. Check optimizer/scheduler states
        if "lora_state_dict" not in state:
            return False, "Missing lora_state_dict"
        if "optimizer_state_dict" not in state:
            return False, "Missing optimizer_state_dict"

        return True, None

    def get_auto_resume_checkpoint(self) -> Optional[Path]:
        """
        Returns the newest valid checkpoint.
        Sequence:
          1. Try the recovery checkpoints (from newest to oldest).
          2. Fall back to snapshot checkpoints if recovery checking failed.
          3. If nothing is found or valid, return None.
        """
        # 1. Scan recovery checkpoints
        recovery_files = list(self.recovery_dir.glob("recovery-*.pt"))
        recovery_with_steps = []
        for f in recovery_files:
            try:
                step_num = int(f.stem.split("-")[-1])
                recovery_with_steps.append((step_num, f))
            except ValueError:
                pass
        # Sort descending to try newest first
        recovery_with_steps.sort(key=lambda x: x[0], reverse=True)

        for _, f in recovery_with_steps:
            valid, err = self.validate_checkpoint(f)
            if valid:
                return f
            # Log warning here in real engine

        # 2. Scan snapshot checkpoints
        snapshot_files = list(self.snapshots_dir.glob("snapshot-*.pt"))
        snapshot_with_steps = []
        for f in snapshot_files:
            try:
                step_num = int(f.stem.split("-")[-1])
                snapshot_with_steps.append((step_num, f))
            except ValueError:
                pass
        # Sort descending to try newest first
        snapshot_with_steps.sort(key=lambda x: x[0], reverse=True)

        for _, f in snapshot_with_steps:
            valid, err = self.validate_checkpoint(f)
            if valid:
                return f

        return None
