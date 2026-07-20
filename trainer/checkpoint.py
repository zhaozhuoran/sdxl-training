import os
import time
import json
import threading
import queue
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from trainer.config import TrainingPipelineConfig, load_config
from trainer.logging import get_logger
from trainer.methods.lora.injection import LoRAInjectionManager

logger = get_logger()


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

        # Background worker that serializes checkpoints off the training thread
        # so that disk I/O (and the duplicate "latest" copy) never blocks a step.
        self._save_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._save_worker, name="checkpoint-writer", daemon=True
        )
        self._worker_thread.start()

    @staticmethod
    def _to_cpu(obj: Any) -> Any:
        """
        Recursively detach, move to CPU and clone every tensor so the snapshot
        is fully decoupled from GPU memory (avoids GPU OOM) and from subsequent
        in-place training updates. Non-tensor leaves are returned untouched.
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().clone()
        if isinstance(obj, dict):
            return {k: CheckpointManager._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(CheckpointManager._to_cpu(v) for v in obj)
        return obj

    def _enqueue_save(self, step: int, is_snapshot: bool, state: Dict[str, Any]) -> None:
        """Hand a fully CPU-resident snapshot to the background writer."""
        self._save_queue.put({
            "step": step,
            "is_snapshot": is_snapshot,
            "state": state,
        })

    def _save_worker(self) -> None:
        """Drain the save queue, serializing checkpoints off the main thread."""
        while True:
            job = self._save_queue.get()
            if job is None:
                self._save_queue.task_done()
                break
            try:
                self._write_checkpoint(job)
            except Exception as e:  # never let one failure kill the worker
                logger.error(f"[checkpoint] background save failed: {e}")
            finally:
                self._save_queue.task_done()

    def _write_checkpoint(self, job: Dict[str, Any]) -> None:
        step = job["step"]
        is_snapshot = job["is_snapshot"]
        state = job["state"]

        if is_snapshot:
            checkpoint_path = self.snapshots_dir / f"snapshot-{step:06d}.pt"
        else:
            checkpoint_path = self.recovery_dir / f"recovery-{step:06d}.pt"

        # Atomically write to avoid partial writes (corruptions)
        temp_path = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temp_path)
        os.replace(temp_path, checkpoint_path)

        # Update the 'latest' index. The full-file copy is now done in the
        # background so it no longer stalls the training step.
        latest_pt_path = self.latest_dir / "trainer_state.pt"
        shutil.copy2(checkpoint_path, latest_pt_path)

        # Write latest metadata index atomically
        training_state = {
            "latest_step": step,
            "latest_epoch": state.get("epoch", step),
            "latest_checkpoint_file": str(checkpoint_path.relative_to(self.output_dir)),
            "metadata": state.get("metadata", {}),
        }
        latest_json_path = self.latest_dir / "training_state.json"
        temp_json = latest_json_path.with_suffix(".tmp")
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=4)
        os.replace(temp_json, latest_json_path)

        # Enforce rolling history for recovery checkpoints
        if not is_snapshot:
            self._cleanup_old_recovery_checkpoints()

    def shutdown(self, timeout: float = 600.0) -> None:
        """
        Flush all pending checkpoint writes and stop the background worker.
        Safe to call multiple times.
        """
        if self._worker_thread is None:
            return
        try:
            self._save_queue.join()
        except Exception:
            pass
        self._save_queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout)
        self._worker_thread = None

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
        Collects a fully CPU-resident snapshot on the calling (training) thread,
        then hands it to a background worker for serialization so the training
        step is never blocked by disk I/O.

        Both rolling recovery and long-term snapshots store the FULL trainer
        state (LoRA weights + optimizer + scheduler + scaler + RNG). Rolling
        checkpoints are saved frequently and rotated (keep_last), so auto-resume
        recovers from the most recent point with exact optimizer momentum and LR
        schedule. Snapshots are the infrequent, permanently-retained archive.
        """
        metadata = self.build_metadata(step, epoch)

        # LoRA weights are still on the GPU; detach + clone to CPU immediately
        # so the background writer never touches GPU memory (avoids GPU OOM).
        lora_state_dict = self._to_cpu(lora_manager.get_lora_state_dict())

        state: Dict[str, Any] = {
            "metadata": metadata,
            "step": step,
            "epoch": epoch,
            "lora_state_dict": lora_state_dict,
            "optimizer_state_dict": self._to_cpu(optimizer.state_dict()),
            "rng_states": rng_states or self._get_current_rng_states(),
        }
        if hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = self._to_cpu(scheduler.state_dict())
        if grad_scaler is not None:
            state["grad_scaler_state_dict"] = self._to_cpu(grad_scaler.state_dict())

        job = {"step": step, "is_snapshot": is_snapshot, "state": state}

        if is_snapshot:
            # Snapshots are the infrequent, permanently-retained archive. Serialize
            # them on the background thread so a (rare) large snapshot write never
            # stalls a training step.
            self._enqueue_save(step, is_snapshot, state)
        else:
            # Rolling recovery checkpoints are the resume safety net and MUST be
            # durable the instant save_checkpoint returns: a crash right after
            # return has to find the newest recovery on disk. Write synchronously
            # on the caller thread.
            self._write_checkpoint(job)

    def flush(self, timeout: float = 600.0) -> None:
        """
        Block until every queued (snapshot) write has been serialized.

        Recovery checkpoints are written synchronously by save_checkpoint, so this
        only affects the background snapshot writer. Safe to call when no writes are
        pending. Used by tests and by shutdown() before process exit.
        """
        if self._worker_thread is None:
            return
        try:
            self._save_queue.join()
        except Exception:
            pass

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

        # 4. Check core state. Optimizer/scheduler are optional because
        #    rolling recovery checkpoints store only the minimal resume set.
        if "lora_state_dict" not in state:
            return False, "Missing lora_state_dict"
        if "optimizer_state_dict" in state and not isinstance(state["optimizer_state_dict"], dict):
            return False, "optimizer_state_dict is not a dictionary"

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
