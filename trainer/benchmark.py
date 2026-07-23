import os
import json
import time
from typing import List, Dict, Any

import yaml
import torch

from trainer.engine.trainer import SDXLTrainer, AutoOOMError
from trainer.logging import get_logger

logger = get_logger()


class BenchmarkSuite:
    """
    Manages running a series of benchmark trials, gathering telemetry,
    saving/resuming state, and gracefully catching AutoOOMError.
    """
    def __init__(
        self,
        base_config_path: str,
        trials_configs: List[Dict[str, Any]],
        state_file_path: str = "benchmark_state.json",
        is_test_mode: bool = False
    ):
        self.base_config_path = base_config_path
        self.trials_configs = trials_configs
        self.state_file_path = state_file_path
        self.is_test_mode = is_test_mode

        self.results = []
        self.current_trial_idx = 0
        self.paused = False

        self.load_state()

    def load_state(self) -> None:
        """Loads state from file if it exists, to support pause and resume."""
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, "r") as f:
                    state = json.load(f)
                self.results = state.get("results", [])
                self.current_trial_idx = state.get("current_trial_idx", 0)
                logger.info(
                    f"Resumed benchmark suite state. Next trial index: {self.current_trial_idx} "
                    f"({len(self.results)} already completed/run)."
                )
            except Exception as e:
                logger.warning(f"Failed to load benchmark state: {e}. Starting fresh.")

    def save_state(self) -> None:
        """Saves current state to file."""
        try:
            state = {
                "results": self.results,
                "current_trial_idx": self.current_trial_idx,
                "timestamp": time.time()
            }
            with open(self.state_file_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved benchmark suite state to: {self.state_file_path}")
        except Exception as e:
            logger.error(f"Failed to save benchmark state: {e}")

    def run(self) -> List[Dict[str, Any]]:
        """Runs the sequence of trials, capturing errors, VRAM, and processing pause/resume."""
        total_trials = len(self.trials_configs)
        logger.info(f"Starting/Resuming benchmark run containing {total_trials} total configurations.")

        while self.current_trial_idx < total_trials:
            if self.paused:
                logger.info("Benchmark runner is PAUSED.")
                break

            trial_meta = self.trials_configs[self.current_trial_idx]
            logger.info(f"--- Running Benchmark Trial {self.current_trial_idx + 1}/{total_trials}: {trial_meta.get('name', 'unnamed')} ---")

            trial_result = self._run_single_trial(trial_meta)
            self.results.append(trial_result)

            self.current_trial_idx += 1
            self.save_state()

        return self.results

    def _run_single_trial(self, trial_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a single training run using temporary config files based on base_config."""
        name = trial_meta.get("name", f"trial_{self.current_trial_idx}")
        overrides = trial_meta.get("overrides", {})

        # Load the base config as dictionary to apply overrides
        with open(self.base_config_path, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f)

        # Deep merge/apply overrides
        for section, key_val in overrides.items():
            if section not in base_data:
                base_data[section] = {}
            for k, v in key_val.items():
                base_data[section][k] = v

        # Set output/experiment specifically for the trial to keep artifacts separated
        if "output" not in base_data:
            base_data["output"] = {}
        base_data["output"]["experiment_name"] = f"benchmark_{name}"

        # Setup unique, secure temp config file
        import tempfile
        temp_config_path = None

        # Result placeholder
        result = {
            "name": name,
            "status": "pending",
            "overrides": overrides,
            "metrics": {},
            "error": None
        }

        # Setup and run trainer
        trainer = None
        start_time = time.time()
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                yaml.safe_dump(base_data, f)
                temp_config_path = f.name

            # We enforce is_test_mode matching configuration
            trainer = SDXLTrainer(temp_config_path, is_test_mode=self.is_test_mode)

            # If the user requested a specific mock VRAM, propagate it for testing/OOM verification
            mock_vram = trial_meta.get("mock_free_vram_bytes")
            if mock_vram is not None:
                trainer._mock_free_vram_bytes = mock_vram

            # Run trainer
            trainer.run()

            duration = time.time() - start_time
            result["status"] = "completed"
            result["metrics"] = {
                "duration_seconds": duration,
                "global_steps": trainer.global_step,
            }

            # Gather CUDA peak VRAM if available
            if torch.cuda.is_available():
                result["metrics"]["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
                result["metrics"]["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)

        except AutoOOMError as oom_err:
            logger.warning(f"Auto OOM detected on trial: {name}. Marking as canceled.")
            result["status"] = "canceled (Auto OOM detected)"
            result["error"] = str(oom_err)
        except Exception as e:
            logger.exception(f"Trial failed with unexpected error: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        finally:
            # Cleanup temp config file
            if temp_config_path and os.path.exists(temp_config_path):
                try:
                    os.remove(temp_config_path)
                except Exception:
                    pass
            # Clear GPU memory
            if trainer:
                del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result
