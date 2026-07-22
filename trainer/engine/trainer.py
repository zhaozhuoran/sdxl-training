import os
import gc
import time
import shutil
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from torch.utils.data import DataLoader

from trainer.config import TrainingPipelineConfig, load_config
from trainer.logging import get_logger, setup_file_logger
from trainer.dataset import create_dataloader, _load_cache_file, _save_cache_file
from trainer.checkpoint import CheckpointManager
from trainer.methods.lora.injection import LoRAInjectionManager
from trainer.methods.lora.exporter import export_kohya_safetensors

logger = get_logger()


class SDXLTrainer:
    """
    Core, high-performance, and extensible SDXL LoRA Training Engine.
    Handles component setup, the custom training loop, precision scaling,
    checkpoints intervals, and automated resume recovery.
    """
    def __init__(self, config_path: str, is_test_mode: bool = False):
        self.config_path = config_path
        self.is_test_mode = is_test_mode

        # 1. Load and strongly validate configuration
        self.config: TrainingPipelineConfig = load_config(config_path)

        # 2. Setup output directories
        self.output_dir = os.path.join(self.config.output.directory, self.config.output.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup output logging directory
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        setup_file_logger(os.path.join(log_dir, "train.log"))

        # Copy configuration file to output dir for absolute reproducibility
        shutil.copy2(config_path, os.path.join(self.output_dir, "config.yaml"))

        # Calculate a training configuration hash
        with open(config_path, "rb") as f:
            self.config_hash = hashlib.sha256(f.read()).hexdigest()

        # Try to identify git commit
        self.git_commit = self._get_git_commit()

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.config,
            output_dir=self.output_dir,
            git_commit=self.git_commit,
            config_hash=self.config_hash
        )

        # 3. Handle strict GPU requirements for production runs
        if not self.is_test_mode and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA-compatible GPU is unavailable. This framework is GPU-first, and CPU training is intentionally "
                "unsupported to prevent unintended config errors and impractical execution times. Exiting."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized training engine on device: {self.device}")

        # Models, optimizer, scheduler variables
        self.unet: Optional[nn.Module] = None
        self.text_encoder_1: Optional[nn.Module] = None
        self.text_encoder_2: Optional[nn.Module] = None
        self.vae: Optional[nn.Module] = None
        self.noise_scheduler: Optional[Any] = None
        self.tokenizer_1: Optional[Any] = None
        self.tokenizer_2: Optional[Any] = None

        self.lora_manager: Optional[LoRAInjectionManager] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.dataloader: Optional[DataLoader] = None
        self.grad_scaler: Optional[torch.amp.GradScaler] = None

        self.global_step = 0
        self.current_epoch = 0

    def _get_git_commit(self) -> str:
        """Helper to safely query current git commit if in repository context."""
        try:
            import subprocess
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def setup_models(self) -> None:
        """
        Loads the underlying SDXL models using diffusers & transformers libraries.
        In test mode, we build tiny mock models to run verification on CPU.
        """
        mp_type = self.config.training.mixed_precision.lower()
        self.weight_dtype = torch.float32
        if mp_type == "fp16":
            self.weight_dtype = torch.float16
        elif mp_type == "bf16":
            self.weight_dtype = torch.bfloat16

        if self.is_test_mode:
            logger.info("Test mode active: setting up tiny mock models (VAE + Text Encoders only; UNet loaded later).")
            # Text Encoder mocks so integration tests can execute on CPU
            class MockEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = nn.Linear(16, 16)
                def forward(self, x):
                    return x

            self.text_encoder_1 = MockEncoder().to(self.device, dtype=self.weight_dtype)
            self.text_encoder_2 = MockEncoder().to(self.device, dtype=self.weight_dtype)
            self.vae = nn.Identity().to(self.device)

            class MockNoiseScheduler:
                def add_noise(self, original_samples, noise, timesteps):
                    return original_samples + noise * timesteps.view(-1, 1, 1, 1) / 1000.0
            self.noise_scheduler = MockNoiseScheduler()
            return

        # Real model loading
        # In actual execution, we load the standard diffusers components
        # (this requires diffusers and transformers packages)
        from diffusers import AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer

        model_path = self.config.model.pretrained_model_name_or_path
        logger.info(f"Loading VAE + Text Encoders from: {model_path}")

        # NOTE: The UNet is intentionally NOT loaded here. It is only required for
        # the training loop, not for caching latents/text-encoder outputs. Deferring
        # it (see setup_unet) keeps VRAM free during the precache phase and avoids OOM.
        is_single_file = os.path.isfile(model_path) and model_path.lower().endswith((".safetensors", ".ckpt"))

        if is_single_file:
            from diffusers import StableDiffusionXLPipeline

            logger.info("Detected single-file checkpoint. Loading via StableDiffusionXLPipeline.from_single_file(...)")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=self.weight_dtype, local_files_only=True
            )
            self.text_encoder_1 = pipe.text_encoder
            self.text_encoder_2 = pipe.text_encoder_2
            # VAE is typically kept in float32 during training to avoid NaN values
            self.vae = pipe.vae.to(dtype=torch.float32)
            self.tokenizer_1 = pipe.tokenizer
            self.tokenizer_2 = pipe.tokenizer_2
            self.noise_scheduler = pipe.scheduler
            # The full pipeline (incl. the UNet, which stays on CPU) is no longer
            # needed once components are extracted. Drop it explicitly so the
            # host-memory it holds (UNet weights, etc.) is reclaimed promptly.
            del pipe
            gc.collect()
        else:
            self.text_encoder_1 = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=self.weight_dtype)

            vae_path = self.config.model.vae_path or model_path
            subfolder_vae = "vae" if not self.config.model.vae_path else None
            # VAE is typically kept in float32 during training to avoid NaN values
            self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder=subfolder_vae, torch_dtype=torch.float32)

            self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
            self.tokenizer_1 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
            self.tokenizer_2 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", use_fast=False)

        # VAE is encoded in float32 during caching. Slicing caps the per-batch
        # activation peak but serializes the batch into per-sample passes (slower).
        # It is off by default (speed-first); enable via dataset.cache_vae_slicing
        # on VRAM-constrained GPUs. Peak VRAM is reclaimed each batch regardless
        # via torch.cuda.empty_cache() in cache_dataset.
        if self.config.dataset.cache_vae_slicing:
            self.vae.enable_slicing()

        # Place VAE + Text Encoders on active device
        self.text_encoder_1.to(self.device)
        self.text_encoder_2.to(self.device)
        self.vae.to(self.device)

        # Reclaim the transient peak from model loading (and, for single-file
        # checkpoints, from building the full pipeline) before the cache phase.
        gc.collect()
        torch.cuda.empty_cache()

        # Freeze original models
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)

    def setup_unet(self) -> None:
        """Loads the UNet (only needed for the training loop, not for caching)."""
        if self.unet is not None:
            return
        if self.is_test_mode:
            logger.info("Test mode: building tiny mock UNet.")

            class MockUNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.to_q = nn.Linear(3, 3)
                    self.to_out = nn.Linear(3, 3)
                def forward(self, x, timesteps=None, encoder_hidden_states=None, added_cond_kwargs=None):
                    b, c, h, w = x.shape
                    x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
                    out_flat = self.to_q(x_flat)
                    out = out_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    return out

            self.unet = MockUNet().to(self.device, dtype=self.weight_dtype)
            return

        from diffusers import UNet2DConditionModel
        model_path = self.config.model.pretrained_model_name_or_path
        logger.info(f"Loading UNet from: {model_path}")
        is_single_file = os.path.isfile(model_path) and model_path.lower().endswith((".safetensors", ".ckpt"))
        if is_single_file:
            # Load ONLY the UNet from the single-file checkpoint. Using
            # UNet2DConditionModel.from_single_file avoids rebuilding the entire
            # StableDiffusionXLPipeline (text encoders, VAE, tokenizers,
            # scheduler) just to grab pipe.unet -- that full pipeline is already
            # loaded (and discarded) in setup_models, so re-instantiating it
            # here would double the single-file load cost.
            self.unet = UNet2DConditionModel.from_single_file(
                model_path, torch_dtype=self.weight_dtype, local_files_only=True
            )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=self.weight_dtype)
        self.unet.to(self.device)
        self.unet.requires_grad_(False)

        if self.config.training.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        try:
            self.unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers memory-efficient attention enabled on UNet.")
        except Exception as e:
            logger.warning(
                "xFormers unavailable (%s); falling back to diffusers' default SDPA attention. "
                "Install xformers (matching the installed torch) for lower attention memory.",
                e,
            )

    def setup_lora(self) -> None:
        """Injects generic parameter-efficient LoRA wrapper modules into model components."""
        logger.info("Injecting customizable parameter-efficient LoRA layers...")
        self.lora_manager = LoRAInjectionManager(
            rank=self.config.network.rank,
            alpha=self.config.network.alpha,
            unet_targets=self.config.network.unet_targets,
            text_encoder_1_targets=self.config.network.text_encoder_1_targets,
            text_encoder_2_targets=self.config.network.text_encoder_2_targets
        )

        self.lora_manager.inject_lora(self.unet, prefix="unet", targets=self.lora_manager.unet_targets)

        if self.config.training.train_text_encoder:
            logger.info("Training of Text Encoders is enabled. Injecting LoRA layers into Text Encoders.")
            self.lora_manager.inject_lora(self.text_encoder_1, prefix="te1", targets=self.lora_manager.te1_targets)
            self.lora_manager.inject_lora(self.text_encoder_2, prefix="te2", targets=self.lora_manager.te2_targets)
        else:
            logger.info("Training of Text Encoders is disabled. UNet-only LoRA training will be performed.")

        # Shift injected parameters to target device with the appropriate dtype
        for wrapper in self.lora_manager.injected_modules.values():
            wrapper.to(self.device, dtype=self.weight_dtype)

    def setup_optimizer(self) -> None:
        """Initializes requested optimizer on our isolated parameters."""
        params = self.lora_manager.get_lora_parameters()
        opt_type = self.config.optimizer.type.lower()
        lr = self.config.optimizer.learning_rate
        weight_decay = self.config.optimizer.weight_decay
        betas = (self.config.optimizer.beta1, self.config.optimizer.beta2)
        eps = self.config.optimizer.epsilon

        logger.info(f"Setting up optimizer: '{opt_type}' with learning_rate={lr}")

        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps
            )
        elif opt_type == "adamw8bit":
            # For standard adamw8bit optimizer, we try to use bitsandbytes.
            # Fallback cleanly to standard torch AdamW if bitsandbytes is not installed.
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps
                )
                logger.info("Successfully loaded 8-bit AdamW from bitsandbytes.")
            except ImportError:
                logger.warning("bitsandbytes package is not installed. Falling back to standard AdamW optimizer.")
                self.optimizer = torch.optim.AdamW(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps
                )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def setup_scheduler(self) -> None:
        """Builds learning rate scheduling behaviors."""
        total_steps = self.config.training.steps
        warmup_steps = self.config.scheduler.warmup_steps
        sched_type = self.config.scheduler.type.lower()

        logger.info(f"Setting up scheduler: '{sched_type}' with warmup_steps={warmup_steps}")

        if sched_type == "constant":
            # Constant scheduler with warmup
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif sched_type == "cosine":
            # Cosine decay scheduler with warmup
            import math
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")

    def setup_dataloader(self) -> None:
        """Creates dataset loader pipeline from image-caption folder."""
        from trainer.caption import CaptionProcessor

        ds = self.config.dataset
        self.caption_processor = CaptionProcessor(
            shuffle_caption=ds.shuffle_caption,
            keep_tokens=ds.keep_tokens,
            tag_dropout_rate=ds.tag_dropout_rate,
            caption_dropout_rate=ds.caption_dropout_rate,
            seed=self.config.training.seed,
        )

        logger.info(f"Setting up dataset Loader from path: {self.config.dataset.path}")
        self.dataloader = create_dataloader(
            directory_path=self.config.dataset.path,
            batch_size=self.config.dataset.batch_size,
            resolution=self.config.dataset.resolution,
            shuffle=self.config.dataset.shuffle,
            num_workers=self.config.dataset.num_workers if not self.is_test_mode else 0,
            bucket_step=self.config.dataset.bucket_step,
            bucket_min_size=self.config.dataset.bucket_min_size,
            bucket_max_size=self.config.dataset.bucket_max_size,
            caption_processor=self.caption_processor,
            seed=self.config.training.seed,
            pin_memory=(self.device.type == "cuda"),
        )

    def setup_precision(self) -> None:
        """Initializes mixed-precision gradient scaling parameters."""
        mp = self.config.training.mixed_precision.lower()
        if mp == "fp16":
            self.grad_scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision FP16 scaling configured.")
        elif mp == "no":
            self.grad_scaler = None
            logger.info("Mixed precision disabled (fp32); no scaler configured.")
            if torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_gb < 16:
                    logger.warning(
                        "mixed_precision='no' keeps the UNet in fp32 (~5.8GB) and disables "
                        "GradScaler; on GPUs with <16GB VRAM this risks OOM and is slower. "
                        "Recommend 'bf16'."
                    )
        else:
            self.grad_scaler = None
            logger.info("Mixed precision bf16 scaling (no scaler required) configured.")

    def handle_resume(self) -> None:
        """
        Coordinates full state recovery in auto-resume or manual config mode.
        Validates found checkpoints and falls back cleanly if corruptions are detected.
        """
        resume_mode = self.config.resume.mode.lower()
        if resume_mode == "none":
            return

        target_path: Optional[os.PathLike] = None

        if resume_mode == "manual":
            target_path = self.config.resume.manual_path
            logger.info(f"Manual resume requested. Pointing to: {target_path}")
            valid, err = self.checkpoint_manager.validate_checkpoint(Path(target_path) if isinstance(target_path, str) else target_path)
            if not valid:
                raise RuntimeError(f"Manual resume path invalid: {err}")
        elif resume_mode == "auto":
            logger.info("Auto resume mode requested. Scanning for latest valid checkpoint...")
            target_path = self.checkpoint_manager.get_auto_resume_checkpoint()
            if not target_path:
                logger.info("No valid checkpoints found for automatic recovery. Starting training from scratch.")
                return

        if target_path:
            logger.info(f"Resuming training from state checkpoint: {target_path}")
            # Map location safely
            state = torch.load(target_path, map_location=self.device)

            # Guard against resuming with a changed configuration. The checkpoint
            # does not store the full config, so a mismatch blends old training
            # state with new hyper-parameters (silently corrupting or crashing).
            self._enforce_config_compatibility(state)

            # Load training markers
            self.global_step = state["step"]
            self.current_epoch = state["epoch"]

            # Load lora weights (always present)
            self.lora_manager.load_lora_state_dict(state["lora_state_dict"])

            # Optimizer/scheduler/scaler are stored in EVERY checkpoint (both
            # rolling recovery and snapshots), so resume restores exact optimizer
            # momentum and LR schedule. The guard below only exists for loading
            # legacy checkpoints that predate this convention.
            if "optimizer_state_dict" in state:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            else:
                logger.warning(
                    "Checkpoint has no optimizer state; resuming with a freshly "
                    "initialized optimizer (momentum/LR schedule restart)."
                )
            if self.scheduler and state.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            if self.grad_scaler and state.get("grad_scaler_state_dict"):
                self.grad_scaler.load_state_dict(state["grad_scaler_state_dict"])

            # Restore random states. The checkpoint may have been loaded with
            # map_location="cuda", which relocates the CPU RNG-state ByteTensor(s)
            # onto the GPU; torch.set_rng_state requires a CPU ByteTensor, so move
            # them back to CPU first.
            rngs = state.get("rng_states", {})
            if rngs.get("torch_cpu") is not None:
                torch.set_rng_state(rngs["torch_cpu"].cpu())
            if rngs.get("torch_cuda") is not None and torch.cuda.is_available():
                cuda_states = rngs["torch_cuda"]
                if isinstance(cuda_states, (list, tuple)):
                    cuda_states = [s.cpu() for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
            if rngs.get("caption") is not None and getattr(self, "caption_processor", None) is not None:
                self.caption_processor.set_rng_state(rngs["caption"])

            logger.info(f"Successfully recovered training progress at global_step={self.global_step}, epoch={self.current_epoch}.")

    def _gather_rng_states(self) -> Dict[str, Any]:
        """Captures all RNG states for checkpointing, including caption augmentation."""
        states = self.checkpoint_manager._get_current_rng_states()
        if getattr(self, "caption_processor", None) is not None:
            states["caption"] = self.caption_processor.get_rng_state()
        return states

    def _enforce_config_compatibility(self, state: dict) -> None:
        """
        Blocks silent recovery when the current config differs from the one the
        checkpoint was saved with. A mismatch blends old training state (step,
        LoRA weights, optimizer momentum) with new hyper-parameters, which can
        corrupt training or crash on structural changes (rank, resolution, model...).

        Requires an explicit interactive confirmation to proceed with replacing
        the old config. Aborts when no interactive input is available.
        """
        old_hash = (state.get("metadata") or {}).get("config_hash")
        new_hash = self.config_hash
        if not old_hash or old_hash == new_hash:
            return

        logger.warning("=" * 64)
        logger.warning("CONFIG MISMATCH: the checkpoint was saved with a DIFFERENT configuration.")
        logger.warning(f"  checkpoint config_hash: {old_hash}")
        logger.warning(f"  current   config_hash: {new_hash}")
        logger.warning("Resuming will DISCARD the old config and continue with the CURRENT one.")
        logger.warning("This can corrupt training if structural params changed")
        logger.warning("(e.g. rank, resolution, model path, dataset path, train_text_encoder).")
        logger.warning("=" * 64)

        if self.is_test_mode:
            logger.warning("Test mode: auto-confirming config replacement (no interactive prompt).")
            return

        try:
            answer = input(
                "Type 'yes' to REPLACE the old config and resume, or anything else to abort: "
            ).strip().lower()
        except EOFError:
            logger.error("No interactive input available; aborting resume to avoid unsafe config replacement.")
            raise RuntimeError("Aborted: config mismatch and no confirmation provided.")

        if answer != "yes":
            raise RuntimeError("Aborted by user: config mismatch not confirmed.")

    def cache_dataset(self) -> None:
        """
        Pre-computes and caches VAE latents and Text Encoder outputs.
        Saves them to RAM (in-memory) or Disk (cache files).
        """
        cache_latents = self.config.dataset.cache_latents
        cache_te = self.config.dataset.cache_text_encoder_outputs and not self.config.training.train_text_encoder

        # Whole-caption dropout cannot be applied to cached Text Encoder outputs
        # (the embeddings would be fixed). Disable TE caching in that case so the
        # dropout is applied fresh per training step instead.
        if cache_te and self.config.dataset.caption_dropout_rate > 0:
            logger.warning(
                "caption_dropout_rate > 0 but cache_text_encoder_outputs is enabled. "
                "Disabling Text Encoder output caching so caption dropout applies per step."
            )
            cache_te = False

        # Record which outputs are actually cached so the training loop can
        # offload the now-unused VAE / Text Encoders to free GPU memory.
        self._latents_cached = cache_latents
        self._te_cached = cache_te

        # Precision for the VAE encode during precaching (fp32 safe, bf16 faster/lower VRAM).
        cache_vae_dtype = self.config.dataset.cache_vae_dtype

        if not cache_latents and not cache_te:
            return

        logger.info("Pre-computing and caching dataset outputs...")
        dataset = self.dataloader.dataset

        # Map each sample's path hash to its bucket / original_size / crop_ltrb so the
        # cache loader can validate latent shapes and store SDXL conditioning metadata.
        meta_by_hash = {}
        for (_, _c, h), m in zip(dataset.samples_with_hashes, dataset.sample_meta):
            meta_by_hash[h] = m

        # Determine disk cache directory.
        # By default the cache lives in a sibling directory named "<dataset>.cache_latents"
        # (e.g. "/path/dataset" -> "/path/dataset.cache_latents") so it never pollutes the
        # dataset folder. The cache is ALWAYS written to disk (kohya-style) regardless of
        # cache_destination, so subsequent runs load instantly instead of re-encoding.
        cache_dir = self.config.dataset.cache_dir
        if not cache_dir:
            ds_path = Path(self.config.dataset.path)
            cache_dir = str(ds_path.parent / (ds_path.name + ".cache_latents"))
        self.cache_dir_path = Path(cache_dir)
        self.cache_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Latent/TextEncoder cache is always persisted to disk at: {self.cache_dir_path} "
            f"(cache_destination='{self.config.dataset.cache_destination}')"
        )

        # Initialize RAM cache dictionaries on the dataset
        dataset.ram_cache = {}
        dataset.cache_destination = self.config.dataset.cache_destination
        dataset.cache_dir_path = self.cache_dir_path
        dataset.cache_latents_enabled = cache_latents
        dataset.cache_te_enabled = cache_te

        # The UNet is loaded after caching (to save VRAM during the precache phase),
        # so only put the VAE + Text Encoders into eval mode here.
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        if hasattr(self.vae, "eval"):
            self.vae.eval()

        cache_destination = self.config.dataset.cache_destination

        # --- Pre-scan: decide which samples actually need GPU encoding ---
        # A sample needs encoding if its on-disk cache is missing or invalid for any
        # of the enabled outputs (latents / text-encoder embeddings). Fully-cached
        # samples are excluded from the encode loop entirely, so the progress bar and
        # ETA reflect only real VAE / Text Encoder work from the very first batch.
        # For RAM caching, fully-cached payloads are loaded straight into ram_cache
        # during this scan (a single read), avoiding a second load in the loop.
        to_encode = []
        for (img_path, caption, path_hash) in dataset.samples_with_hashes:
            meta = meta_by_hash[path_hash]
            expected_latent = (4, meta["bucket"][1] // 8, meta["bucket"][0] // 8)
            disk_path = self.cache_dir_path / f"cache_{path_hash}.pt"

            latents_ok = False
            te_ok = False
            cached = None
            if disk_path.exists():
                try:
                    cached = _load_cache_file(disk_path)
                except Exception:
                    cached = {}
                if cache_latents and cached.get("latents") is not None:
                    if tuple(cached["latents"].shape)[1:] == expected_latent[1:]:
                        latents_ok = True
                if cache_te and cached.get("prompt_embeds") is not None:
                    te_ok = True

            needs = (cache_latents and not latents_ok) or (cache_te and not te_ok)
            if needs:
                to_encode.append((img_path, caption, path_hash))
            elif cache_destination == "ram" and cached:
                # Fully cached: populate RAM cache directly from the scanned payload.
                item_cache = {}
                if cache_latents and cached.get("latents") is not None:
                    item_cache["latents"] = cached["latents"]
                if cache_te and cached.get("prompt_embeds") is not None:
                    item_cache["prompt_embeds"] = cached["prompt_embeds"]
                    item_cache["pooled_prompt_embeds"] = cached.get("pooled_prompt_embeds")
                if item_cache:
                    dataset.ram_cache[path_hash] = item_cache

        total = len(to_encode)
        workers = self.config.dataset.cache_workers
        batch_size = max(1, self.config.dataset.cache_batch_size)
        resolution = self.config.dataset.resolution
        weight_dtype = self.weight_dtype
        device = self.device
        is_test_mode = self.is_test_mode

        def load_item(item):
            img_path, caption, path_hash = item
            meta = meta_by_hash[path_hash]
            bucket = meta["bucket"]
            original_size = meta["original_size"]
            crop_ltrb = meta["crop_ltrb"]
            disk_path = self.cache_dir_path / f"cache_{path_hash}.pt"

            latents = None
            prompt_embeds = None
            pooled_prompt_embeds = None
            pixel_values = None

            expected_latent = (4, bucket[1] // 8, bucket[0] // 8)

            if disk_path.exists():
                try:
                    cached = _load_cache_file(disk_path)
                except Exception:
                    cached = {}
                if cache_latents and cached.get("latents") is not None:
                    lt = cached["latents"]
                    # Validate latent spatial size matches the current bucket; otherwise
                    # the config (resolution/bucket range) changed and we must recompute.
                    if tuple(lt.shape)[1:] == expected_latent[1:]:
                        latents = lt
                if cache_te and cached.get("prompt_embeds") is not None:
                    prompt_embeds = cached["prompt_embeds"]
                    pooled_prompt_embeds = cached.get("pooled_prompt_embeds")

            if is_test_mode:
                if cache_latents and latents is None:
                    latents = torch.randn(1, 3, bucket[1], bucket[0], dtype=weight_dtype)
                if cache_te and prompt_embeds is None:
                    prompt_embeds = torch.randn(1, 77, 2048, dtype=weight_dtype)
                    pooled_prompt_embeds = torch.randn(1, 1280, dtype=weight_dtype)
            elif cache_latents and latents is None:
                from PIL import Image
                try:
                    with Image.open(img_path) as img:
                        image = img.convert("RGB")
                except Exception as e:
                    raise IOError(f"Error loading image {img_path}: {e}")
                pixel_values = dataset._transform(image, meta)

            # Pre-compute caption augmentation (tag shuffle / dropout) here, in the
            # prefetch thread, so the GPU encode path only tokenizes + encodes and
            # the main thread never stalls on CPU caption work. Only needed when TE
            # outputs are being cached; otherwise left as-is (unused downstream).
            processed_caption = self.caption_processor.process(caption) if cache_te else caption

            return {
                "path_hash": path_hash,
                "caption": caption,
                "processed_caption": processed_caption,
                "pixel_values": pixel_values,
                "latents": latents,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "bucket": bucket,
                "original_size": original_size,
                "crop_ltrb": crop_ltrb,
                "disk_path": disk_path,
            }

        def prefetch_samples(samples):
            if workers <= 0:
                for s in samples:
                    yield load_item(s)
                return
            from concurrent.futures import ThreadPoolExecutor
            # Keep a deeper prefetch window than the encode batch so the GPU never
            # idles between flushes waiting on CPU image decode / disk-cache reads.
            window = max(workers * 2, batch_size * 4)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {}
                submitted = 0
                while submitted < len(samples) and len(futures) < window:
                    f = ex.submit(load_item, samples[submitted])
                    futures[f] = submitted
                    submitted += 1
                next_idx = submitted
                while futures:
                    f = next(iter(futures))
                    result = f.result()
                    del futures[f]
                    if next_idx < len(samples):
                        nf = ex.submit(load_item, samples[next_idx])
                        futures[nf] = next_idx
                        next_idx += 1
                    yield result

        from tqdm import tqdm
        progress_bar = tqdm(
            total=total,
            desc="Caching dataset",
            dynamic_ncols=True
        )
        processed = 0

        buffer: list = []

        def flush():
            nonlocal processed
            if not buffer:
                return

            # Latents must be encoded per-bucket (each bucket has its own spatial size),
            # so group the GPU encode by bucket before stacking.
            from collections import defaultdict
            to_encode_latents = [b for b in buffer if cache_latents and b["latents"] is None]
            groups = defaultdict(list)
            for b in to_encode_latents:
                groups[b["bucket"]].append(b)
            for _bucket, grp in groups.items():
                pv = torch.stack([x["pixel_values"] for x in grp]).to(device, dtype=self.vae.dtype)
                # Encode in bf16 (autocast) when requested to cut peak VRAM + time.
                # VAE weights stay fp32; the result is cast to the training weight_dtype
                # for storage, so stored quality is unaffected (only the encode path differs).
                if cache_vae_dtype == "bf16" and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        latents_dist = self.vae.encode(pv).latent_dist
                else:
                    latents_dist = self.vae.encode(pv).latent_dist
                out = latents_dist.sample() if hasattr(latents_dist, "sample") else latents_dist.mode()
                out = (out * self.vae.config.scaling_factor).to(weight_dtype).cpu()
                for x, lat in zip(grp, out):
                    x["latents"] = lat

            to_te = [b for b in buffer if cache_te and b["prompt_embeds"] is None]
            if to_te:
                # Caption augmentation was already applied in the prefetch thread
                # (item["processed_caption"]), so here we only tokenize + encode.
                captions = [b["processed_caption"] for b in to_te]
                pe, ppe = self._encode_prompt(captions)
                pe = pe.cpu()
                ppe = ppe.cpu()
                for b, e in zip(to_te, zip(pe, ppe)):
                    b["prompt_embeds"] = e[0]
                    b["pooled_prompt_embeds"] = e[1]

            for b in buffer:
                # Always persist to disk cache (kohya-style) so reruns are instant.
                # Our own cache schema stores latents + text-encoder outputs together
                # with the SDXL conditioning metadata (original_size, crop, bucket).
                if (cache_latents and b["latents"] is not None) or (cache_te and b["prompt_embeds"] is not None):
                    payload = {
                        "format": "sdxl-trainer-cache",
                        "version": 1,
                        "latents": b["latents"] if cache_latents else None,
                        "prompt_embeds": b["prompt_embeds"] if cache_te else None,
                        "pooled_prompt_embeds": b["pooled_prompt_embeds"] if cache_te else None,
                        "original_size": b["original_size"],
                        "crop_ltrb": b["crop_ltrb"],
                        "bucket_size": b["bucket"],
                    }
                    _save_cache_file(payload, b["disk_path"])

                # Additionally keep an in-RAM copy when destination is ram.
                if cache_destination == "ram":
                    item = {}
                    if cache_latents and b["latents"] is not None:
                        item["latents"] = b["latents"]
                    if cache_te and b["prompt_embeds"] is not None:
                        item["prompt_embeds"] = b["prompt_embeds"]
                        item["pooled_prompt_embeds"] = b["pooled_prompt_embeds"]
                    if item:
                        dataset.ram_cache[b["path_hash"]] = item

                processed += 1
                progress_bar.update(1)
            buffer.clear()

            # Reclaim freed GPU memory after each flush() so the CUDA caching
            # allocator returns blocks to the driver instead of growing reserved
            # memory monotonically. A single flush may encode several buckets of
            # different sizes, whose varying-shape activations are the main source
            # of allocator fragmentation (mirrors kohya's clean_memory_on_device
            # called after each cache batch).
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

        try:
            with torch.no_grad():
                for loaded in prefetch_samples(to_encode):
                    buffer.append(loaded)
                    if len(buffer) >= batch_size:
                        flush()
                flush()
        finally:
            progress_bar.close()

        if total == 0:
            logger.info("All dataset outputs already cached; no GPU encoding required.")

        logger.info("Dataset caching completed successfully.")

    def _offload_unused_models(self) -> None:
        """Moves models that are fully cached and not trained off the GPU.

        During the training loop, cached latents/TE outputs are read straight
        from the dataset, so the VAE and Text Encoders are never invoked. Leaving
        them resident on the GPU wastes ~1.5-2GB of VRAM (kohya offloads them).
        They are moved back to the device on demand inside train_step if a
        non-cached path is actually taken.
        """
        if getattr(self, "_latents_cached", False):
            self.vae.to("cpu")
        if getattr(self, "_te_cached", False):
            self.text_encoder_1.to("cpu")
            self.text_encoder_2.to("cpu")

        # Reclaim the now-free GPU memory immediately so the CUDA caching
        # allocator returns the blocks to the driver instead of holding them
        # reserved (the offloaded models are no longer used in the loop).
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def _encode_prompt(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes prompts using text encoders 1 & 2 for SDXL conditioning.
        Produces:
          - prompt_embeds (batch, sequence_length, embedding_dim)
          - pooled_prompt_embeds (batch, pooled_embedding_dim)
        """
        # Tokenizer 1 and Text Encoder 1
        text_inputs_1 = self.tokenizer_1(
            prompts,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_1 = text_inputs_1.input_ids.to(self.device)

        # We output hidden states for the penultimate layer of Text Encoder 1
        encoder_outputs_1 = self.text_encoder_1(input_ids_1, output_hidden_states=True)
        # SDXL uses the penultimate layer's hidden states
        prompt_embeds_1 = encoder_outputs_1.hidden_states[-2]

        # Tokenizer 2 and Text Encoder 2
        text_inputs_2 = self.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_2 = text_inputs_2.input_ids.to(self.device)

        encoder_outputs_2 = self.text_encoder_2(input_ids_2, output_hidden_states=True)
        prompt_embeds_2 = encoder_outputs_2.hidden_states[-2]
        # Use .text_embeds attribute to avoid shape mismatch (CLIPTextModelWithProjection output text_embeds is the pooled embedding projection)
        pooled_prompt_embeds = encoder_outputs_2.text_embeds

        # Concatenate embeddings on the last dimension
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def _get_add_time_ids(
        self,
        original_sizes: List[Tuple[int, int]],
        crop_originals: List[Tuple[int, int]],
        target_sizes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Builds SDXL micro-conditioning IDs for a batch from real per-image metadata.

        Each row follows the SDXL/diffusers convention
        ``[orig_h, orig_w, crop_top, crop_left, target_h, target_w]`` using the
        TRUE original image size and the crop offsets expressed in original-image
        space (matching kohya / diffusers), not the resized-to-cover size.
        """
        rows = []
        for (ow, oh), (t, l), (tw, th) in zip(original_sizes, crop_originals, target_sizes):
            rows.append([oh, ow, t, l, th, tw])
        # SDXL micro-conditioning time_ids are conditioning metadata, not model
        # weights; keep them float32 regardless of the training precision so the
        # UNet's added-cond projection is not fed low-precision (bf16/fp16) inputs.
        return torch.tensor(rows, device=self.device, dtype=torch.float32)

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Executes a single step forward and backward pass.
        Implements actual custom SDXL LoRA noise prediction loss.
        """
        # Gradient accumulation: only clear gradients at the start of an
        # accumulation window so they accumulate across micro-steps.
        acc_steps = max(1, int(self.config.training.gradient_accumulation_steps))
        if self.global_step % acc_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Mixed precision compute context
        mp_type = self.config.training.mixed_precision.lower()
        device_type = "cuda" if "cuda" in self.device.type else "cpu"
        dtype = torch.float32
        if mp_type == "fp16":
            dtype = torch.float16
        elif mp_type == "bf16":
            dtype = torch.bfloat16

        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            if self.is_test_mode:
                if "latents" in batch:
                    # In test mode with caching
                    latents = batch["latents"].to(self.device, dtype=dtype, non_blocking=True)
                    # For mock UNet in test mode, UNet forward expects [b, c, h, w] matching latents
                    output = self.unet(latents)
                else:
                    pixel_values = batch["pixel_values"].to(self.device, dtype=dtype, non_blocking=True)
                    output = self.unet(pixel_values)
                loss = torch.mean((output - 0.0) ** 2)
            else:
                # Real SDXL forward pass
                # 1. Get or encode images to latents
                if "latents" in batch:
                    latents = batch["latents"].to(self.device, dtype=dtype, non_blocking=True)
                else:
                    if self.vae.device != self.device:
                        self.vae.to(self.device)
                    pixel_values = batch["pixel_values"].to(self.device, dtype=self.vae.dtype, non_blocking=True)
                    # Pass through standard VAE to extract latents
                    latents = self.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    latents = latents.to(dtype)

                # 2. Add noise mathematically correctly using DDPMScheduler
                noise = torch.randn_like(latents)
                # Sample random timestep
                timesteps = torch.randint(
                    0,
                    getattr(self.noise_scheduler.config, "num_train_timesteps", 1000),
                    (latents.shape[0],),
                    device=self.device
                ).long()
                # Use standard noise scheduler API
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Get or encode prompt captions
                if "prompt_embeds" in batch:
                    prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=dtype, non_blocking=True)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, dtype=dtype, non_blocking=True)
                else:
                    # Text Encoder outputs are not cached (e.g. caption dropout enabled):
                    # process captions fresh, applying whole-caption dropout per step.
                    if self.text_encoder_1.device != self.device:
                        self.text_encoder_1.to(self.device)
                        self.text_encoder_2.to(self.device)
                    captions = [self.caption_processor.maybe_drop_caption(c) for c in batch["captions"]]
                    prompt_embeds, pooled_prompt_embeds = self._encode_prompt(captions)
                    prompt_embeds = prompt_embeds.to(dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype)

                # 4. SDXL micro-conditioning from the REAL per-image original size / crop.
                add_time_ids = self._get_add_time_ids(
                    batch["true_original_sizes"], batch["crop_originals"], batch["bucket_sizes"]
                )

                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }

                # 5. Execute noise prediction using UNet
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # 6. L2 Loss
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Scale the loss so gradients average correctly over the accumulation
        # window, then accumulate. The optimizer/scheduler only advance at the
        # end of a window (or on the very last step) to honor accumulation.
        scaled_loss = loss / acc_steps
        if self.grad_scaler:
            self.grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_window_end = (
            (self.global_step + 1) % acc_steps == 0
            or (self.global_step + 1) >= self.config.training.steps
        )
        if is_window_end:
            if self.grad_scaler:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        return loss.item()

    def _log_vram(self, tag: str) -> None:
        """Logs current + peak CUDA memory at a phase boundary (no-op on CPU)."""
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(
            f"[VRAM][{tag}] allocated={alloc:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB"
        )

    def run(self) -> None:
        """Main training cycle coordinating epochs, checkpoints, and telemetry logging."""
        logger.info("Initializing SDXL LoRA training pipeline...")

        # Enable TF32 matmul + cuDNN benchmark for free throughput on Ampere+ GPUs.
        # These are global, process-wide settings; only meaningful on CUDA.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled TF32 matmul + cuDNN benchmark for CUDA.")

        # Setup all dependencies.
        # Models needed for caching (VAE + Text Encoders) are loaded first; the UNet
        # is deferred until after caching to keep VRAM free during the precache phase.
        self.setup_models()
        self._log_vram("after_setup_models")
        self.setup_dataloader()

        # Cache dataset (latents & text encoder outputs) if enabled
        self.cache_dataset()
        self._log_vram("after_cache_dataset")

        # Load the UNet only now, after caching is done.
        self.setup_unet()
        self.setup_lora()

        # Optionally compile the UNet for faster training. Must happen AFTER LoRA
        # injection so the compiled graph includes the LoRA wrapper layers.
        # dynamic=True because bucket sizes vary across batches (no recompile per
        # bucket). Skipped in test mode (mock models, compilation is meaningless
        # and requires a host C compiler). Real compilation failures surface at
        # the first forward and fall back to eager via the guard below.
        if (
            self.config.training.compile_unet
            and not self.is_test_mode
            and self.device.type == "cuda"
        ):
            try:
                self.unet = torch.compile(self.unet, dynamic=True)
                logger.info("torch.compile enabled on UNet (dynamic shapes).")
            except Exception as e:
                logger.warning("torch.compile failed (%s); using eager UNet.", e)

        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_precision()
        self._log_vram("after_setup_unet_lora")

        # Resume state if required
        self.handle_resume()

        # Offload VAE / Text Encoders that are fully cached and not trained; they
        # are unused during the training loop and only consume resident VRAM.
        if torch.cuda.is_available():
            self._offload_unused_models()

        # Main Loop
        last_recovery_time = time.time()
        last_snapshot_time = time.time()
        start_step = self.global_step

        logger.info(f"Starting training loop from step {start_step} to {self.config.training.steps}")

        from tqdm import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        progress_bar = tqdm(
            total=self.config.training.steps,
            initial=start_step,
            desc="Training steps",
            dynamic_ncols=True
        )

        # Reset peak stats so the loop-phase telemetry reflects training only
        # (not the model-load / precache spikes measured above).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._log_vram("before_training_loop")

        try:
            with logging_redirect_tqdm(loggers=[logger]):
                while self.global_step < self.config.training.steps:
                    # UNet is set to training mode
                    self.unet.train()
                    # Keep frozen model parts in eval mode
                    self.text_encoder_1.eval()
                    self.text_encoder_2.eval()
                    if hasattr(self.vae, "eval"):
                        self.vae.eval()

                    epoch_loss = 0.0
                    step_count = 0

                    for batch in self.dataloader:
                        if self.global_step >= self.config.training.steps:
                            break

                        step_start_time = time.time()
                        loss_val = self.train_step(batch)

                        self.global_step += 1
                        step_count += 1
                        epoch_loss += loss_val

                        # Calculate step time
                        step_duration = time.time() - step_start_time

                        # Structured step logging (throttled: one line per 1k steps;
                        # the tqdm bar still updates every step for live progress)
                        if self.global_step % 1000 == 0:
                            peak = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
                            logger.info(
                                f"[Epoch {self.current_epoch}] [Step {self.global_step}/{self.config.training.steps}] "
                                f"Loss: {loss_val:.4f} | Time: {step_duration:.3f}s | VRAM_peak: {peak:.0f}MB"
                            )

                        # Update tqdm progress bar with postfix metadata
                        progress_bar.set_postfix({
                            "loss": f"{loss_val:.4f}",
                            "epoch": self.current_epoch
                        })
                        progress_bar.update(1)

                        # Recovery Checkpoint Policy by Steps or Time
                        time_since_recovery = time.time() - last_recovery_time
                        trigger_step = (
                            self.config.checkpoint.save_every_steps
                            and self.global_step % self.config.checkpoint.save_every_steps == 0
                        )
                        trigger_time = (
                            self.config.checkpoint.save_every_seconds
                            and time_since_recovery >= self.config.checkpoint.save_every_seconds
                        )

                        if trigger_step or trigger_time:
                            logger.info(f"Triggering rolling recovery checkpoint at step={self.global_step}...")
                            self.checkpoint_manager.save_checkpoint(
                                self.global_step,
                                self.current_epoch,
                                self.lora_manager,
                                self.optimizer,
                                self.scheduler,
                                self.grad_scaler,
                                rng_states=self._gather_rng_states(),
                            )
                            last_recovery_time = time.time()

                        # Snapshot Checkpoint Policy by Steps or Time (measured completely independently of recovery triggers)
                        time_since_snapshot = time.time() - last_snapshot_time
                        trigger_snap_step = (
                            self.config.checkpoint.snapshot_every_steps
                            and self.global_step % self.config.checkpoint.snapshot_every_steps == 0
                        )
                        trigger_snap_time = (
                            self.config.checkpoint.snapshot_every_seconds
                            and time_since_snapshot >= self.config.checkpoint.snapshot_every_seconds
                        )

                        if trigger_snap_step or trigger_snap_time:
                            logger.info(f"Triggering long-term snapshot checkpoint at step={self.global_step}...")
                            self.checkpoint_manager.save_checkpoint(
                                self.global_step,
                                self.current_epoch,
                                self.lora_manager,
                                self.optimizer,
                                self.scheduler,
                                self.grad_scaler,
                                is_snapshot=True,
                                rng_states=self._gather_rng_states(),
                            )
                            last_snapshot_time = time.time()

                # End of epoch telemetry
                avg_loss = epoch_loss / max(1, step_count)
                logger.info(f"--- Completed Epoch {self.current_epoch} | Average Loss: {avg_loss:.4f} ---")
                self.current_epoch += 1
        finally:
            progress_bar.close()
            # Flush any pending background checkpoint writes before exiting.
            self.checkpoint_manager.shutdown()

        self._log_vram("after_training")

        # Post training: Export final LoRA safetensors to lora directory
        logger.info("Training completed successfully. Exporting final compatible LoRA safetensors...")
        final_lora_path = os.path.join(
            self.checkpoint_manager.lora_dir,
            f"step-{self.global_step:06d}.safetensors"
        )
        metadata = self.checkpoint_manager.build_metadata(self.global_step, self.current_epoch)
        export_kohya_safetensors(
            self.lora_manager.get_lora_state_dict(),
            self.config.network.alpha,
            final_lora_path,
            metadata=metadata
        )
        logger.info(f"Exported compatible LoRA file successfully to: {final_lora_path}")
