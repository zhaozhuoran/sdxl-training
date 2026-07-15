import os
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
from trainer.dataset import create_dataloader
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
            logger.info("Test mode active: setting up tiny mock models.")
            # Build tiny mock models so integration tests can execute on CPU
            class MockUNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Match a target layer name so injection catches it.
                    # Input pixel_values have 3 channels.
                    self.to_q = nn.Linear(3, 3)
                    self.to_out = nn.Linear(3, 3)
                def forward(self, x, timesteps=None, encoder_hidden_states=None, added_cond_kwargs=None):
                    b, c, h, w = x.shape
                    x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
                    out_flat = self.to_q(x_flat)
                    out = out_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
                    return out

            class MockEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = nn.Linear(16, 16)
                def forward(self, x):
                    return x

            self.unet = MockUNet().to(self.device, dtype=self.weight_dtype)
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
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer

        model_path = self.config.model.pretrained_model_name_or_path
        logger.info(f"Loading pretrained models from: {model_path}")

        is_single_file = os.path.isfile(model_path) and model_path.lower().endswith((".safetensors", ".ckpt"))

        if is_single_file:
            # Single-file checkpoint (e.g. a standalone SDXL .safetensors / .ckpt)
            # Load the full pipeline once, then extract individual components.
            from diffusers import StableDiffusionXLPipeline

            logger.info("Detected single-file checkpoint. Loading via StableDiffusionXLPipeline.from_single_file(...)")
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=self.weight_dtype, local_files_only=True
            )

            self.unet = pipe.unet
            self.text_encoder_1 = pipe.text_encoder
            self.text_encoder_2 = pipe.text_encoder_2
            # VAE is typically kept in float32 during training to avoid NaN values
            self.vae = pipe.vae.to(dtype=torch.float32)
            self.tokenizer_1 = pipe.tokenizer
            self.tokenizer_2 = pipe.tokenizer_2
            # Reuse the pipeline scheduler (defaults to a compatible DDPM-like scheduler)
            self.noise_scheduler = pipe.scheduler
        else:
            # Standard diffusers directory layout
            # Load models
            self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=self.weight_dtype)
            self.text_encoder_1 = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype)
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=self.weight_dtype)

            vae_path = self.config.model.vae_path or model_path
            subfolder_vae = "vae" if not self.config.model.vae_path else None
            # VAE is typically kept in float32 during training to avoid NaN values
            self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder=subfolder_vae, torch_dtype=torch.float32)

            # Load noise scheduler
            self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

            # Tokenizers
            self.tokenizer_1 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
            self.tokenizer_2 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", use_fast=False)

        # Place models on active device
        self.unet.to(self.device)
        self.text_encoder_1.to(self.device)
        self.text_encoder_2.to(self.device)
        self.vae.to(self.device)

        # Freeze original models
        self.unet.requires_grad_(False)
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)

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
        logger.info(f"Setting up dataset Loader from path: {self.config.dataset.path}")
        self.dataloader = create_dataloader(
            directory_path=self.config.dataset.path,
            batch_size=self.config.dataset.batch_size,
            resolution=self.config.dataset.resolution,
            shuffle=self.config.dataset.shuffle,
            num_workers=self.config.dataset.num_workers if not self.is_test_mode else 0
        )

    def setup_precision(self) -> None:
        """Initializes mixed-precision gradient scaling parameters."""
        mp = self.config.training.mixed_precision.lower()
        if mp == "fp16":
            self.grad_scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision FP16 scaling configured.")
        else:
            self.grad_scaler = None
            logger.info("Mixed precision default / bf16 scaling (no scaler required) configured.")

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

            # Load training markers
            self.global_step = state["step"]
            self.current_epoch = state["epoch"]

            # Load lora, optimizer, scheduler, scaler
            self.lora_manager.load_lora_state_dict(state["lora_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if self.scheduler and state.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            if self.grad_scaler and state.get("grad_scaler_state_dict"):
                self.grad_scaler.load_state_dict(state["grad_scaler_state_dict"])

            # Restore random states
            rngs = state.get("rng_states", {})
            if rngs.get("torch_cpu") is not None:
                torch.set_rng_state(rngs["torch_cpu"])
            if rngs.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rngs["torch_cuda"])

            logger.info(f"Successfully recovered training progress at global_step={self.global_step}, epoch={self.current_epoch}.")

    def cache_dataset(self) -> None:
        """
        Pre-computes and caches VAE latents and Text Encoder outputs.
        Saves them to RAM (in-memory) or Disk (cache files).
        """
        cache_latents = self.config.dataset.cache_latents
        cache_te = self.config.dataset.cache_text_encoder_outputs and not self.config.training.train_text_encoder

        if not cache_latents and not cache_te:
            return

        logger.info("Pre-computing and caching dataset outputs...")
        dataset = self.dataloader.dataset

        # Determine disk cache directory
        cache_dir = self.config.dataset.cache_dir
        if not cache_dir:
            cache_dir = os.path.join(self.config.dataset.path, ".cache_latents")
        self.cache_dir_path = Path(cache_dir)

        if self.config.dataset.cache_destination == "disk":
            self.cache_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching to disk at: {self.cache_dir_path}")
        else:
            logger.info(f"Caching to RAM. Will look up pre-existing disk cache at: {self.cache_dir_path}")

        # Initialize RAM cache dictionaries on the dataset
        dataset.ram_cache = {}
        dataset.cache_destination = self.config.dataset.cache_destination
        dataset.cache_dir_path = self.cache_dir_path
        dataset.cache_latents_enabled = cache_latents
        dataset.cache_te_enabled = cache_te

        self.unet.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()
        if hasattr(self.vae, "eval"):
            self.vae.eval()

        from tqdm import tqdm
        progress_bar = tqdm(
            total=len(dataset),
            desc="Caching dataset",
            dynamic_ncols=True
        )

        try:
            with torch.no_grad():
                for idx in range(len(dataset)):
                    img_path, caption, path_hash = dataset.samples_with_hashes[idx]

                    latents = None
                    prompt_embeds = None
                    pooled_prompt_embeds = None

                    disk_latent_path = self.cache_dir_path / f"latent_{path_hash}.pt" if self.cache_dir_path else None
                    disk_te_path = self.cache_dir_path / f"te_{path_hash}.pt" if self.cache_dir_path else None

                    # 1. Latent calculation
                    if cache_latents:
                        if disk_latent_path and disk_latent_path.exists():
                            latents = torch.load(disk_latent_path, map_location="cpu")
                        else:
                            if self.is_test_mode:
                                latents = torch.randn(1, 3, self.config.dataset.resolution, self.config.dataset.resolution, dtype=self.weight_dtype)
                            else:
                                from PIL import Image
                                try:
                                    with Image.open(img_path) as img:
                                        image = img.convert("RGB")
                                except Exception as e:
                                    raise IOError(f"Error loading image {img_path}: {e}")

                                pixel_values = dataset.transform(image).unsqueeze(0).to(self.device, dtype=torch.float32)
                                latents_dist = self.vae.encode(pixel_values).latent_dist
                                latents = latents_dist.sample() if hasattr(latents_dist, "sample") else latents_dist.mode()
                                latents = latents * self.vae.config.scaling_factor
                                latents = latents.to(self.weight_dtype).cpu()

                            if disk_latent_path and self.config.dataset.cache_destination == "disk":
                                torch.save(latents, disk_latent_path)

                    # 2. Text Encoder outputs calculation
                    if cache_te:
                        if disk_te_path and disk_te_path.exists():
                            te_data = torch.load(disk_te_path, map_location="cpu")
                            prompt_embeds = te_data["prompt_embeds"]
                            pooled_prompt_embeds = te_data["pooled_prompt_embeds"]
                        else:
                            if self.is_test_mode:
                                prompt_embeds = torch.randn(1, 77, 2048, dtype=self.weight_dtype)
                                pooled_prompt_embeds = torch.randn(1, 1280, dtype=self.weight_dtype)
                            else:
                                pe, ppe = self._encode_prompt([caption])
                                prompt_embeds = pe.cpu()
                                pooled_prompt_embeds = ppe.cpu()

                            if disk_te_path and self.config.dataset.cache_destination == "disk":
                                torch.save({
                                    "prompt_embeds": prompt_embeds,
                                    "pooled_prompt_embeds": pooled_prompt_embeds
                                }, disk_te_path)

                    # Store in RAM cache if destination is ram
                    if self.config.dataset.cache_destination == "ram":
                        cache_item = {}
                        if latents is not None:
                            cache_item["latents"] = latents
                        if prompt_embeds is not None:
                            cache_item["prompt_embeds"] = prompt_embeds
                            cache_item["pooled_prompt_embeds"] = pooled_prompt_embeds
                        dataset.ram_cache[path_hash] = cache_item

                    progress_bar.update(1)
        finally:
            progress_bar.close()

        logger.info("Dataset caching completed successfully.")

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
        self, original_size: Tuple[int, int], crops_coords_top_left: Tuple[int, int], target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Constructs SDXL standard micro-conditioning sizes and coordinates IDs."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.weight_dtype)
        return add_time_ids

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Executes a single step forward and backward pass.
        Implements actual custom SDXL LoRA noise prediction loss.
        """
        # Optimize gradient zeroing by setting to none
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
                    latents = batch["latents"].to(self.device, dtype=dtype)
                    # For mock UNet in test mode, UNet forward expects [b, c, h, w] matching latents
                    output = self.unet(latents)
                else:
                    pixel_values = batch["pixel_values"].to(self.device, dtype=dtype)
                    output = self.unet(pixel_values)
                loss = torch.mean((output - 0.0) ** 2)
            else:
                # Real SDXL forward pass
                # 1. Get or encode images to latents
                if "latents" in batch:
                    latents = batch["latents"].to(self.device, dtype=dtype)
                else:
                    pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float32)
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
                    prompt_embeds = batch["prompt_embeds"].to(self.device, dtype=dtype)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(self.device, dtype=dtype)
                else:
                    prompt_embeds, pooled_prompt_embeds = self._encode_prompt(batch["captions"])
                    prompt_embeds = prompt_embeds.to(dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype)

                # 4. Set up standard SDXL micro-conditioning added_cond_kwargs
                # Default SDXL coordinates (e.g. 1024x1024 original, no crop)
                res = self.config.dataset.resolution
                add_time_ids = self._get_add_time_ids((res, res), (0, 0), (res, res))
                # Repeat for batch size
                add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

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

        # Accumulation handling and scaling
        if self.grad_scaler:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    def run(self) -> None:
        """Main training cycle coordinating epochs, checkpoints, and telemetry logging."""
        logger.info("Initializing SDXL LoRA training pipeline...")

        # Setup all dependencies
        self.setup_models()
        self.setup_lora()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_dataloader()
        self.setup_precision()

        # Cache dataset (latents & text encoder outputs) if enabled
        self.cache_dataset()

        # Resume state if required
        self.handle_resume()

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

                        # Structured step logging
                        logger.info(
                            f"[Epoch {self.current_epoch}] [Step {self.global_step}/{self.config.training.steps}] "
                            f"Loss: {loss_val:.4f} | Time: {step_duration:.3f}s"
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
                            self.grad_scaler
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
                            is_snapshot=True
                        )
                        last_snapshot_time = time.time()

                # End of epoch telemetry
                avg_loss = epoch_loss / max(1, step_count)
                logger.info(f"--- Completed Epoch {self.current_epoch} | Average Loss: {avg_loss:.4f} ---")
                self.current_epoch += 1
        finally:
            progress_bar.close()

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
