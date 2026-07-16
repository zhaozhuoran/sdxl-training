import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str = Field(
        ..., description="Path to pretrained model or model identifier from huggingface.co/models."
    )
    vae_path: Optional[str] = Field(None, description="Path to custom VAE.")


class DatasetConfig(BaseModel):
    path: str = Field(..., description="Path to image-caption directory.")
    batch_size: int = Field(1, description="Training batch size.")
    resolution: int = Field(1024, description="Base resolution for aspect-ratio bucketing. Images are assigned to the bucket whose aspect ratio is closest to their own.")
    shuffle: bool = Field(True, description="Shuffle dataset.")
    num_workers: int = Field(4, description="Number of subprocesses for data loading.")
    cache_latents: bool = Field(True, description="Pre-computes and caches VAE latents.")
    cache_text_encoder_outputs: bool = Field(True, description="Pre-computes and caches Text Encoder outputs (only when Text Encoder is not trained).")
    cache_destination: str = Field("ram", description="Destination for caching: 'ram' or 'disk'.")
    cache_dir: Optional[str] = Field(None, description="Custom directory for disk caching. Defaults to a sibling '<dataset>.cache_latents' directory if null.")
    cache_workers: int = Field(8, description="Number of worker threads for parallel image loading and disk-cache reads during precaching.")
    cache_batch_size: int = Field(4, description="Number of samples encoded together on the GPU during precaching. Higher is faster but uses more VRAM.")

    # Aspect-ratio bucketing
    bucket_step: int = Field(64, description="Bucket size step in pixels. Both bucket dimensions are multiples of this (must be 8-safe for the VAE).")
    bucket_min_size: Optional[int] = Field(None, description="Minimum bucket dimension. Defaults to bucket_step if null.")
    bucket_max_size: Optional[int] = Field(None, description="Maximum bucket dimension. Defaults to 1.5x resolution if null.")

    # Caption augmentations (Kohya-style)
    shuffle_caption: bool = Field(False, description="Shuffle comma-separated tags (keeping the first keep_tokens fixed).")
    keep_tokens: int = Field(0, description="Number of leading comma-separated tags to keep in place when shuffle_caption is enabled.")
    caption_dropout_rate: float = Field(0.0, description="Probability of dropping the whole caption (empty) at training time. Requires cache_text_encoder_outputs=false.")
    tag_dropout_rate: float = Field(0.0, description="Probability of dropping each individual comma-separated tag.")

    @field_validator("cache_destination")
    @classmethod
    def validate_cache_destination(cls, v: str) -> str:
        if v not in {"ram", "disk"}:
            raise ValueError("cache_destination must be either 'ram' or 'disk'")
        return v


class TrainingConfig(BaseModel):
    steps: int = Field(..., description="Total number of training steps.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility.")
    gradient_accumulation_steps: int = Field(1, description="Number of updates steps to accumulate before performing a backward/update pass.")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing.")
    mixed_precision: str = Field("bf16", description="Mixed precision compute type. Must be 'fp16', 'bf16', or 'no'.")
    train_text_encoder: bool = Field(False, description="Whether to train the Text Encoders. If False, only train UNet.")

    @field_validator("mixed_precision")
    @classmethod
    def validate_mixed_precision(cls, v: str) -> str:
        if v not in {"fp16", "bf16", "no"}:
            raise ValueError("mixed_precision must be 'fp16', 'bf16', or 'no'")
        return v


class NetworkConfig(BaseModel):
    type: str = Field("lora", description="Network type. Currently only 'lora' is supported in the MVP.")
    rank: int = Field(64, description="Rank of LoRA network.")
    alpha: float = Field(32.0, description="Alpha parameter of LoRA network.")

    # Custom targeted modules overrides
    unet_targets: Optional[List[str]] = Field(None, description="Custom target modules for UNet.")
    text_encoder_1_targets: Optional[List[str]] = Field(None, description="Custom target modules for Text Encoder 1.")
    text_encoder_2_targets: Optional[List[str]] = Field(None, description="Custom target modules for Text Encoder 2.")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v != "lora":
            raise ValueError("Only network type 'lora' is supported in this release.")
        return v


class OptimizerConfig(BaseModel):
    type: str = Field("adamw8bit", description="Optimizer type: 'adamw' or 'adamw8bit'.")
    learning_rate: float = Field(1e-4, description="Learning rate.")
    weight_decay: float = Field(1e-2, description="Weight decay.")
    beta1: float = Field(0.9, description="Beta1 parameter for AdamW.")
    beta2: float = Field(0.999, description="Beta2 parameter for AdamW.")
    epsilon: float = Field(1e-8, description="Epsilon parameter for AdamW.")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in {"adamw", "adamw8bit"}:
            raise ValueError("Only 'adamw' and 'adamw8bit' optimizers are supported in this release.")
        return v


class SchedulerConfig(BaseModel):
    type: str = Field("constant", description="Learning rate scheduler type: 'constant' or 'cosine'.")
    warmup_steps: int = Field(0, description="Number of warmup steps.")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in {"constant", "cosine"}:
            raise ValueError("Only 'constant' and 'cosine' schedulers are supported in this release.")
        return v


class CheckpointConfig(BaseModel):
    save_every_steps: Optional[int] = Field(None, description="Save recovery checkpoint every N steps.")
    save_every_seconds: Optional[int] = Field(None, description="Save recovery checkpoint every N seconds.")
    keep_last_recovery: int = Field(120, description="Maintain a rolling history of the most recent N recovery checkpoints.")

    snapshot_every_steps: Optional[int] = Field(None, description="Save snapshot checkpoint every N steps.")
    snapshot_every_seconds: Optional[int] = Field(None, description="Save snapshot checkpoint every N seconds.")


class ResumeConfig(BaseModel):
    mode: str = Field("none", description="Resume mode: 'none', 'auto', or 'manual'.")
    manual_path: Optional[str] = Field(None, description="Path to specific trainer state checkpoint file if mode is manual.")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in {"none", "auto", "manual"}:
            raise ValueError("Resume mode must be 'none', 'auto', or 'manual'")
        return v

    @model_validator(mode="after")
    def validate_manual_path(self) -> "ResumeConfig":
        if self.mode == "manual" and not self.manual_path:
            raise ValueError("manual_path must be specified when resume mode is 'manual'.")
        return self


class OutputConfig(BaseModel):
    directory: str = Field("outputs/", description="Base output directory.")
    experiment_name: str = Field("default_experiment", description="Name of experiment.")


class TrainingPipelineConfig(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Union[str, Path]) -> TrainingPipelineConfig:
    """Loads and strongly validates the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError("Configuration file is empty.")
    return TrainingPipelineConfig(**data)
