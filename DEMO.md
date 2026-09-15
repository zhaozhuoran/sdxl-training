# Why There Is No Live Demo

Due to the nature of this project, providing a live demo is not practical. This toolkit is designed for training SDXL LoRA models, which requires specific hardware configurations and datasets that are not easily shared or demonstrated in a live environment.

## Demo Video

We have created a [demo video](https://youtu.be/1PRoAOIgnng) showcasing the training workflow and results.

Note: The video uses only 10 training steps for demonstration purposes. This is not representative of a real training run. The purpose of the video is simply to provide a visual overview of how the toolkit works.

For actual SDXL LoRA training, we recommend at least 5,000 steps as a starting point for a training.

## Verified hardware

The following setup has been **confirmed working** by the maintainer. Use it as a known-good reference point when configuring your own run:

You need **AT LEAST** the following hardware to run SDXL LoRA training with this toolkit:

| Item       | Specification                                          |
| ---------- | ------------------------------------------------------ |
| OS         | Ubuntu 24.04 LTS                                       |
| System RAM | 64 GB                                                  |
| GPU VRAM   | 12 GB                                                  |
| Storage    | NVMe SSD (recommended for latent/text-encoder caching) |
| Dataset    | ≤ 1024px resolution, ~2.5k images                      |
| Base Model | `waiIllustriousSDXL_v160.safetensors`                  |

With this configuration, training at 1024px with UNet-only LoRA (`train_text_encoder: false`), `bf16` mixed precision, and `optimizer.type: adamw8bit` runs comfortably within the 12 GB VRAM budget. If you raise `batch_size`, enable `train_text_encoder`, or train at larger effective resolutions, enable `training.gradient_checkpointing: true` and consider `dataset.cache_vae_slicing: true` to stay within VRAM limits.
