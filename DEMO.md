# Why There Is No DEMO

> **This project does not provide — and cannot reasonably provide — a runnable DEMO.**

## Why a DEMO is not feasible

A "DEMO" for this toolkit would imply one of the following, all of which are impractical or impossible:

1. **Training is GPU-bound and slow.** This is an SDXL LoRA training toolkit, not an inference/demo app. A single meaningful LoRA training run takes thousands of optimizer steps and hours of GPU time. There is no "instant" version that demonstrates real results.
2. **No meaningful output without a real dataset + base model.** A LoRA is only useful relative to a specific base model and a specific set of training images. Shipping a canned "demo" run would produce a model that is useless to you, while still requiring the same heavy GPU/VRAM footprint.
3. **Heavy resource requirements.** Training needs a CUDA-capable GPU with sufficient VRAM, large model weights (several GB), and a dataset of images. These cannot be bundled into a portable demo nor run in a web/CI sandbox.
4. **Reproducibility, not convenience, is the priority.** The toolkit is built around reproducible, user-controlled YAML configs. The "demo" is your run, on your hardware, with your data.

**Conclusion:** The fastest, most honest path is for you to run it yourself. The instructions are short and are documented in [`USAGE.md`](./USAGE.md).

## Verified hardware

The following setup has been **confirmed working** by the maintainer. Use it as a known-good reference point when configuring your own run:

You need AT LEAST the following hardware to run SDXL LoRA training with this toolkit:

| Item       | Specification                                          |
| ---------- | ------------------------------------------------------ |
| OS         | Ubuntu 24.04 LTS                                       |
| System RAM | 64 GB                                                  |
| GPU VRAM   | 12 GB                                                  |
| Storage    | NVMe SSD (recommended for latent/text-encoder caching) |
| Dataset    | ≤ 1024px resolution, ~2.5k images                      |
| Base Model | `waiIllustriousSDXL_v160.safetensors`                  |

With this configuration, training at 1024px with UNet-only LoRA (`train_text_encoder: false`), `bf16` mixed precision, and `optimizer.type: adamw8bit` runs comfortably within the 12 GB VRAM budget. If you raise `batch_size`, enable `train_text_encoder`, or train at larger effective resolutions, enable `training.gradient_checkpointing: true` and consider `dataset.cache_vae_slicing: true` to stay within VRAM limits.

## What to do instead

1. Read [`USAGE.md`](./USAGE.md) for a step-by-step walkthrough.
2. Copy and adapt `configs/examples/lora_example.yaml` for your base model and dataset.
3. Run `python train.py your_config.yaml`.
