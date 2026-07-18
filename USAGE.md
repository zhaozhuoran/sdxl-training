# USAGE

A short, end-to-end guide to training an SDXL LoRA with this toolkit.

> Prerequisites: a CUDA-capable NVIDIA GPU, Python 3.11, and a local SDXL base model (e.g. a `.safetensors` checkpoint).

---

## 1. Environment setup

It is strongly recommended to use a dedicated virtual environment.

```bash
conda create -n sdxl-training python==3.11.13
conda activate sdxl-training
```

All following commands must run inside this activated environment.

## 2. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Optional, for lower UNet attention memory (must match your torch build):

```bash
pip install .[xformers]
```

Without it, the trainer automatically falls back to SDPA attention.

## 3. Prepare your dataset

Use the image-caption directory format. Every image needs a matching UTF-8 `.txt` caption with the same base filename:

```
dataset/
    00001.png
    00001.txt
    00002.webp
    00002.txt
```

- Supported image formats: `png`, `jpg/jpeg`, `webp`
- One caption line per `.txt` file (Kohya-style comma-separated tags work well).

## 4. Configure your run

Copy the example config and edit it for your setup:

```bash
cp configs/examples/lora_example.yaml my_config.yaml
```

Key fields to change:

| Field                                 | What to set                                                          |
| ------------------------------------- | -------------------------------------------------------------------- |
| `model.pretrained_model_name_or_path` | Path to your base model, e.g. `waiIllustriousSDXL_v160.safetensors`  |
| `model.vae_path`                      | `null` to use the base model's VAE, or a custom VAE path             |
| `dataset.path`                        | Path to your image-caption dataset folder                            |
| `dataset.resolution`                  | `1024` (standard SDXL)                                               |
| `dataset.batch_size`                  | `1` on 12 GB VRAM; raise only if you have headroom                   |
| `training.mixed_precision`            | `bf16` (recommended) or `fp16`                                       |
| `training.train_text_encoder`         | `false` (UNet-only) to save VRAM; `true` only if you have spare VRAM |
| `training.gradient_checkpointing`     | `true` if you hit OOM or raise batch size / train text encoders      |
| `network.rank` / `network.alpha`      | `64` / `32` is a good default                                        |
| `optimizer.type`                      | `adamw8bit` (recommended, low VRAM)                                  |
| `output.experiment_name`              | A name for this run's output folder                                  |

For 12 GB VRAM machines (e.g. RTX 3060 / 4070), the bundled `lora_example.yaml` defaults already target this budget: UNet-only LoRA, `bf16`, `adamw8bit`, `batch_size: 1`.

## 5. Run training

```bash
python train.py my_config.yaml
```

To resume from the latest valid checkpoint after an interruption, leave `resume.mode: "auto"` (the default). To start fresh, set it to `"none"`.

## 6. Check progress and outputs

- Checkpoints and LoRA exports are written under `output.directory / output.experiment_name/` (default `outputs/test_lora_run/`).
- Rolling recovery checkpoints (`save_every_steps`) and preserved snapshots (`snapshot_every_steps`) are created per your config.
- The final LoRA is exported as a `.safetensors` file in Kohya format with standard `__metadata__`, compatible with AUTOMATIC1111, ComfyUI, and Forge.

---

## Troubleshooting

- **Out of memory (OOM):** enable `training.gradient_checkpointing: true`, set `dataset.batch_size: 1`, enable `dataset.cache_vae_slicing: true`, or set `training.train_text_encoder: false`.
- **Slow data loading on Windows:** set `dataset.num_workers: 0` for debugging.
- **No CUDA GPU detected:** the trainer fails gracefully with a clear error. Training requires an NVIDIA CUDA GPU.
