# Training

## Install

Use Python 3.11 for the pinned dependency set. SDXL training requires an NVIDIA
CUDA GPU; CPU mode is only for tests with mock models.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` includes xformers for the pinned PyTorch version. The trainer
uses diffusers' default attention if xformers is unavailable. The 8-bit AdamW
optimizer uses bitsandbytes; without it the trainer falls back to PyTorch AdamW,
which uses more memory.

## Dataset

Put images and UTF-8 captions in one directory:

```text
dataset/
  00001.png
  00001.txt
  00002.webp
  00002.txt
```

The loader reads PNG, JPEG, and WebP files directly inside this folder. It does
not search subdirectories. A missing caption is treated as an empty string.
Images are resized and center-cropped into aspect-ratio buckets.

## Configure and run

```bash
cp configs/examples/lora_example.yaml my_config.yaml
```

Set `model.pretrained_model_name_or_path`, `dataset.path`, and
`output.experiment_name` before running:

```bash
python train.py my_config.yaml
# Equivalent after installation:
sdxl-train my_config.yaml
```

The model can be a diffusers directory, a Hugging Face model ID, or a local
`.safetensors` / `.ckpt` file. Single-file loading uses `local_files_only=True`,
so its supporting model configuration and tokenizers must be available locally.

The example trains only the UNet adapters at 1024 resolution, with batch size 1,
rank 64, alpha 32, and `bf16` precision. These are example settings, not a memory
or output-quality guarantee. Set `training.mixed_precision` to `fp16` or `no`
if your hardware does not support `bf16`.

## Memory and caching

- `dataset.cache_latents` caches VAE outputs.
- `dataset.cache_text_encoder_outputs` caches text embeddings when the text
  encoders are frozen. Disable it for whole-caption dropout.
- `dataset.cache_destination` selects RAM or disk reads. With DataLoader workers
  enabled, the trainer uses disk reads to avoid copying the RAM cache into each
  worker. Both modes write a disk cache for reuse on later runs.
- `dataset.cache_dir` overrides the default sibling `<dataset>.cache_latents`
  directory.

If caching runs out of GPU memory, lower `dataset.cache_batch_size` or enable
`dataset.cache_vae_slicing`. If the training loop runs out of memory, keep batch
size at 1, enable `training.gradient_checkpointing`, or leave
`training.train_text_encoder` disabled.

`network.unet_targets` and the text encoder target lists match substrings of
module names. They do not accept regular expressions. Leave them `null` to use
the built-in targets.

## Outputs and recovery

Each run is stored under `output.directory / output.experiment_name`:

```text
config.yaml
logs/train.log
checkpoints/
  latest/
  recovery/recovery-000100.pt
  snapshots/snapshot-000500.pt
  lora/step-001000.safetensors
```

Recovery files are rotated according to `checkpoint.keep_last_recovery`;
snapshots are retained. Both contain the adapter weights and training state.
The final `.safetensors` file contains the adapter for inference.

The example sets `resume.mode: auto`: it tries recovery checkpoints from newest
to oldest, then snapshots if none of the recovery files are valid. Set the mode
to `none` to start from scratch, preferably with a new experiment name. For a
specific checkpoint, use `manual` and set `resume.manual_path` to its `.pt` file.
The configuration default, when the resume section is omitted, is `none`.

## CPU smoke test

```bash
python -m pip install pytest
python -m pytest -q tests/test_trainer_integration.py
```

These tests create temporary images and use tiny mock networks. The CLI also
has `--test-mode` for this purpose; its exports are test artifacts, not trained
SDXL adapters.
