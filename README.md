# sdxl-training

Train SDXL LoRA adapters from a folder of images and captions. The trainer uses
PyTorch and diffusers, with a YAML file for each run. It supports UNet-only or
UNet + text encoder training, aspect-ratio buckets, cached latents and text
embeddings, and checkpoint recovery.

## Run it

You need Python 3.10 or newer and an NVIDIA GPU with CUDA for SDXL training.
The pinned dependencies in `requirements.txt` target Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
cp configs/examples/lora_example.yaml my_config.yaml
```

Before starting, edit these fields in `my_config.yaml`:

- `model.pretrained_model_name_or_path`: your SDXL model path or Hugging Face ID.
- `dataset.path`: your folder of images and matching `.txt` captions.
- `output.experiment_name`: the name of this run.

```bash
python train.py my_config.yaml
```

The final adapter is saved to
`outputs/<experiment_name>/checkpoints/lora/step-<step>.safetensors` in Kohya
format. Recovery checkpoints also include the optimizer and scheduler state.

See [USAGE.md](USAGE.md) for dataset layout, memory settings, and resuming runs.
The [demo notes](DEMO.md) link to the existing recording and describe its limits.

## Working on the trainer

The training loop is in `trainer/engine/trainer.py`; adapter injection and export
are in `trainer/methods/lora/`. Dataset loading, bucketing, caption processing,
and checkpoint handling live in separate modules under `trainer/`.

```bash
python -m pip install pytest
python -m pytest -q
```

The tests use small mock models and temporary datasets. They can run without a
GPU or an SDXL download, but do not establish training quality on real models.

## License

[AGPL-3.0](LICENSE)
