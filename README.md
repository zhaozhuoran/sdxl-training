# sdxl-training

A lightweight, highly-configurable, and extensible toolkit for SDXL training.

This repository implements its own high-performance training framework independently of existing training libraries, providing a clean, modern, and robust developer experience for the SDXL ecosystem.

---

## Key Features

- **Ecosystem Compatibility:** Generates LoRA weights directly in the industry-standard `.safetensors` Kohya format (with kohya-standard `__metadata__`: SDXL base version, network dim/alpha), fully compatible with AUTOMATIC1111, ComfyUI, Forge, and other third-party tools.
- **GPU-First Architecture:** Tailored for robust production environments; automatically fails gracefully with clear error logging if a CUDA-compatible GPU is unavailable.
- **Memory-Efficient Training:** Gradient checkpointing on the UNet, automatic offloading of the VAE / Text Encoders to CPU when their outputs are cached, and optional xFormers / SDPA memory-efficient attention keep peak VRAM low at 1024px SDXL.
- **Advanced Parameter-Efficient Training (LoRA):** Supports target-injection for UNet, Text Encoder 1, and Text Encoder 2, while providing advanced configuration for custom layer targets.
- **Correct SDXL Conditioning:** Aspect-ratio bucketing with real per-image `original_size` / crop offsets (true original image space, correct `[h, w]` time-ids ordering), plus Kohya-style caption augmentation (tag shuffle / dropout, whole-caption dropout).
- **Reproducible Training:** Caption augmentation uses a dedicated seeded RNG captured in checkpoints, so resumes reproduce the exact augmentation sequence.
- **Modern Hierarchical Configurations:** Powered by strongly validated YAML configurations with Pydantic schemas.
- **Robust Recovery & Checkpoint Fallbacks:** Features dual-checkpoint policies (rolling time/step recovery checkpoints and preserved snapshot checkpoints) with automatic deserialization verification, corrupted-checkpoint recovery, and a guard requiring explicit confirmation when resuming with a mismatched config.

---

## Directory Structure

```
sdxl-training/
├── configs/
│   └── examples/
│       └── lora_example.yaml
├── trainer/
│   ├── engine/
│   │   └── trainer.py
│   ├── methods/
│   │   └── lora/
│   │       ├── injection.py
│   │       └── exporter.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── dataset.py
│   └── logging.py
├── tests/
├── train.py
├── pyproject.toml
└── README.md
```

---

## Getting Started

Full, step-by-step setup and usage instructions (environment, installation, dataset preparation, configuration, and running training) are documented in **[`USAGE.md`](./USAGE.md)**.

Quick start:

```bash
pip install -r requirements.txt
pip install -e .
cp configs/examples/lora_example.yaml my_config.yaml
python train.py my_config.yaml
```

---

## Configuration Reference

The training run is controlled entirely by the modular YAML file. An example with explanations of keys can be found in `configs/examples/lora_example.yaml`, and a complete walkthrough is in [`USAGE.md`](./USAGE.md).

---

## Development & Testing

Unit and integration tests are located in the `tests/` directory and can be executed via `pytest`:

```bash
python3 -m pytest
```

The non-training components (configuration parsing, validation, dataset pipeline, and checkpoint validations) are built to remain fully testable on CPU.

---

## License

This project is licensed under the GNU Affero General Public License v3 (AGPLv3) - see the [LICENSE](./LICENSE) file for details.
