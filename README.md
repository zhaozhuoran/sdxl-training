# sdxl-training

A lightweight, highly-configurable, and extensible toolkit for SDXL training.

This repository implements its own high-performance training framework independently of existing training libraries, providing a clean, modern, and robust developer experience for the SDXL ecosystem.

---

## Key Features

- **Ecosystem Compatibility:** Generates LoRA weights directly in the industry-standard `.safetensors` Kohya format, fully compatible with AUTOMATIC1111, ComfyUI, Forge, and other third-party tools.
- **GPU-First Architecture:** Tailored for robust production environments; automatically fails gracefully with clear error logging if a CUDA-compatible GPU is unavailable.
- **Advanced Parameter-Efficient Training (LoRA):** Supports target-injection for UNet, Text Encoder 1, and Text Encoder 2, while providing advanced configuration for custom layer targets.
- **Modern Hierarchical Configurations:** Powered by strongly validated YAML configurations with Pydantic schemas.
- **Robust Recovery & Checkpoint Fallbacks:** Features dual-checkpoint policies (rolling time/step recovery checkpoints and preserved snapshot checkpoints) with automatic deserialization verification and corrupted-checkpoint recovery fallbacks.

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

### Environment Setup

It is strongly recommended to run this project inside a **dedicated virtual environment** so that the pinned dependency set does not conflict with other projects. Using Conda:

```bash
conda create -n sdxl-training python==3.11.13
conda activate sdxl-training
```

All following installation steps must be executed inside this activated `sdxl-training` environment.

### Installation

1. Install the pinned, reproducible dependency set. Every version is locked to a single exact release in `requirements.txt` to guarantee a stable, deterministic environment:
    ```bash
    pip install -r requirements.txt
    ```
2. Build and install the toolkit in editable mode:
    ```bash
    pip install -e .
    ```

### Dataset Structure

The MVP supports the common image-caption directory format. Each training image should have a corresponding UTF-8 encoded `.txt` caption file with the same filename.

```
dataset/
    00001.png
    00001.txt
    00002.webp
    00002.txt
```

### Running Training

To run training with a structured config file:

```bash
python train.py configs/examples/lora_example.yaml
```

To run verification unit or integration tests on CPU (e.g., in testing environments without an active GPU):

```bash
python train.py configs/examples/lora_example.yaml --test-mode
```

---

## Configuration Reference

The training run is controlled entirely by the modular YAML file. An example with explanations of keys can be found in `configs/examples/lora_example.yaml`.

---

## Development & Testing

Unit and integration tests are located in the `tests/` directory and can be executed via `pytest`:

```bash
python3 -m pytest
```

The non-training components (configuration parsing, validation, dataset pipeline, and checkpoint validations) are built to remain fully testable on CPU.

---

## License

This project is licensed under the GPU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.
