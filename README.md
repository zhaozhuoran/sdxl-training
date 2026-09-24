# sdxl-training

A lightweight, configurable toolkit for training SDXL models.

## Features

- YAML-based configuration
- LoRA training
- Extensible training methods
- Checkpoint management
- Dataset and logging utilities

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .

cp configs/examples/lora_example.yaml my_config.yaml
python train.py my_config.yaml
```

See [`USAGE.md`](./USAGE.md) for configuration and training details.

## Structure

```text
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

## License

[AGPL-3.0](./LICENSE)