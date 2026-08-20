# sdxl-training

A lightweight, highly-configurable, and extensible toolkit for SDXL training.

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

## Getting Started

**[`USAGE.md`](./USAGE.md)**.

Quick start:

```bash
pip install -r requirements.txt
pip install -e .
cp configs/examples/lora_example.yaml my_config.yaml
python train.py my_config.yaml
```

## Configuration Reference

The training run is controlled entirely by the modular YAML file. An example with explanations of keys can be found in `configs/examples/lora_example.yaml`, and a complete walkthrough is in [`USAGE.md`](./USAGE.md).

## License

This project is licensed under the GNU Affero General Public License v3 (AGPLv3) - see the [LICENSE](./LICENSE) file for details.
