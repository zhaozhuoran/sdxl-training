import argparse

from trainer.logging import get_logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an SDXL LoRA from a YAML config.")
    parser.add_argument("config", help="Path to the training YAML file.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use tiny mock models for testing; does not train an SDXL adapter.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logger = get_logger()

    try:
        from trainer.engine.trainer import SDXLTrainer

        trainer = SDXLTrainer(args.config, is_test_mode=args.test_mode)
        trainer.run()
    except KeyboardInterrupt:
        logger.info("Training interrupted.")
        raise SystemExit(130)
    except Exception:
        logger.exception("Training failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
