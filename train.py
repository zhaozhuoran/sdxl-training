#!/usr/bin/env python3
import sys
import argparse
from trainer.engine.trainer import SDXLTrainer
from trainer.logging import get_logger

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="sdxl-training: A modern, lightweight, configurable SDXL training toolkit."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the training YAML configuration file."
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Launches training in test fallback mode utilizing tiny mock models on CPU."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing sdxl-training framework...")

    try:
        # Initialize the high-performance training engine
        trainer = SDXLTrainer(args.config, is_test_mode=args.test_mode)
        # Execute the main custom training loop
        trainer.run()
        logger.info("Training completed successfully! Exiting.")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Exiting immediately.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training run failed with a critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
