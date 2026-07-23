#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from trainer.benchmark import BenchmarkSuite
from trainer.logging import get_logger

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="sdxl-training: Run VRAM and performance benchmark suite for SDXL training configurations."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base training YAML configuration file."
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        required=True,
        help="Path to the benchmark configurations YAML/JSON file defining trials."
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="benchmark_state.json",
        help="Path to save/load the benchmark state for pause and resume support."
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Launches benchmarking trials in test fallback mode utilizing tiny mock models on CPU."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing SDXL training benchmark runner...")

    if not os.path.exists(args.config):
        logger.error(f"Base config file does not exist: {args.config}")
        sys.exit(1)

    bench_cfg_path = args.benchmark_config
    if not os.path.exists(bench_cfg_path):
        logger.error(f"Benchmark config file does not exist: {bench_cfg_path}")
        sys.exit(1)
    with open(bench_cfg_path, "r", encoding="utf-8") as f:
        try:
            if bench_cfg_path.endswith(".yaml") or bench_cfg_path.endswith(".yml"):
                bench_data = yaml.safe_load(f)
            else:
                import json
                bench_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse benchmark config: {e}")
            sys.exit(1)

    trials = bench_data if isinstance(bench_data, list) else bench_data.get("trials", [])
    if not trials:
        logger.error("No trials found in benchmark configuration file.")
        sys.exit(1)

    suite = BenchmarkSuite(
        base_config_path=args.config,
        trials_configs=trials,
        state_file_path=args.state_file,
        is_test_mode=args.test_mode
    )

    try:
        results = suite.run()
        logger.info("Benchmark suite execution run completed!")
        for r in results:
            logger.info(f"Trial: {r['name']} | Status: {r['status']} | Metrics: {r['metrics']} | Error: {r['error']}")
    except KeyboardInterrupt:
        logger.info("Benchmark run paused/interrupted by user. Progress saved.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Benchmark suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
