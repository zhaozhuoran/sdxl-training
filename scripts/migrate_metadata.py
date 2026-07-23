#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Optional
from safetensors import safe_open
from safetensors.torch import save_file, load_file

def migrate_file(file_path: Path, model_name: Optional[str] = None, overwrite: bool = True, output_path: Optional[Path] = None):
    """Loads a .safetensors file, standardizes its metadata, and saves it back."""
    print(f"Processing file: {file_path}")

    # 1. Extract existing metadata first to enable fail-fast before loading heavy weights
    metadata = {}
    try:
        with safe_open(str(file_path), framework="pt") as f:
            metadata = f.metadata() or {}
            # Metadata keys/values from safe_open are usually strings
            metadata = dict(metadata)
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return False

    # 2. Load weights
    try:
        weights = load_file(str(file_path))
    except Exception as e:
        print(f"Error loading weights from {file_path}: {e}")
        return False

    # 3. Modify metadata
    # - Update/set ss_base_model_version to sdxl_base_v1-0
    metadata["ss_base_model_version"] = "sdxl_base_v1-0"

    # - Update/set ss_output_name using provided model_name or fallback to filename without extension
    resolved_name = model_name if model_name else file_path.stem
    metadata["ss_output_name"] = resolved_name

    # 4. Save file atomically to prevent corruption on failure
    target_path = output_path if output_path else (file_path if overwrite else file_path.with_name(f"{file_path.stem}_migrated{file_path.suffix}"))
    temp_path = target_path.with_suffix(".tmp")

    try:
        # Save to a temporary file first
        save_file(weights, str(temp_path), metadata=metadata)
        # Atomically replace the target file
        os.replace(temp_path, target_path)
        print(f"Successfully migrated and saved to: {target_path}")
        return True
    except Exception as e:
        print(f"Error saving file {target_path}: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Migrate old/legacy safetensors model metadata to standard Kohya-compatible format."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a .safetensors file or a directory containing .safetensors files."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model name to save under 'ss_output_name'. Defaults to filename without extension if not provided."
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Save migrated file with '_migrated' suffix instead of overwriting the original file."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional specific output path for saving the migrated file (only valid when processing a single file)."
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{path}' does not exist.")
        return

    overwrite = not args.no_overwrite
    output_path = Path(args.output_path) if args.output_path else None

    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            print(f"Error: Single file path must point to a '.safetensors' file. Got: '{path}'")
            return
        migrate_file(path, model_name=args.model_name, overwrite=overwrite, output_path=output_path)
    elif path.is_dir():
        if output_path:
            print("Error: --output-path cannot be used when processing a directory.")
            return

        print(f"Scanning directory '{path}' for .safetensors files...")
        files = list(path.rglob("*.safetensors"))
        if not files:
            print("No .safetensors files found in the directory.")
            return

        print(f"Found {len(files)} files to process.")
        success_count = 0
        for f in files:
            if migrate_file(f, model_name=args.model_name, overwrite=overwrite):
                success_count += 1
        print(f"Processing complete: {success_count}/{len(files)} files successfully migrated.")

if __name__ == "__main__":
    main()
