import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageCaptionDataset(Dataset):
    """
    A lightweight, extensible PyTorch Dataset that loads image-caption pairs from a given directory.
    Supports png, jpg, jpeg, webp.
    """
    def __init__(self, directory_path: str, resolution: int = 1024):
        self.directory_path = Path(directory_path)
        self.resolution = resolution

        if not self.directory_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory_path}")

        self.supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        self.samples = self._load_samples()

        # Pre-compute SHA256 hashes of the image paths to avoid CPU overhead during training
        self.samples_with_hashes = []
        for img_path, caption in self.samples:
            path_hash = hashlib.sha256(str(img_path).encode("utf-8")).hexdigest()
            self.samples_with_hashes.append((img_path, caption, path_hash))

        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1] range for latent diffusion models
        ])

        # Caching support fields
        self.ram_cache = {}
        self.cache_destination = None
        self.cache_dir_path = None
        self.cache_latents_enabled = False
        self.cache_te_enabled = False

    def _load_samples(self) -> List[Tuple[Path, str]]:
        samples = []
        # Support recursive walk or flat lookup. Flat lookup is standard.
        for path in self.directory_path.iterdir():
            if path.is_file() and path.suffix.lower() in self.supported_extensions:
                # Corresponding .txt file
                txt_path = path.with_suffix(".txt")
                caption = ""
                if txt_path.exists():
                    try:
                        caption = txt_path.read_text(encoding="utf-8").strip()
                    except Exception as e:
                        # Fallback to empty string but log error if needed
                        pass

                samples.append((path, caption))

        # Sort samples to ensure determinism
        samples.sort(key=lambda x: x[0].name)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, caption, path_hash = self.samples_with_hashes[idx]

        # Check if we have pre-computed items in RAM cache
        item = {}
        if self.cache_destination == "ram" and path_hash in self.ram_cache:
            ram_item = self.ram_cache[path_hash]
            if "latents" in ram_item:
                item["latents"] = ram_item["latents"]
            if "prompt_embeds" in ram_item:
                item["prompt_embeds"] = ram_item["prompt_embeds"]
                item["pooled_prompt_embeds"] = ram_item["pooled_prompt_embeds"]

        # Check if we should load from Disk cache
        elif self.cache_destination == "disk" and self.cache_dir_path:
            if self.cache_latents_enabled:
                disk_latent_path = self.cache_dir_path / f"latent_{path_hash}.pt"
                if disk_latent_path.exists():
                    item["latents"] = torch.load(disk_latent_path, map_location="cpu")
            if self.cache_te_enabled:
                disk_te_path = self.cache_dir_path / f"te_{path_hash}.pt"
                if disk_te_path.exists():
                    te_data = torch.load(disk_te_path, map_location="cpu")
                    item["prompt_embeds"] = te_data["prompt_embeds"]
                    item["pooled_prompt_embeds"] = te_data["pooled_prompt_embeds"]

        pixel_values = None
        if "latents" not in item:
            # Open image safely using context manager to avoid file handle leaks
            try:
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
            except Exception as e:
                # If an image is corrupted, raise an error or return a fallback
                raise IOError(f"Error loading image {img_path}: {e}")

            # Apply transformations
            pixel_values = self.transform(image)

        result = {
            "caption": caption,
            "image_path": str(img_path)
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if "latents" in item:
            result["latents"] = item["latents"]
        if "prompt_embeds" in item:
            result["prompt_embeds"] = item["prompt_embeds"]
            result["pooled_prompt_embeds"] = item["pooled_prompt_embeds"]

        return result


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}

    # Check if we have pixel_values
    if "pixel_values" in examples[0]:
        result["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])

    # Check if we have cached latents
    if "latents" in examples[0]:
        result["latents"] = torch.stack([example["latents"].squeeze(0) for example in examples])

    # Check if we have cached text encoder outputs
    if "prompt_embeds" in examples[0]:
        result["prompt_embeds"] = torch.stack([example["prompt_embeds"].squeeze(0) for example in examples])
        result["pooled_prompt_embeds"] = torch.stack([example["pooled_prompt_embeds"].squeeze(0) for example in examples])

    result["captions"] = [example["caption"] for example in examples]
    result["image_paths"] = [example["image_path"] for example in examples]

    return result


def create_dataloader(
    directory_path: str,
    batch_size: int,
    resolution: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Helper function to construct and return a standard DataLoader."""
    dataset = ImageCaptionDataset(directory_path, resolution=resolution)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
