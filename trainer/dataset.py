import os
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

        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1] range for latent diffusion models
        ])

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
        img_path, caption = self.samples[idx]

        # Open image safely using context manager to avoid file handle leaks
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            # If an image is corrupted, raise an error or return a fallback
            raise IOError(f"Error loading image {img_path}: {e}")

        # Apply transformations
        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "image_path": str(img_path)
        }


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    captions = [example["caption"] for example in examples]
    image_paths = [example["image_path"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "captions": captions,
        "image_paths": image_paths
    }


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
