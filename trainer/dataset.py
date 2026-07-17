import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from trainer.bucketing import (
    make_buckets,
    select_bucket,
    compute_bucket_assignment,
    BucketBatchSampler,
)
from trainer.caption import CaptionProcessor


class ImageCaptionDataset(Dataset):
    """Image-caption dataset with aspect-ratio bucketing and SDXL conditioning metadata.

    Each image is assigned to the bucket whose aspect ratio is closest to its own.
    The image is resized (preserving aspect ratio, scaling to cover) and center-cropped
    to the exact bucket size. The pre-crop size (``original_size``) and crop offsets
    (``crop_ltrb``) are recorded so SDXL micro-conditioning (``add_time_ids``) can use
    the real values instead of a fake constant.
    """

    def __init__(
        self,
        directory_path: str,
        resolution: int = 1024,
        bucket_step: int = 64,
        bucket_min_size: Optional[int] = None,
        bucket_max_size: Optional[int] = None,
        caption_processor: Optional[CaptionProcessor] = None,
    ):
        self.directory_path = Path(directory_path)
        self.resolution = resolution

        if not self.directory_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory_path}")

        self.supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        self.samples = self._load_samples()

        # Pre-compute SHA256 hashes of the image paths to avoid CPU overhead during training
        self.samples_with_hashes: List[Tuple[Path, str, str]] = []
        for img_path, caption in self.samples:
            path_hash = hashlib.sha256(str(img_path).encode("utf-8")).hexdigest()
            self.samples_with_hashes.append((img_path, caption, path_hash))

        # Build buckets and assign each sample to its closest-aspect bucket.
        self.buckets = make_buckets(
            base_resolution=resolution,
            bucket_step=bucket_step,
            min_size=bucket_min_size,
            max_size=bucket_max_size,
        )
        self.caption_processor = caption_processor or CaptionProcessor()

        self.sample_meta: List[Dict[str, Any]] = []
        self.bucket_of_index: List[Tuple[int, int]] = []
        for img_path, _caption, _h in self.samples_with_hashes:
            try:
                with Image.open(img_path) as img:
                    iw, ih = img.size
            except Exception:
                iw, ih = resolution, resolution
            bucket = select_bucket((iw, ih), self.buckets)
            assign = compute_bucket_assignment((iw, ih), bucket)
            self.sample_meta.append({
                "bucket": assign.bucket,
                "original_size": assign.original_size,
                "crop_ltrb": assign.crop_ltrb,
                "true_original_size": assign.true_original_size,
                "crop_original": assign.crop_original,
            })
            self.bucket_of_index.append(assign.bucket)

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1] for latent diffusion models
        ])

        # Caching support fields
        self.ram_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_destination: Optional[str] = None
        self.cache_dir_path: Optional[Path] = None
        self.cache_latents_enabled = False
        self.cache_te_enabled = False

    def _load_samples(self) -> List[Tuple[Path, str]]:
        samples = []
        for path in self.directory_path.iterdir():
            if path.is_file() and path.suffix.lower() in self.supported_extensions:
                txt_path = path.with_suffix(".txt")
                caption = ""
                if txt_path.exists():
                    try:
                        caption = txt_path.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass
                samples.append((path, caption))
        samples.sort(key=lambda x: x[0].name)
        return samples

    def _transform(self, image: Image.Image, meta: Dict[str, Any]) -> torch.Tensor:
        bw, bh = meta["bucket"]
        rw, rh = meta["original_size"]
        left, top, _right, _bottom = meta["crop_ltrb"]
        image = image.resize((rw, rh), Image.BILINEAR)
        image = image.crop((left, top, left + bw, top + bh))
        return self.base_transform(image)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, caption, path_hash = self.samples_with_hashes[idx]
        meta = self.sample_meta[idx]
        bucket = meta["bucket"]
        original_size = meta["original_size"]
        crop_ltrb = meta["crop_ltrb"]
        true_original_size = meta["true_original_size"]
        crop_original = meta["crop_original"]

        item: Dict[str, Any] = {
            "caption": caption,
            "image_path": str(img_path),
            "original_size": original_size,
            "crop_ltrb": crop_ltrb,
            "true_original_size": true_original_size,
            "crop_original": crop_original,
            "bucket_size": bucket,
        }

        # Check if we have pre-computed items in RAM cache
        if self.cache_destination == "ram" and path_hash in self.ram_cache:
            ram_item = self.ram_cache[path_hash]
            if "latents" in ram_item:
                item["latents"] = ram_item["latents"]
            if "prompt_embeds" in ram_item:
                item["prompt_embeds"] = ram_item["prompt_embeds"]
                item["pooled_prompt_embeds"] = ram_item["pooled_prompt_embeds"]

        # Check if we should load from Disk cache
        elif self.cache_destination == "disk" and self.cache_dir_path is not None:
            disk_path = self.cache_dir_path / f"cache_{path_hash}.pt"
            if disk_path.exists():
                cached = torch.load(disk_path, map_location="cpu")
                if self.cache_latents_enabled and cached.get("latents") is not None:
                    item["latents"] = cached["latents"]
                if self.cache_te_enabled and cached.get("prompt_embeds") is not None:
                    item["prompt_embeds"] = cached["prompt_embeds"]
                    item["pooled_prompt_embeds"] = cached["pooled_prompt_embeds"]

        # Decode the image only if we still need pixel values (no cached latents)
        if "latents" not in item:
            try:
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
            except Exception as e:
                raise IOError(f"Error loading image {img_path}: {e}")
            item["pixel_values"] = self._transform(image, meta)

        return item


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    # Each optional tensor key must be present in EVERY sample of the batch or in
    # NONE. BucketBatchSampler groups by bucket, so spatial sizes match, but cache
    # status is per-sample; a mixed batch (some cached, some not) is unsupported
    # and must fail loudly instead of raising a cryptic KeyError during stacking.
    tensor_keys = {
        "pixel_values": False,
        "latents": True,
        "prompt_embeds": True,
        "pooled_prompt_embeds": True,
    }
    for key, squeeze in tensor_keys.items():
        present = [ex for ex in examples if key in ex]
        if not present:
            continue
        if len(present) != len(examples):
            raise ValueError(
                f"Collate inconsistency: '{key}' present in {len(present)}/{len(examples)} "
                f"samples of a batch. Mixed cached/uncached samples in one batch are not supported; "
                f"ensure the dataset is fully cached before training."
            )
        if squeeze:
            result[key] = torch.stack([ex[key].squeeze(0) for ex in examples])
        else:
            result[key] = torch.stack([ex[key] for ex in examples])

    result["captions"] = [example["caption"] for example in examples]
    result["image_paths"] = [example["image_path"] for example in examples]
    result["original_sizes"] = [example["original_size"] for example in examples]
    result["crop_ltrbs"] = [example["crop_ltrb"] for example in examples]
    result["true_original_sizes"] = [example["true_original_size"] for example in examples]
    result["crop_originals"] = [example["crop_original"] for example in examples]
    result["bucket_sizes"] = [example["bucket_size"] for example in examples]

    return result


def create_dataloader(
    directory_path: str,
    batch_size: int,
    resolution: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    bucket_step: int = 64,
    bucket_min_size: Optional[int] = None,
    bucket_max_size: Optional[int] = None,
    caption_processor: Optional[CaptionProcessor] = None,
) -> DataLoader:
    """Constructs a DataLoader that batches samples by aspect-ratio bucket."""
    dataset = ImageCaptionDataset(
        directory_path,
        resolution=resolution,
        bucket_step=bucket_step,
        bucket_min_size=bucket_min_size,
        bucket_max_size=bucket_max_size,
        caption_processor=caption_processor,
    )
    batch_sampler = BucketBatchSampler(dataset.bucket_of_index, batch_size, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        # drop_last is handled implicitly: bucket groups may yield partial last batches
    )
