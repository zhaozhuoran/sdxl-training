import os
import json
import struct
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from safetensors.torch import save_file

from trainer.bucketing import (
    make_buckets,
    select_bucket,
    compute_bucket_assignment,
    BucketBatchSampler,
)
from trainer.caption import CaptionProcessor


def _read_safetensors_bytes(path: Path) -> Dict[str, torch.Tensor]:
    """Load a safetensors file into torch tensors WITHOUT memory-mapping.

    ``safetensors.load_file`` / ``safe_open`` memory-map the file by default.
    mmap hangs or raises on several filesystems (network / 9P / exFAT mounts,
    some USB drives), which silently broke dataset caching on those setups.
    Reading the bytes once and parsing the header + raw tensor buffers in RAM
    sidesteps mmap entirely and works everywhere. BF16 is stored as raw uint16
    and is reinterpreted via ``torch.frombuffer`` (no copy).
    """
    _SAFE_DTYPES = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    out: Dict[str, torch.Tensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = _SAFE_DTYPES[meta["dtype"]]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        buf = bytearray(data[start:end])
        out[name] = torch.frombuffer(buf, dtype=dtype).reshape(shape)
    return out


def _load_cache_file(path: Path) -> Dict[str, Any]:
    """Load a dataset cache file, supporting both the new safetensors format and
    legacy pickle (.pt) files for backwards compatibility.

    Returns a dict with the canonical keys (latents / prompt_embeds /
    pooled_prompt_embeds / original_size / crop_ltrb / bucket_size) so callers
    behave identically regardless of the on-disk format. Corrupt or unreadable
    files fall back to an empty dict (treated as a cache miss).
    """
    try:
        st = _read_safetensors_bytes(path)
        return {
            "format": "sdxl-trainer-cache",
            "version": 1,
            "latents": st.get("latents"),
            "prompt_embeds": st.get("prompt_embeds"),
            "pooled_prompt_embeds": st.get("pooled_prompt_embeds"),
            "original_size": tuple(int(v) for v in st["original_size"].tolist()),
            "crop_ltrb": tuple(int(v) for v in st["crop_ltrb"].tolist()),
            "bucket_size": tuple(int(v) for v in st["bucket_size"].tolist()),
        }
    except Exception:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return {}


def _save_cache_file(payload: Dict[str, Any], path: Path) -> None:
    """Persist a dataset cache payload using the safetensors format (faster and
    safer than pickle). Tensor outputs are stored as-is; scalar SDXL
    conditioning metadata is stored as int32 tensors, and format/version as the
    safetensors header metadata.
    """
    st: Dict[str, torch.Tensor] = {}
    if payload.get("latents") is not None:
        st["latents"] = payload["latents"].contiguous()
    if payload.get("prompt_embeds") is not None:
        st["prompt_embeds"] = payload["prompt_embeds"].contiguous()
    if payload.get("pooled_prompt_embeds") is not None:
        st["pooled_prompt_embeds"] = payload["pooled_prompt_embeds"].contiguous()
    st["original_size"] = torch.tensor(payload["original_size"], dtype=torch.int32)
    st["crop_ltrb"] = torch.tensor(payload["crop_ltrb"], dtype=torch.int32)
    st["bucket_size"] = torch.tensor(payload["bucket_size"], dtype=torch.int32)
    save_file(st, str(path), metadata={"format": "sdxl-trainer-cache", "version": "1"})


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
                cached = _load_cache_file(disk_path)
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
    seed: Optional[int] = None,
    pin_memory: bool = False,
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
    batch_sampler = BucketBatchSampler(
        dataset.bucket_of_index, batch_size, shuffle=shuffle, seed=seed
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        # drop_last is handled implicitly: bucket groups may yield partial last batches
    )
