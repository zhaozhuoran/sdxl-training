"""Aspect-ratio bucketing (SDXL LoRA training).

Our own, self-contained implementation inspired by the bucket concept used in
kohya-ss/sd-scripts, but with a simpler/cleaner API tailored to this trainer.

Key ideas
---------
* A set of ``(width, height)`` buckets is generated around a base resolution.
  Every bucket has both dimensions as multiples of ``bucket_step`` (8-safe for
  the VAE which downscales by 8).
* Each image is assigned to the bucket whose aspect ratio is closest to the
  image's own aspect ratio. The image is resized (preserving aspect ratio,
  scaling to *cover* the bucket) and then center-cropped to the exact bucket
  size.
* The size the image had *before* the center crop (``original_size``) and the
  crop offsets (``crop_ltrb``) are recorded. SDXL micro-conditioning
  (``add_time_ids``) needs the real original size and crop, not a fake constant.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Sampler


@dataclass
class BucketAssignment:
    bucket: Tuple[int, int]          # (width, height) exact target size
    original_size: Tuple[int, int]   # (width, height) after resize-to-cover, before crop (used for pixel transform)
    crop_ltrb: Tuple[int, int, int, int]  # (left, top, right, bottom) crop within the resized image (pixel transform)
    true_original_size: Tuple[int, int]   # (width, height) actual image size (SDXL conditioning)
    crop_original: Tuple[int, int]         # (top, left) crop offsets in ORIGINAL image space (SDXL conditioning)


def make_buckets(
    base_resolution: int,
    bucket_step: int = 64,
    min_size: int = None,
    max_size: int = None,
) -> List[Tuple[int, int]]:
    """Build the list of (width, height) buckets around ``base_resolution``.

    Buckets whose area is within a sensible band around ``base_resolution**2`` are
    kept. Both dimensions are multiples of ``bucket_step``. The canonical square
    bucket ``(base_resolution, base_resolution)`` is always included.
    """
    if bucket_step < 1:
        bucket_step = 1
    if min_size is None:
        min_size = bucket_step
    else:
        min_size = max(bucket_step, (min_size // bucket_step) * bucket_step)
    if max_size is None:
        max_size = int(base_resolution * 1.5)
    max_size = max(max_size, bucket_step)
    # Snap max_size to a step multiple
    max_size = (max_size // bucket_step) * bucket_step

    base_area = base_resolution * base_resolution
    lo_area = base_area * 0.4
    hi_area = base_area * 1.6

    buckets: List[Tuple[int, int]] = []
    seen = set()
    for h in range(min_size, max_size + 1, bucket_step):
        for w in range(min_size, max_size + 1, bucket_step):
            area = w * h
            if area < lo_area or area > hi_area:
                continue
            key = (w, h)
            if key in seen:
                continue
            seen.add(key)
            buckets.append(key)

    base = (base_resolution, base_resolution)
    if base not in seen:
        buckets.append(base)

    # Sort by aspect ratio (h/w) then by area for deterministic ordering.
    buckets.sort(key=lambda b: (b[1] / b[0], b[0] * b[1]))
    return buckets


def select_bucket(image_size: Tuple[int, int], buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Pick the bucket whose aspect ratio is closest to the image's.

    Prefers a bucket whose area covers the image (so we downscale, preserving
    quality). If no bucket is large enough to cover the image, falls back to the
    largest bucket with the closest aspect ratio (upscaling).
    """
    iw, ih = image_size
    aspect = ih / iw  # height / width
    image_area = iw * ih

    covering = [b for b in buckets if (b[0] * b[1]) >= image_area]
    if covering:
        # Closest aspect, then smallest area (avoid unnecessary upscaling).
        return min(covering, key=lambda b: (abs((b[1] / b[0]) - aspect), b[0] * b[1]))
    # No bucket covers the image: closest aspect, then largest area.
    return min(buckets, key=lambda b: (abs((b[1] / b[0]) - aspect), -(b[0] * b[1])))


def compute_bucket_assignment(
    image_size: Tuple[int, int], bucket: Tuple[int, int]
) -> BucketAssignment:
    """Resize-to-cover then center-crop an image of ``image_size`` into ``bucket``.

    Returns the exact bucket size, the size after resize (before crop) and the
    center-crop offsets.
    """
    iw, ih = image_size
    bw, bh = bucket

    # Scale so the scaled image fully covers the bucket, then center crop.
    scale = max(bw / iw, bh / ih)
    rw = max(1, round(iw * scale))
    rh = max(1, round(ih * scale))

    # Center crop to (bw, bh)
    left = (rw - bw) // 2
    top = (rh - bh) // 2
    right = rw - bw - left
    bottom = rh - bh - top

    # Crop offsets expressed in the ORIGINAL image's coordinate space (the
    # convention SDXL / diffusers expect for add_time_ids), rather than in the
    # resized-to-cover space used for the pixel transform above.
    crop_top_orig = round(top / scale)
    crop_left_orig = round(left / scale)

    return BucketAssignment(
        bucket=(bw, bh),
        original_size=(rw, rh),
        crop_ltrb=(left, top, right, bottom),
        true_original_size=(iw, ih),
        crop_original=(crop_top_orig, crop_left_orig),
    )


class BucketBatchSampler(Sampler):
    """Yields batches of indices that share the same bucket.

    This guarantees that every tensor in a batch (pixel_values or cached
    latents) has identical spatial dimensions and can be ``torch.stack``-ed.
    """

    def __init__(
        self,
        bucket_of_index: List[Tuple[int, int]],
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.bucket_of_index = bucket_of_index
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        # Dedicated RNG so dataset ordering is reproducible given a seed.
        # A single instance is reused across epochs: the within-bucket order is
        # fixed at construction, while the batch order is reshuffled in __iter__,
        # giving varied-but-deterministic ordering per epoch.
        self._rng = random.Random(seed)

        # Group sample indices by bucket key
        self.bucket_groups = {}
        for idx, b in enumerate(bucket_of_index):
            self.bucket_groups.setdefault(b, []).append(idx)

        self.batches = []
        for b, indices in self.bucket_groups.items():
            if self.shuffle:
                self._rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i : i + self.batch_size])

    def __iter__(self):
        if self.shuffle:
            self._rng.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
