"""Caption processing for SDXL LoRA training.

Implements Kohya-style augmentations applied to comma-separated tag captions:
* ``shuffle_caption``  - randomly shuffle the comma-separated tags (keeping the
  first ``keep_tokens`` tags fixed in place, matching Kohya's ``--shuffle_caption``
  / ``--keep_tokens``).
* ``tag_dropout_rate`` - randomly drop individual tags with the given probability.
* ``caption_dropout_rate`` - randomly replace the whole caption with an empty
  string (used at training time; incompatible with caching TE outputs).

Each processor is constructed from a config subset and is deterministic-free
beyond ``random``; callers may call ``process`` repeatedly to get fresh variants.
"""

import random
from typing import Optional


class CaptionProcessor:
    def __init__(
        self,
        shuffle_caption: bool = False,
        keep_tokens: int = 0,
        tag_dropout_rate: float = 0.0,
        caption_dropout_rate: float = 0.0,
        sep: str = ",",
    ):
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = max(0, int(keep_tokens))
        self.tag_dropout_rate = min(1.0, max(0.0, float(tag_dropout_rate)))
        self.caption_dropout_rate = min(1.0, max(0.0, float(caption_dropout_rate)))
        self.sep = sep

    def maybe_drop_caption(self, caption: str, rate: Optional[float] = None) -> str:
        """Whole-caption dropout. Returns '' with probability ``rate`` (default: caption_dropout_rate)."""
        r = self.caption_dropout_rate if rate is None else rate
        if r > 0.0 and random.random() < r:
            return ""
        return caption

    def process(self, caption: str) -> str:
        """Apply tag shuffling + per-tag dropout. Returns the processed caption string."""
        caption = (caption or "").strip()
        if not caption:
            return caption

        tags = [t.strip() for t in caption.split(self.sep) if t.strip()]
        if not tags:
            return caption

        # Keep the first ``keep_tokens`` tags in place; shuffle the remainder.
        head = tags[: self.keep_tokens]
        tail = tags[self.keep_tokens :]
        if self.shuffle_caption and len(tail) > 1:
            shuffled = tail[:]
            random.shuffle(shuffled)
            tags = head + shuffled
        else:
            tags = head + tail

        # Per-tag dropout
        if self.tag_dropout_rate > 0.0:
            tags = [t for t in tags if random.random() >= self.tag_dropout_rate]

        return self.sep.join(tags)
