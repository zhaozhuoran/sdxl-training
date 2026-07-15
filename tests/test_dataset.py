import os
from pathlib import Path
from PIL import Image
import torch
from trainer.dataset import ImageCaptionDataset, collate_fn, create_dataloader

def test_image_caption_dataset(tmp_path):
    # Setup mock dataset
    img_dir = tmp_path / "dataset"
    img_dir.mkdir()

    # Create mock images
    img1 = Image.new("RGB", (800, 800), color="red")
    img1.save(img_dir / "00001.png")
    (img_dir / "00001.txt").write_text("a red square", encoding="utf-8")

    img2 = Image.new("RGB", (1200, 900), color="blue")
    img2.save(img_dir / "00002.webp")
    (img_dir / "00002.txt").write_text("a blue rectangle", encoding="utf-8")

    dataset = ImageCaptionDataset(str(img_dir), resolution=512)
    assert len(dataset) == 2

    # Sort order validation
    assert dataset.samples[0][0].name == "00001.png"
    assert dataset.samples[0][1] == "a red square"
    assert dataset.samples[1][0].name == "00002.webp"
    assert dataset.samples[1][1] == "a blue rectangle"

    # Load item validation
    item = dataset[0]
    assert "pixel_values" in item
    assert item["caption"] == "a red square"
    assert item["pixel_values"].shape == (3, 512, 512)

    # Test collation
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["pixel_values"].shape == (2, 3, 512, 512)
    assert batch["captions"] == ["a red square", "a blue rectangle"]

    # Test DataLoader helper
    dataloader = create_dataloader(str(img_dir), batch_size=2, resolution=256, shuffle=False, num_workers=0)
    batch_dl = next(iter(dataloader))
    assert batch_dl["pixel_values"].shape == (2, 3, 256, 256)
