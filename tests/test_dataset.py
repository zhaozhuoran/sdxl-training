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
    img1 = Image.new("RGB", (512, 512), color="red")
    img1.save(img_dir / "00001.png")
    (img_dir / "00001.txt").write_text("a red square", encoding="utf-8")

    img2 = Image.new("RGB", (512, 512), color="blue")
    img2.save(img_dir / "00002.webp")
    (img_dir / "00002.txt").write_text("a blue rectangle", encoding="utf-8")

    dataset = ImageCaptionDataset(str(img_dir), resolution=512)
    assert len(dataset) == 2

    # Sort order validation
    assert dataset.samples[0][0].name == "00001.png"
    assert dataset.samples[0][1] == "a red square"
    assert dataset.samples[1][0].name == "00002.webp"
    assert dataset.samples[1][1] == "a blue rectangle"

    # Load item validation (square image -> base bucket 512x512)
    item = dataset[0]
    assert "pixel_values" in item
    assert item["caption"] == "a red square"
    assert item["pixel_values"].shape == (3, 512, 512)
    assert item["bucket_size"] == (512, 512)

    # Test collation (same bucket -> stackable)
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["pixel_values"].shape == (2, 3, 512, 512)
    assert batch["captions"] == ["a red square", "a blue rectangle"]

    # Test DataLoader helper (bucket-batched)
    dataloader = create_dataloader(str(img_dir), batch_size=2, resolution=512, shuffle=False, num_workers=0)
    batch_dl = next(iter(dataloader))
    assert batch_dl["pixel_values"].shape == (2, 3, 512, 512)


def test_dataset_ram_caching(tmp_path):
    img_dir = tmp_path / "dataset"
    img_dir.mkdir()

    img1 = Image.new("RGB", (128, 128), color="red")
    img1.save(img_dir / "00001.png")
    (img_dir / "00001.txt").write_text("a red square", encoding="utf-8")

    dataset = ImageCaptionDataset(str(img_dir), resolution=128)

    # Simulate population of RAM cache (e.g. by SDXLTrainer.cache_dataset)
    import hashlib
    path_hash = hashlib.sha256(str(img_dir / "00001.png").encode("utf-8")).hexdigest()

    dataset.cache_destination = "ram"
    dataset.cache_latents_enabled = True
    dataset.cache_te_enabled = True

    mock_latent = torch.randn(1, 4, 16, 16)
    mock_prompt_embeds = torch.randn(1, 77, 2048)
    mock_pooled_prompt_embeds = torch.randn(1, 1280)

    dataset.ram_cache[path_hash] = {
        "latents": mock_latent,
        "prompt_embeds": mock_prompt_embeds,
        "pooled_prompt_embeds": mock_pooled_prompt_embeds,
    }

    # Retrieve and verify
    item = dataset[0]
    assert "pixel_values" not in item  # pixel_values not loaded since latents are cached
    assert "latents" in item
    assert "prompt_embeds" in item
    assert "pooled_prompt_embeds" in item

    assert torch.equal(item["latents"], mock_latent)
    assert torch.equal(item["prompt_embeds"], mock_prompt_embeds)
    assert torch.equal(item["pooled_prompt_embeds"], mock_pooled_prompt_embeds)

    # Verify collation
    batch = collate_fn([item])
    assert "pixel_values" not in batch
    assert batch["latents"].shape == (1, 4, 16, 16)
    assert batch["prompt_embeds"].shape == (1, 77, 2048)
    assert batch["pooled_prompt_embeds"].shape == (1, 1280)


def test_dataset_disk_caching(tmp_path):
    img_dir = tmp_path / "dataset"
    img_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    img1 = Image.new("RGB", (128, 128), color="red")
    img1.save(img_dir / "00001.png")
    (img_dir / "00001.txt").write_text("a red square", encoding="utf-8")

    dataset = ImageCaptionDataset(str(img_dir), resolution=128)

    # Save mock tensors to disk cache directory
    import hashlib
    path_hash = hashlib.sha256(str(img_dir / "00001.png").encode("utf-8")).hexdigest()

    dataset.cache_destination = "disk"
    dataset.cache_dir_path = cache_dir
    dataset.cache_latents_enabled = True
    dataset.cache_te_enabled = True

    mock_latent = torch.randn(1, 4, 16, 16)
    mock_prompt_embeds = torch.randn(1, 77, 2048)
    mock_pooled_prompt_embeds = torch.randn(1, 1280)

    # New combined cache schema: cache_{hash}.pt
    torch.save({
        "format": "sdxl-trainer-cache",
        "version": 1,
        "latents": mock_latent,
        "prompt_embeds": mock_prompt_embeds,
        "pooled_prompt_embeds": mock_pooled_prompt_embeds,
        "original_size": (128, 128),
        "crop_ltrb": (0, 0, 0, 0),
        "bucket_size": (128, 128),
    }, cache_dir / f"cache_{path_hash}.pt")

    # Retrieve and verify
    item = dataset[0]
    assert "pixel_values" not in item
    assert "latents" in item
    assert "prompt_embeds" in item
    assert torch.equal(item["latents"], mock_latent)
    assert torch.equal(item["prompt_embeds"], mock_prompt_embeds)
