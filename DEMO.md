# Demo

[Existing training recording](https://youtu.be/1PRoAOIgnng)

The recording shows a 10-step run. It demonstrates starting the trainer and
writing outputs; it does not demonstrate a finished LoRA's image quality.
The recording has been reported to contain long black sections and needs a
replacement capture. The source video is not included in this repository.

## Replacement recording

Keep the terminal and configuration visible. Show the dataset layout, start the
run, and show the exported `.safetensors` file when it finishes. Cut loading
waits and blank footage, and label any time skips. If showing generated images,
include the prompt, seed, base model, and LoRA strength so the comparison can be
repeated.

Watch the exported file from start to finish before uploading it, then replace
the link above. A CPU mock run should be labeled as a software smoke test.

## Previously reported setup

The maintainer previously reported a run with Ubuntu 24.04, 64 GB RAM, 12 GB
VRAM, an NVMe SSD, and roughly 2,500 images at up to 1024px, using
`waiIllustriousSDXL_v160.safetensors`. The settings were UNet-only training,
`bf16`, and `adamw8bit`.

This is a reported configuration, not a measured minimum requirement. Memory
use depends on the model, bucket sizes, batch size, and caching settings.
