# Diffusion From Scratch

A text-to-image diffusion transformer trained with dual-stream joint attention, 2D RoPE, adaLN-Zero, and rectified flow.

![denoising](assets/landscape.gif)

## Architecture

- **Dual-stream transformer** with joint attention between text and image tokens
- **Rectified flow** — linear interpolation between data and noise, ~28 sampling steps
- **2D RoPE** for spatial position encoding on image patches
- **adaLN-Zero** modulation (SiLU → Linear → shift/scale/gate)
- **RMSNorm** for QK normalization
- **GELU-approximate** feedforward
- **Classifier-free guidance (CFG)** at sampling time
- **EMA** weights for stable generation

Default config: 12 layers, 12 heads, 64 head dim → 768 hidden dim, **257M parameters**

## Setup

```bash
pip install -r requirements.txt
```

## Prepare Dataset

```bash
python prepare_dataset.py --num_samples 200000
```

## Train

```bash
python train.py
```

```bash
# resume from checkpoint
python train.py --resume checkpoints/best.pt

# custom settings
python train.py --batch_size 64 --epochs 300 --data_dir data/text2img
```

## Sample

```bash
python sample.py --prompt "a sunset over mountains"
```
