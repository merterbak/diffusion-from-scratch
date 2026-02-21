# Diffusion From Scratch

A ~258M parameter text-to-image model that generates 256x256 images from text prompts using rectified flow and a diffusion transformer, built from scratch in PyTorch. Trained on 200k image-text pairs.

![denoising](assets/landscape.gif)

## Overview

The VAE compresses a 256x256 image down to a 4x32x32 latent. Noise gets mixed in, and the model learns to predict the velocity that takes it back to the clean image. At generation time it starts from pure noise and walks back to a clean image over 28 Euler steps.

```
"a sunset over mountains" --> CLIP --> [77 text tokens]
                                            |
random noise --> [256 image tokens] --> 12x DualStreamBlock --> denoise --> VAE decode --> image
                                            |
                              timestep --> adaLN-Zero conditioning
```

## Architecture

Archtecture is inspired from [Qwen-Image](https://github.com/QwenLM/Qwen-Image).

### Dual-Stream Blocks

Text and image tokens live in separate streams but attend to each other. Both streams get modulated by the timestep through adaLN-Zero which produces shift, scale, and gate values for each one. Q, K, and V are computed separately per stream and normalized with RMSNorm, then concatenated into a single sequence of 333 tokens (77 text + 256 image) for joint attention. After that the streams split back and each one passes through its own feedforward network. All gates start at zero so the model begins as an identity function, which helps keep training stable early on.

### MSRoPE

Standard RoPE only handles 1D sequences, but images are 2D grids with text tokens mixed in. MSRoPE from the Qwen-Image paper handles this by placing image tokens on a 2D grid centered at the origin. A 16x16 grid spans positions -8 to +7 on both height and width, and the head dimension gets split in half between the two axes. Text tokens sit on the diagonal starting just outside the image region, so token 0 lands at (8,8), token 1 at (9,9), and so on. Sharing the same position on both axes makes text encoding effectively 1D. Centering at (0,0) also means the model can generalize to higher resolutions at inference since the center positions stay familiar.

```
         width -->
    -8  -7  ...  +7
-8  [img] [img] [img]
-7  [img] [img] [img]      <-- image: centered 2D grid
...
+7  [img] [img] [img]
                    (8,8)   txt_0
                    (9,9)   txt_1   <-- text: diagonal
                   (10,10)  txt_2
```

### Rectified Flow

Instead of a complex noise schedule, rectified flow just draws a straight line between data and noise: `z_t = (1-t) * x_0 + t * noise`. The model predicts the velocity `v = noise - x_0` and the loss is simply MSE between predicted and target velocity. Sampling walks 28 Euler steps from t=1 (pure noise) down to t=0 (clean image). Timesteps during training come from a logit-normal distribution (`t = sigmoid(randn())`), which focuses more of the training on the harder intermediate steps rather than the trivial endpoints.

### Classifier-Free Guidance

During training, 10% of text embeddings get randomly replaced with a null embedding so the model learns both conditional and unconditional generation. At inference each denoising step runs two forward passes, one with the prompt and one without, and then extrapolates the difference:

```
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

A higher cfg_scale pushes the output closer to the prompt but reduces diversity. The default is 4.0.

## Config

Qwen-Image runs at 20B parameters with 60 layers, a 7B vision-language model for text encoding, and a custom 16-channel VAE. This project inspired from its architecture at a scale that lower computing requirements.

| | Qwen-Image | This project |
|---|---|---|
| **Parameters** | 20B | **~258M** |
| **Layers** | 60 | 12 |
| **Heads / Head dim** | 24 / 128 | 12 / 64 |
| **Hidden dim** | 3072 | 768 |
| **Text encoder** | Qwen2.5-VL (7B) | CLIP ViT-L/14 |
| **VAE** | 16-ch Wan-2.1 | 4-ch SD VAE |
| **RoPE** | 3-axis (frame, h, w) | 2-axis (h, w) |
| **Resolution** | up to 1328px | 256px |

## Get started

```bash
pip install -r requirements.txt
```

### 1. Prepare the dataset

Downloads images from [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) and encodes them into VAE latents and CLIP embeddings:

```bash
cd src
python prepare_dataset.py --num_samples 200000
```

### 2. Train

```bash
python train.py --data_dir data/text2img --batch_size 128 --epochs 200
```

```bash
# resume from checkpoint
python train.py --resume checkpoints/best.pt --epochs 300
```

### 3. Generate

```bash
python sample.py --prompt "a sunset over mountains"
python sample.py --prompt "a red sports car" --cfg 6.0 --steps 50 --num 4
```

