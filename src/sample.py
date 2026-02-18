import argparse
import os
import torch
import torchvision.utils as vutils
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from model import Config, DiT
from diffusion import RectifiedFlow


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = Config(**ckpt["model_config"])
    model = DiT(config).to(device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    return model, config


def encode_prompt(prompt, device):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)

    tokens = tokenizer(
        prompt, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_emb = clip_model(**tokens).last_hidden_state

    null_tokens = tokenizer(
        "", padding="max_length", max_length=77, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        null_emb = clip_model(**null_tokens).last_hidden_state

    del clip_model, tokenizer
    torch.cuda.empty_cache()

    return text_emb, null_emb


def main():
    parser = argparse.ArgumentParser(description="generate images")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape")
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--output", type=str, default="samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    if not os.path.exists(args.ckpt):
        print(f"checkpoint not found: {args.ckpt}")
        return

    os.makedirs(args.output, exist_ok=True)

    model, config = load_model(args.ckpt, device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)
    flow = RectifiedFlow(num_steps=args.steps, device=device)

    text_emb, null_emb = encode_prompt(args.prompt, device)
    text_emb = text_emb.expand(args.num, -1, -1)

    print(f"\ngenerating {args.num} images with cfg_scale={args.cfg}, steps={args.steps}")

    latents = flow.sample(
        model=model,
        shape=(args.num, config.in_channels, config.latent_size, config.latent_size),
        text_embeddings=text_emb,
        null_text_embed=null_emb,
        cfg_scale=args.cfg,
        verbose=True,
    )

    print("decoding latents with VAE...")
    with torch.no_grad():
        images = vae.decode(latents / 0.18215).sample
    samples = (images.clamp(-1, 1) + 1) / 2

    safe_prompt = args.prompt[:50].replace(" ", "_").replace("/", "_")
    grid_path = os.path.join(args.output, f"{safe_prompt}_cfg{args.cfg}.png")
    grid = vutils.make_grid(samples, nrow=min(args.num, 4), padding=2, normalize=False)
    vutils.save_image(grid, grid_path)
    print(f"\nsaved to {grid_path}")

    for i in range(args.num):
        img_path = os.path.join(args.output, f"{safe_prompt}_{i}.png")
        vutils.save_image(samples[i], img_path)

    print(f"saved {args.num} individual images to {args.output}/")

if __name__ == "__main__":
    main()
