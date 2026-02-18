import argparse
import json
import os
import torch
import torchvision.transforms as T
from datasets import load_dataset
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm


def download_images(num_samples, img_size=256):

    ds = load_dataset("jackyhate/text-to-image-2M", split="train", streaming=True)
    resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.LANCZOS)

    images = []
    prompts = []
    skipped = 0

    for i, sample in enumerate(tqdm(ds, total=num_samples, desc="downloading")):
        if len(images) >= num_samples:
            break
        try:
            img = sample["jpg"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(resize(img))
            prompts.append(sample["json"]["prompt"])
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  skip {i}: {e}")

    if skipped:
        print(f"  skipped {skipped} bad samples")
    print(f"  got {len(images)} images at {img_size}x{img_size}")
    return images, prompts


def encode_to_latents(images, vae, batch_size=16, device="cuda"):
    print(f"encoding {len(images)} images to VAE latents...")

    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    all_latents = []
    for i in tqdm(range(0, len(images), batch_size), desc="VAE encoding"):
        batch = images[i : i + batch_size]
        pixels = torch.stack([to_tensor(img) for img in batch]).to(device)

        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample()
            latents = latents * 0.18215

        all_latents.append(latents.cpu())

    all_latents = torch.cat(all_latents)
    print(f"  latents shape: {all_latents.shape}")
    print(f"  std: {all_latents.std():.3f} (want ~1.0)")
    return all_latents


def encode_prompts(prompts, clip_model, tokenizer, batch_size=64, device="cuda"):
    print(f"encoding {len(prompts)} prompts with CLIP...")

    all_embs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="CLIP encoding"):
        batch = prompts[i : i + batch_size]
        tokens = tokenizer(
            batch, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            emb = clip_model(**tokens).last_hidden_state
        all_embs.append(emb.cpu().half())

    embeddings = torch.cat(all_embs)
    print(f"  embeddings shape: {embeddings.shape}")
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="data/text2img")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    images, prompts = download_images(args.num_samples, args.img_size)

    print(f"\nloading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)
    latents = encode_to_latents(images, vae, args.batch_size, device)
    del vae, images
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nloading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)

    embeddings = encode_prompts(prompts, clip, tokenizer, batch_size=64, device=device)

    null_tokens = tokenizer(
        "", padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        null_embed = clip(**null_tokens).last_hidden_state.cpu().half()

    del clip, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    os.makedirs(args.output, exist_ok=True)
    print(f"\nsaving to {args.output}/")

    torch.save(latents, os.path.join(args.output, "latents.pt"))
    print(f"  latents.pt:    {latents.shape}")

    torch.save(embeddings, os.path.join(args.output, "embeddings.pt"))
    print(f"  embeddings.pt: {embeddings.shape}")

    torch.save(null_embed, os.path.join(args.output, "null_embed.pt"))
    print(f"  null_embed.pt: {null_embed.shape}")

    with open(os.path.join(args.output, "prompts.json"), "w") as f:
        json.dump(prompts, f)
    print(f"  prompts.json:  {len(prompts)} prompts")


if __name__ == "__main__":
    main()
