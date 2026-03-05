import argparse
import json
import os
import torch
import torchvision.transforms as T
from datasets import load_dataset
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm

def load_checkpoint(output_dir):
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"shards_done": 0, "samples_done": 0}


def save_checkpoint(output_dir, shards_done, samples_done, ds_state=None):
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    payload = {"shards_done": shards_done, "samples_done": samples_done}
    if ds_state is not None:
        payload["ds_state"] = ds_state
    with open(checkpoint_path, "w") as f:
        json.dump(payload, f)


def stream_shard(ds_iter, shard_size, img_size):
    resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.LANCZOS)

    def process_sample(sample):
        try:
            img = sample["jpg"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return resize(img), sample["json"]["prompt"]
        except Exception:
            return None, None

    raw_samples = []
    for sample in tqdm(ds_iter, total=shard_size, desc="downloading"):
        raw_samples.append(sample)
        if len(raw_samples) >= shard_size:
            break

    if not raw_samples:
        return [], []

    images = []
    prompts = []
    for img, prompt in tqdm(map(process_sample, raw_samples), total=len(raw_samples), desc="resizing"):
        if img is not None:
            images.append(img)
            prompts.append(prompt)

    return images, prompts


def encode_to_latents(images, vae, batch_size, device):
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    all_latents = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pixels = torch.stack([to_tensor(img) for img in batch]).to(device)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * 0.18215
        all_latents.append(latents.cpu())

    return torch.cat(all_latents)


def encode_prompts(prompts, t5_model, tokenizer, batch_size, device):
    all_embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        tokens = tokenizer(
            batch,
            padding="max_length",
            max_length=170,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = t5_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
        all_embeddings.append(output.last_hidden_state.cpu().half())

    return torch.cat(all_embeddings)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/text2img")
    parser.add_argument("--fresh", action="store_true", help="stream from position 0, no skip")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")

    os.makedirs(args.output, exist_ok=True)

    checkpoint = load_checkpoint(args.output)
    shards_done = checkpoint["shards_done"]
    samples_done = checkpoint["samples_done"]

    if samples_done >= args.num_samples:
        print(f"already completed {samples_done} samples, ready to train.")
        return

    if samples_done > 0:
        print(f"resuming from shard {shards_done} ({samples_done} samples already done)")

    print("loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)

    print("loading T5...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_model = T5EncoderModel.from_pretrained("google/flan-t5-base").eval().to(device)

    null_embed_path = os.path.join(args.output, "null_embed.pt")
    if not os.path.exists(null_embed_path):
        null_tokens = tokenizer(
            "", padding="max_length", max_length=170, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            null_embed = t5_model(
                input_ids=null_tokens.input_ids,
                attention_mask=null_tokens.attention_mask,
            ).last_hidden_state.cpu().half()
        torch.save(null_embed, null_embed_path)
        print(f"saved null_embed.pt {null_embed.shape}")

    print(f"opening dataset...")
    ds = load_dataset("jackyhate/text-to-image-2M", split="train", streaming=True)
    ds_state = checkpoint.get("ds_state")
    if ds_state is not None:
        print(f"restoring dataset position from checkpoint (instant resume)...")
        ds.load_state_dict(ds_state)
        ds_iter = iter(ds)
    else:
        ds_iter = iter(ds)
        if samples_done > 0:
            print(f"no saved position — skipping {samples_done} samples (est. {samples_done // 1500 // 60:.0f}-{samples_done // 800 // 60:.0f} min)...")
            for _ in tqdm(range(samples_done), desc="skipping", unit="samples", miniters=1000):
                next(ds_iter)

    shard_idx = shards_done
    while samples_done < args.num_samples:
        this_shard_size = min(args.shard_size, args.num_samples - samples_done)
        print(f"\nshard {shard_idx} — {samples_done}/{args.num_samples} done")

        images, prompts = stream_shard(ds_iter, this_shard_size, args.img_size)
        if not images:
            print("no more samples in dataset, stopping")
            break

        print(f"VAE encoding {len(images)} images...")
        latents = encode_to_latents(images, vae, args.batch_size, device)
        del images

        print(f"T5 encoding {len(prompts)} prompts...")
        embeddings = encode_prompts(prompts, t5_model, tokenizer, args.batch_size, device)

        shard_prefix = os.path.join(args.output, f"shard_{shard_idx:04d}")
        torch.save(latents, shard_prefix + "_latents.pt")
        torch.save(embeddings, shard_prefix + "_embeddings.pt")
        with open(shard_prefix + "_prompts.json", "w") as f:
            json.dump(prompts, f)

        samples_done += len(prompts)
        shard_idx += 1
        save_checkpoint(args.output, shard_idx, samples_done, ds_state=ds.state_dict())
        print(f"shard saved — {samples_done}/{args.num_samples} total")

    print(f"\ndone! {samples_done} samples across {shard_idx} shards")


if __name__ == "__main__":
    main()
