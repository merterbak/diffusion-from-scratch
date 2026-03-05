import argparse
import os
import time
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from diffusion import RectifiedFlow
from model import Config as DiTConfig
from model import DiT
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel


SAMPLE_PROMPTS = [
    "a sunset over mountains",
    "a snowy cabin in the woods",
    "a waterfall in a dense jungle",
    "a red sports car on a highway",
]


@dataclass
class TrainSettings:
    data_dir: str = "data/text2img"
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 1e-4
    warmup_steps: int = 2000
    grad_clip: float = 0.1
    ema_decay: float = 0.9999
    text_dropout: float = 0.1
    log_every: int = 100
    sample_every: int = 10
    checkpoint_every: int = 50
    sample_cfg: float = 4.0
    sample_steps: int = 28
    patch_size: int = 2
    in_channels: int = 4
    latent_size: int = 32
    n_layer: int = 12
    n_head: int = 12
    head_dim: int = 64
    text_dim: int = 768
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    bias: bool = True
    max_shards: int = 20


class ShardDataset(Dataset):
    def __init__(self, data_dir: str, max_shards: int, null_embed: torch.Tensor, text_dropout: float = 0.1):
        super().__init__()
        shard_files = sorted(glob(os.path.join(data_dir, "shard_*_latents.pt")))[:max_shards]
        if not shard_files:
            raise FileNotFoundError(f"no shards found in {data_dir}")

        print(f"loading {len(shard_files)} shards into RAM...")
        all_latents, all_embeddings = [], []
        for i, path in enumerate(shard_files):
            prefix = path.replace("_latents.pt", "")
            print(f"  [{i+1}/{len(shard_files)}] {os.path.basename(path)}", flush=True)
            all_latents.append(torch.load(path, weights_only=True, map_location="cpu"))
            all_embeddings.append(torch.load(prefix + "_embeddings.pt", weights_only=True, map_location="cpu"))

        self.latents = torch.cat(all_latents)
        self.embeddings = torch.cat(all_embeddings)
        self.null_embed = null_embed
        self.text_dropout = text_dropout
        print(f"loaded {len(self.latents):,} samples — latents {tuple(self.latents.shape)}, embeddings {tuple(self.embeddings.shape)}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index: int):
        latent = self.latents[index].float()
        text = self.embeddings[index].float()

        if torch.rand(()) < 0.5:
            latent = latent.flip(-1)
        if self.text_dropout > 0.0 and torch.rand(()) < self.text_dropout:
            text = self.null_embed.float()

        return latent, text


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_loss: float,
    settings: TrainSettings,
    model_config: DiTConfig,
):
    torch.save({
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "train_config": asdict(settings),
        "model_config": asdict(model_config),
    }, path)


@torch.no_grad()
def generate_preview(
    ema_model: torch.nn.Module,
    flow: RectifiedFlow,
    settings: TrainSettings,
    epoch: int,
    sample_text: torch.Tensor,
    null_embed: torch.Tensor,
    vae: AutoencoderKL,
):
    ema_model.eval()
    samples = flow.sample(
        model=ema_model,
        shape=(len(sample_text), settings.in_channels, settings.latent_size, settings.latent_size),
        text_embeddings=sample_text.to("cuda"),
        null_text_embed=null_embed.to("cuda"),
        cfg_scale=settings.sample_cfg,
        verbose=False,
    )
    decoded = vae.decode(samples / 0.18215).sample
    images = (decoded.clamp(-1, 1) + 1) / 2
    grid = vutils.make_grid(images, nrow=min(len(sample_text), 4), padding=2)
    os.makedirs("samples", exist_ok=True)
    output_path = Path("samples") / f"epoch_{epoch + 1:04d}.png"
    vutils.save_image(grid, output_path)
    print(f"  saved preview {output_path}")


def train():
    defaults = TrainSettings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=defaults.data_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--max_shards", type=int, default=defaults.max_shards)
    args = parser.parse_args()

    settings = TrainSettings(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_shards=args.max_shards,
    )

    torch.backends.cuda.matmul.fp32_precision = "tf32"
    os.makedirs("checkpoints", exist_ok=True)

    print("loading T5...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5 = T5EncoderModel.from_pretrained("google/flan-t5-base").eval().to("cuda")
    with torch.no_grad():
        tokens = tokenizer(
            SAMPLE_PROMPTS, padding="max_length", max_length=170,
            truncation=True, return_tensors="pt",
        ).to("cuda")
        sample_text = t5(**tokens).last_hidden_state.float()
        null_tokens = tokenizer("", padding="max_length", max_length=170, return_tensors="pt").to("cuda")
        null_embed = t5(**null_tokens).last_hidden_state.float()
    del t5, tokenizer
    torch.cuda.empty_cache()

    dataset = ShardDataset(
        data_dir=settings.data_dir,
        max_shards=settings.max_shards,
        null_embed=null_embed.squeeze(0).cpu(),
        text_dropout=settings.text_dropout,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model_config = DiTConfig(
        patch_size=settings.patch_size,
        in_channels=settings.in_channels,
        n_layer=settings.n_layer,
        n_head=settings.n_head,
        head_dim=settings.head_dim,
        text_dim=settings.text_dim,
        latent_size=settings.latent_size,
        mlp_ratio=settings.mlp_ratio,
        rope_theta=settings.rope_theta,
        bias=settings.bias,
    )
    model = DiT(model_config).to("cuda")
    ema_model = DiT(model_config).to("cuda")
    ema_model.load_state_dict(model.state_dict())
    print("compiling model...")
    model = torch.compile(model)
    ema_model.eval()
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    flow = RectifiedFlow(num_steps=settings.sample_steps, device="cuda")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to("cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / settings.warmup_steps, 1.0),
    )

    global_step, start_epoch, best_loss = 0, 0, float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        print(f"resumed from {args.resume} at epoch {start_epoch + 1}")

    print(f"training {len(dataset):,} samples | epochs {settings.epochs} | batch {settings.batch_size}")

    for epoch in range(start_epoch, settings.epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for latents, text_embeds in loader:
            latents = latents.to("cuda", non_blocking=True)
            text_embeds = text_embeds.to("cuda", non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = flow.compute_loss(model, latents, text_embeddings=text_embeds)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler.step()
            update_ema(ema_model=ema_model, model=model, decay=settings.ema_decay)

            running_loss += loss.item()
            global_step += 1

            if global_step % settings.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  step {global_step:7d} | loss {loss.item():.4f} | lr {lr:.3e}")

        avg_loss = running_loss / len(loader)
        elapsed = time.time() - epoch_start
        print(f"epoch {epoch + 1:3d}/{settings.epochs} | loss {avg_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.3e} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                path=Path("checkpoints/best.pt"),
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                settings=settings,
                model_config=model_config,
            )
            print("  updated best checkpoint")

        if (epoch + 1) % settings.checkpoint_every == 0:
            save_checkpoint(
                path=Path(f"checkpoints/epoch_{epoch + 1}.pt"),
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                settings=settings,
                model_config=model_config,
            )
            print("  saved periodic checkpoint")

        if (epoch + 1) % settings.sample_every == 0:
            generate_preview(
                ema_model=ema_model,
                flow=flow,
                settings=settings,
                epoch=epoch,
                sample_text=sample_text,
                null_embed=null_embed,
                vae=vae,
            )

    save_checkpoint(
        path=Path("checkpoints/final.pt"),
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=settings.epochs - 1,
        global_step=global_step,
        best_loss=best_loss,
        settings=settings,
        model_config=model_config,
    )


if __name__ == "__main__":
    train()