import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from diffusion import RectifiedFlow
from model import Config as DiTConfig
from model import DiT
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel



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
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    text_dropout: float = 0.1
    log_every: int = 100
    checkpoint_every: int = 50
    sample_every: int = 10
    sample_cfg: float = 4.0
    sample_steps: int = 28
    num_workers: int = 4
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


class LatentTextDataset(Dataset):
    def __init__(self, data_dir: str, text_dropout: float = 0.1):
        super().__init__()
        root = Path(data_dir)
        self.latents = torch.load(root / "latents.pt", map_location="cpu", weights_only=True)
        self.text_embeddings = torch.load(root / "embeddings.pt", map_location="cpu", weights_only=True)
        self.null_embedding = torch.load(root / "null_embed.pt", map_location="cpu", weights_only=True)
        self.text_dropout = text_dropout

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, index: int):
        latent = self.latents[index].float()
        text = self.text_embeddings[index].float()

        if torch.rand(()) < 0.5:
            latent = latent.flip(-1)
        if self.text_dropout > 0.0 and torch.rand(()) < self.text_dropout:
            text = self.null_embedding[0].float()

        return latent, text

@torch.no_grad()
def update_ema_weights(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
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
    payload = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "train_config": asdict(settings),
        "model_config": asdict(model_config),
    }
    torch.save(payload, path)


@torch.no_grad()
def generate_preview(
    ema_model: torch.nn.Module,
    flow: RectifiedFlow,
    settings: TrainSettings,
    epoch: int,
    prompt_bank: torch.Tensor,
    null_embed: torch.Tensor,
    vae,
):
    ema_model.eval()
    prompt_bank = prompt_bank.to("cuda")
    null_embed = null_embed.to("cuda")

    samples = flow.sample(
        model=ema_model,
        shape=(
            prompt_bank.shape[0],
            settings.in_channels,
            settings.latent_size,
            settings.latent_size,
        ),
        text_embeddings=prompt_bank,
        null_text_embed=null_embed,
        cfg_scale=settings.sample_cfg,
        verbose=False,
    )

    decoded = vae.decode(samples / 0.18215).sample
    images = (decoded.clamp(-1, 1) + 1) / 2

    grid = vutils.make_grid(images, nrow=min(prompt_bank.shape[0], 4), padding=2)
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
    args = parser.parse_args()

    settings = TrainSettings(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


    torch.backends.cuda.matmul.fp32_precision = "tf32"

    os.makedirs("checkpoints", exist_ok=True)

    print("loading dataset...")
    dataset = LatentTextDataset(settings.data_dir, text_dropout=settings.text_dropout)

    dataloader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=True,
    )


    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to("cuda")

    tokens = tokenizer(
        SAMPLE_PROMPTS, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        sample_text = clip_model(**tokens).last_hidden_state.float()

    null_tokens = tokenizer(
        "", padding="max_length", max_length=77, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        null_embed = clip_model(**null_tokens).last_hidden_state.float()

    del clip_model, tokenizer
    torch.cuda.empty_cache()


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
    ema_model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total_params:,}")

    flow = RectifiedFlow(num_steps=settings.sample_steps, device="cuda")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to("cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    warmup_steps = settings.warmup_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0),
    )

    global_step = 0
    start_epoch = 0
    best_loss = float("inf")

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        print(f"resumed from {resume_path} at epoch {start_epoch + 1}")

    print(f"epochs: {settings.epochs} (start at {start_epoch + 1})")

    for epoch in range(start_epoch, settings.epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for latents, text_embeds in dataloader:
            latents = latents.to("cuda", non_blocking=True)
            text_embeds = text_embeds.to("cuda", non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = flow.compute_loss(model, latents, text_embeddings=text_embeds)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler.step()
            update_ema_weights(ema_model, model, settings.ema_decay)

            batch_loss = float(loss.item())
            running_loss += batch_loss
            global_step += 1

            if global_step % settings.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  step {global_step:7d} | loss {batch_loss:.4f} | lr {lr:.3e}")

        avg_loss = running_loss / len(dataloader)
        elapsed = time.time() - epoch_start
        print(f"epoch {epoch + 1:3d}/{settings.epochs} | loss {avg_loss:.4f} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                Path("checkpoints/best.pt"),
                model,
                ema_model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_loss,
                settings,
                model_config,
            )
            print("  updated best checkpoint")

        if (epoch + 1) % settings.checkpoint_every == 0:
            save_checkpoint(
                Path("checkpoints") / f"epoch_{epoch + 1}.pt",
                model,
                ema_model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_loss,
                settings,
                model_config,
            )
            print("  saved periodic checkpoint")

        if (epoch + 1) % settings.sample_every == 0:
            generate_preview(
                ema_model=ema_model,
                flow=flow,
                settings=settings,
                epoch=epoch,
                prompt_bank=sample_text,
                null_embed=null_embed,
                vae=vae,
            )

    save_checkpoint(
        Path("checkpoints/final.pt"),
        model,
        ema_model,
        optimizer,
        scheduler,
        settings.epochs - 1,
        global_step,
        best_loss,
        settings,
        model_config,
    )


if __name__ == "__main__":
    train()

