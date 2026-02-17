import torch
import torch.nn.functional as F


class RectifiedFlow:
    
    def __init__(self, num_steps: int = 28, device: str = "cpu"):
        self.num_steps = num_steps
        self.device = device

    def sample_timesteps(self, batch_size: int):
        u = torch.randn(batch_size, device=self.device)
        t = torch.sigmoid(u)
        return t

    def compute_loss(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        text_embeddings: torch.Tensor,
    ):
        batch_size = x_0.shape[0]
        t = self.sample_timesteps(batch_size)
        noise = torch.randn_like(x_0)

        t_broadcast = t.view(-1, 1, 1, 1)
        z_t = (1 - t_broadcast) * x_0 + t_broadcast * noise

        velocity = noise - x_0

        t_input = t * 1000

        v_pred = model(z_t, t_input, text_embeddings=text_embeddings)

        loss = F.mse_loss(v_pred, velocity)
        return loss

    @torch.no_grad()
    def p_sample_step(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t_curr: float,
        t_next: float,
        text_embeddings: torch.Tensor,
        null_text_embed: torch.Tensor,
        cfg_scale: float = 0.0,
    ):
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t_curr * 1000, device=x_t.device)

        if cfg_scale > 0:
            x_combined = torch.cat([x_t, x_t], dim=0)
            t_combined = torch.cat([t_tensor, t_tensor], dim=0)

            null_expanded = null_text_embed.expand(batch_size, -1, -1)
            txt_combined = torch.cat([text_embeddings, null_expanded], dim=0)
            v_combined = model(x_combined, t_combined, text_embeddings=txt_combined)

            v_cond, v_uncond = v_combined.chunk(2, dim=0)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x_t, t_tensor, text_embeddings=text_embeddings)

        dt = t_curr - t_next
        x_next = x_t - dt * v

        return x_next

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        text_embeddings: torch.Tensor,
        null_text_embed: torch.Tensor,
        cfg_scale: float = 4.0,
        verbose: bool = True,
    ):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)

        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

        for i in range(self.num_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()

            x = self.p_sample_step(
                model, x, t_curr, t_next,
                text_embeddings=text_embeddings,
                null_text_embed=null_text_embed,
                cfg_scale=cfg_scale,
            )

            if verbose and (i + 1) % 5 == 0:
                print(f"  step {i+1}/{self.num_steps}")

        return x
