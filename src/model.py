import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    patch_size: int = 2
    in_channels: int = 4
    out_channels: Optional[int] = None
    n_layer: int = 12
    n_head: int = 12
    head_dim: int = 64
    text_dim: int = 768
    latent_size: int = 32
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    bias: bool = True

    @property
    def n_embd(self):
        return self.n_head * self.head_dim

    @property
    def grid_size(self):
        return self.latent_size // self.patch_size


def sinusoidal_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))

    return emb


def apply_rotary_pos_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    cos = freqs_cos[None, None, :, :]
    sin = freqs_sin[None, None, :, :]

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon) * self.weight


class StepEmbedding(nn.Module):

    def __init__(self, n_embd: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype):
        timesteps_proj = sinusoidal_embedding(
            timestep, self.freq_dim,
            flip_sin_to_cos=True, downscale_freq_shift=0, scale=1,
        )
        return self.mlp(timesteps_proj.to(dtype=hidden_dtype))


class MSRoPE2D(nn.Module):

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        assert head_dim % 2 == 0, "head_dim must be even"
        self.axis_dim = head_dim // 2

    def _get_freqs(self, positions: torch.Tensor, dim: int):
        assert dim % 2 == 0
        inv_freq = 1.0 / torch.pow(
            self.theta,
            torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim
        )
        angles = torch.outer(positions.float(), inv_freq)
        return torch.cos(angles), torch.sin(angles)

    def forward(self, grid_h: int, grid_w: int, text_seq_len: int, device: torch.device):
        h_positions = torch.arange(grid_h, device=device) - grid_h // 2
        w_positions = torch.arange(grid_w, device=device) - grid_w // 2

        h_cos, h_sin = self._get_freqs(h_positions, self.axis_dim)
        w_cos, w_sin = self._get_freqs(w_positions, self.axis_dim)

        h_cos = h_cos[:, None, :].expand(grid_h, grid_w, -1)
        h_sin = h_sin[:, None, :].expand(grid_h, grid_w, -1)
        w_cos = w_cos[None, :, :].expand(grid_h, grid_w, -1)
        w_sin = w_sin[None, :, :].expand(grid_h, grid_w, -1)

        img_cos = torch.cat([h_cos, w_cos], dim=-1)
        img_sin = torch.cat([h_sin, w_sin], dim=-1)

        img_cos = img_cos.reshape(grid_h * grid_w, -1)
        img_sin = img_sin.reshape(grid_h * grid_w, -1)

        img_cos = img_cos.repeat_interleave(2, dim=-1)
        img_sin = img_sin.repeat_interleave(2, dim=-1)

        diagonal_start = max(grid_h // 2, grid_w // 2)
        txt_positions = torch.arange(text_seq_len, device=device) + diagonal_start

        txt_axis_cos, txt_axis_sin = self._get_freqs(txt_positions, self.axis_dim)

        txt_cos = torch.cat([txt_axis_cos, txt_axis_cos], dim=-1)
        txt_sin = torch.cat([txt_axis_sin, txt_axis_sin], dim=-1)

        txt_cos = txt_cos.repeat_interleave(2, dim=-1)
        txt_sin = txt_sin.repeat_interleave(2, dim=-1)

        return img_cos, img_sin, txt_cos, txt_sin


class JointAttention(nn.Module):
    def __init__(self, config: Config):

        super().__init__()
        n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim

        self.vis_q_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.vis_k_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.vis_v_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.vis_o_proj = nn.Linear(n_embd, n_embd, bias=config.bias)

        self.txt_q_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.txt_k_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.txt_v_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.txt_o_proj = nn.Linear(n_embd, n_embd, bias=config.bias)

        self.vis_q_norm = RMSNorm(self.head_dim)
        self.vis_k_norm = RMSNorm(self.head_dim)
        self.txt_q_norm = RMSNorm(self.head_dim)
        self.txt_k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        vis_x: torch.Tensor,
        txt_x: torch.Tensor,
        img_rope_cos: Optional[torch.Tensor] = None,
        img_rope_sin: Optional[torch.Tensor] = None,
        txt_rope_cos: Optional[torch.Tensor] = None,
        txt_rope_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):

        B, N_vis, C = vis_x.shape
        N_txt = txt_x.shape[1]

        q_vis = self.vis_q_proj(vis_x).unflatten(-1, (self.n_head, self.head_dim)).transpose(1, 2)
        k_vis = self.vis_k_proj(vis_x).unflatten(-1, (self.n_head, self.head_dim)).transpose(1, 2)
        v_vis = self.vis_v_proj(vis_x).unflatten(-1, (self.n_head, self.head_dim)).transpose(1, 2)

        q_txt = self.txt_q_proj(txt_x).unflatten(-1, (self.n_head, self.head_dim)).transpose(1, 2)
        k_txt = self.txt_k_proj(txt_x).unflatten(-1, (self.n_head, self.head_dim)).transpose(1, 2)
        v_txt = self.txt_v_proj(txt_x).unflatten(-1, (self.n_head, self.head_dim)).transpose(1, 2)

        q_vis = self.vis_q_norm(q_vis)
        k_vis = self.vis_k_norm(k_vis)
        q_txt = self.txt_q_norm(q_txt)
        k_txt = self.txt_k_norm(k_txt)

        if img_rope_cos is not None:
            q_vis = apply_rotary_pos_emb(q_vis, img_rope_cos, img_rope_sin)
            k_vis = apply_rotary_pos_emb(k_vis, img_rope_cos, img_rope_sin)
        if txt_rope_cos is not None:
            q_txt = apply_rotary_pos_emb(q_txt, txt_rope_cos, txt_rope_sin)
            k_txt = apply_rotary_pos_emb(k_txt, txt_rope_cos, txt_rope_sin)

        q = torch.cat([q_txt, q_vis], dim=2)
        k = torch.cat([k_txt, k_vis], dim=2)
        v = torch.cat([v_txt, v_vis], dim=2)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

        txt_out = attn_out[:, :, :N_txt, :]
        vis_out = attn_out[:, :, N_txt:, :]

        vis_out = vis_out.transpose(1, 2).flatten(2)
        txt_out = txt_out.transpose(1, 2).flatten(2)

        vis_out = self.vis_o_proj(vis_out)
        txt_out = self.txt_o_proj(txt_out)

        return vis_out, txt_out


class FeedForward(nn.Module):

    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: float = 4.0):

        super().__init__()
        dim_out = dim_out or dim
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class DualStreamBlock(nn.Module):

    def __init__(self, config: Config):

        super().__init__()
        n_embd = config.n_embd

        self.vis_mod = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, 6 * n_embd, bias=True))
        self.vis_norm1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.vis_norm2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.vis_ff = FeedForward(dim=n_embd, dim_out=n_embd, mult=config.mlp_ratio)

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, 6 * n_embd, bias=True))
        self.txt_norm1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.txt_ff = FeedForward(dim=n_embd, dim_out=n_embd, mult=config.mlp_ratio)

        self.attn = JointAttention(config)

    def modulate(self, x: torch.Tensor, mod_params: torch.Tensor):

        shift, scale, gate = mod_params.chunk(3, dim=-1)
        x_modulated = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        gate = gate.unsqueeze(1)
        return x_modulated, gate

    def forward(
        self,
        vis: torch.Tensor,
        txt: torch.Tensor,
        cond: torch.Tensor,
        img_rope_cos: Optional[torch.Tensor] = None,
        img_rope_sin: Optional[torch.Tensor] = None,
        txt_rope_cos: Optional[torch.Tensor] = None,
        txt_rope_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        vis_mod_params = self.vis_mod(cond)
        txt_mod_params = self.txt_mod(cond)

        vis_mod1, vis_mod2 = vis_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        vis_normed = self.vis_norm1(vis)
        vis_modulated, vis_gate1 = self.modulate(vis_normed, vis_mod1)

        txt_normed = self.txt_norm1(txt)
        txt_modulated, txt_gate1 = self.modulate(txt_normed, txt_mod1)

        vis_attn_out, txt_attn_out = self.attn(
            vis_modulated, txt_modulated,
            img_rope_cos, img_rope_sin,
            txt_rope_cos, txt_rope_sin,
            attn_mask,
        )

        vis = vis + vis_gate1 * vis_attn_out
        txt = txt + txt_gate1 * txt_attn_out

        vis_normed = self.vis_norm2(vis)
        vis_modulated, vis_gate2 = self.modulate(vis_normed, vis_mod2)
        vis = vis + vis_gate2 * self.vis_ff(vis_modulated)

        txt_normed = self.txt_norm2(txt)
        txt_modulated, txt_gate2 = self.modulate(txt_normed, txt_mod2)
        txt = txt + txt_gate2 * self.txt_ff(txt_modulated)

        return vis, txt


class AdaNorm(nn.Module):

    def __init__(self, n_embd: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(n_embd, elementwise_affine=False, eps=eps)
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embd, 2 * n_embd, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        params = self.linear(cond)
        shift, scale = params.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class DiT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        n_embd = config.n_embd
        self._out_channels = config.out_channels or config.in_channels

        self.patch_embed = nn.Conv2d(
            config.in_channels, n_embd,
            kernel_size=config.patch_size, stride=config.patch_size, bias=True,
        )

        self.rope = MSRoPE2D(head_dim=config.head_dim, theta=config.rope_theta)

        self.step_embed = StepEmbedding(n_embd=n_embd, freq_dim=256)

        self.text_norm = RMSNorm(config.text_dim, eps=1e-6)
        self.text_proj = nn.Linear(config.text_dim, n_embd)

        self.layers = nn.ModuleList([
            DualStreamBlock(config) for _ in range(config.n_layer)
        ])

        self.output_norm = AdaNorm(n_embd, eps=1e-6)
        self.output_proj = nn.Linear(
            n_embd,
            config.patch_size * config.patch_size * self._out_channels,
            bias=True,
        )

        self.init_model_weights()

    def init_model_weights(self):
        std = 0.02

        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)

        nn.init.normal_(self.step_embed.mlp[0].weight, std=std)
        nn.init.normal_(self.step_embed.mlp[2].weight, std=std)

        nn.init.normal_(self.text_proj.weight, std=std)

        for layer in self.layers:
            nn.init.zeros_(layer.vis_mod[-1].weight)
            nn.init.zeros_(layer.vis_mod[-1].bias)
            nn.init.zeros_(layer.txt_mod[-1].weight)
            nn.init.zeros_(layer.txt_mod[-1].bias)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.output_norm.linear[-1].weight)
        nn.init.zeros_(self.output_norm.linear[-1].bias)

    def unpatchify(self, x: torch.Tensor):
        B = x.shape[0]
        P = self.config.patch_size
        C = self._out_channels
        G = self.config.grid_size
        S = self.config.latent_size

        x = x.reshape(B, G, G, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, C, S, S)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ):
        vis = self.patch_embed(x)
        vis = vis.flatten(2).transpose(1, 2)

        timestep = timestep.to(vis.dtype)
        cond = self.step_embed(timestep, vis.dtype)

        txt = self.text_proj(self.text_norm(text_embeddings))
        text_seq_len = txt.shape[1]

        img_rope_cos, img_rope_sin, txt_rope_cos, txt_rope_sin = self.rope(
            self.config.grid_size, self.config.grid_size, text_seq_len, device=x.device
        )

        attn_mask = None
        if text_mask is not None:
            batch_size, img_seq_len = vis.shape[:2]
            image_ones = torch.ones(batch_size, img_seq_len, dtype=torch.bool, device=x.device)
            joint_mask = torch.cat([text_mask, image_ones], dim=1)
            attn_mask = torch.where(joint_mask, 0.0, float("-inf"))
            attn_mask = attn_mask.to(dtype=vis.dtype)
            attn_mask = attn_mask[:, None, None, :]

        for layer in self.layers:
            vis, txt = layer(
                vis=vis,
                txt=txt,
                cond=cond,
                img_rope_cos=img_rope_cos,
                img_rope_sin=img_rope_sin,
                txt_rope_cos=txt_rope_cos,
                txt_rope_sin=txt_rope_sin,
                attn_mask=attn_mask,
            )

        vis = self.output_norm(vis, cond)
        output = self.output_proj(vis)
        output = self.unpatchify(output)

        return output
