"""
A conditional U-Net that predicts the noise added to the target chart Y,
given:
  - the noisy target itself (1 channel)
  - the conditioning input X: fundamentals+news+price, already fused as
    image channels (3 channels) - concatenated directly onto the noisy
    input, the same way the "core novelty" fusion works throughout this
    project
  - the current diffusion timestep (standard DDPM noise-level conditioning)
  - the requested forecast horizon (which of the 6 configured horizons -
    this is what lets ONE model predict at any distance up to a year out,
    instead of training a separate model per horizon)

Deliberately kept compact (not a large-scale published U-Net) so it can
be smoke-tested on CPU in seconds. Scale up base_channels / num_res_blocks
once you have real GPU compute and a fully-fetched dataset - see README.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer/DDPM sinusoidal position embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Conv residual block with conditioning (timestep+horizon embedding)
    injected via a learned per-channel shift (FiLM-style, additive)."""

    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        img_channels: int = 1,       # target Y has 1 channel
        cond_channels: int = 3,      # X (fundamentals+news+price) has 3 channels
        num_horizons: int = 6,
        base_channels: int = 48,     # kept small for CPU smoke-testing; scale up for real training
        channel_mults: tuple = (1, 2, 4),
        time_emb_dim: int = 128,
        film_x: bool = False,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.horizon_emb = nn.Embedding(num_horizons, time_emb_dim)
        # Optional pooled encoding of X, added to the FiLM shift. Off by
        # default: the shipped checkpoint and the paper concatenate X only
        # at the stem. forecast_runner turns this on only if the checkpoint
        # actually contains x_encoder weights.
        self.film_x = film_x
        if film_x:
            self.x_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(cond_channels * 16, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.x_encoder = None

        in_ch = img_channels + cond_channels  # noisy Y concatenated with X
        chs = [base_channels * m for m in channel_mults]

        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_ch = chs[0]
        for ch in chs:
            self.down_blocks.append(ResBlock(prev_ch, ch, time_emb_dim))
            self.downs.append(Down(ch))
            prev_ch = ch

        self.mid_block1 = ResBlock(prev_ch, prev_ch, time_emb_dim)
        self.mid_block2 = ResBlock(prev_ch, prev_ch, time_emb_dim)

        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for ch in reversed(chs):
            self.ups.append(Up(prev_ch))
            self.up_blocks.append(ResBlock(prev_ch + ch, ch, time_emb_dim))  # +ch for skip connection
            prev_ch = ch

        self.out_norm = nn.GroupNorm(8, prev_ch)
        self.out_conv = nn.Conv2d(prev_ch, img_channels, 3, padding=1)

    def forward(self, noisy_y, x_cond, t, horizon_idx):
        """
        noisy_y: (B, 1, H, W)
        x_cond:  (B, 3, H, W)
        t:       (B,) int64 diffusion timesteps
        horizon_idx: (B,) int64 which configured horizon this sample targets
        """
        cond = self.time_mlp(sinusoidal_embedding(t, self.time_emb_dim)) + self.horizon_emb(horizon_idx)
        if self.x_encoder is not None:
            cond = cond + self.x_encoder(x_cond)

        h = torch.cat([noisy_y, x_cond], dim=1)
        h = self.in_conv(h)

        skips = []
        for block, down in zip(self.down_blocks, self.downs):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
