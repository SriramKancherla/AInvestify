"""Prove the two tonight fixes: +5% ≠ +50% in pixels, and FiLM-X + CFG run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from render_channels import render_price_chart
from model import ConditionalUNet
from diffusion import GaussianDiffusion


def _window(end_return: float, n: int = 21) -> pd.DataFrame:
    closes = np.linspace(100.0, 100.0 * (1.0 + end_return), n)
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "Open": closes, "High": closes, "Low": closes, "Close": closes,
    })


def test_plus5_matches_plus50() -> None:
    """Autoscaled windows: same shape, different amplitude, same image."""
    a = render_price_chart(_window(0.05))
    b = render_price_chart(_window(0.50))
    mae = float(np.mean(np.abs(a - b)))
    assert mae < 0.02, f"+5% and +50% diverged (mae={mae:.4f})"
    print(f"  +5% vs +50% mean abs pixel diff = {mae:.4f}")


def test_film_x_and_cfg() -> None:
    import torch
    m = ConditionalUNet(base_channels=24, film_x=True)
    d = GaussianDiffusion(timesteps=10, device="cpu")
    y = torch.randn(2, 1, 128, 128)
    x = torch.randn(2, 3, 128, 128)
    h = torch.tensor([0, 1])
    loss = d.training_loss(m, y, x, h, cfg_drop_prob=0.5)
    out = d.ddim_sample(m, x, h, img_size=128, num_steps=4, guidance_scale=2.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert out.shape == (2, 1, 128, 128)
    print(f"  FiLM-X+CFG forward ok  loss={float(loss):.4f}  sample={tuple(out.shape)}")


if __name__ == "__main__":
    test_plus5_matches_plus50()
    test_film_x_and_cfg()
    print("ok")
