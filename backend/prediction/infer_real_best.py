"""
Interactive inference for the STABLE model - the one trained by
train_with_real_best.py, saved in checkpoints_stable/, using your original
fixed Optuna hyperparameters (not the shared, currently-unstable tuned
JSON).

Defaults to latest.pt - the only checkpoint shipped with this integration
(epoch 25, val_loss≈0.0026). If you later run train_with_real_best.py again
and it produces a checkpoints_stable/best.pt, set USE_BEST_CHECKPOINT back
to True to prefer that instead.

Usage:
    python infer_real_best.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import config
from model import ConditionalUNet
from diffusion import GaussianDiffusion

# ---------------------------------------------------------------------
# FIXED hyperparameters - must match train_with_real_best.py exactly, since these
# determine the model's architecture (base_channels) and diffusion
# schedule (diffusion_timesteps). Hardcoded on purpose, not loaded from
# optuna/best_hyperparameters.json - that file belongs to the other,
# unstable run.
# ---------------------------------------------------------------------
BASE_CHANNELS = 24
DIFFUSION_TIMESTEPS = 1000

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints_stable"
USE_BEST_CHECKPOINT = False  # best.pt (protected, never overwritten by a bad epoch) vs
                               # latest.pt (most recent epoch, whatever its quality) - False
                               # because only latest.pt shipped with this integration; flip
                               # to True once a checkpoints_stable/best.pt exists
CHECKPOINT_PATH = CHECKPOINT_DIR / ("best.pt" if USE_BEST_CHECKPOINT else "latest.pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest() -> pd.DataFrame:
    manifest_path = config.PROCESSED_DIR / "manifest_split.csv"
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}.")
        print("Run build_dataset.py and make_splits.py first.")
        sys.exit(1)
    return pd.read_csv(manifest_path)


def load_model(device):
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}.")
        print("Run train_with_real_best.py first - this needs at least one saved epoch to generate anything.")
        sys.exit(1)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = ConditionalUNet(
        num_horizons=len(config.HORIZON_DAYS_OPTIONS),
        base_channels=BASE_CHANNELS,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = GaussianDiffusion(timesteps=DIFFUSION_TIMESTEPS, device=device)
    for attr in ["betas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                 "sqrt_recip_alphas", "posterior_variance"]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    label = "best" if USE_BEST_CHECKPOINT else "latest"
    val_loss = ckpt.get("val_loss")
    val_loss_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
    print(f"Loaded {label} checkpoint from epoch {ckpt['epoch']}{val_loss_str} (device: {device})")
    print(f"  ({CHECKPOINT_PATH})")
    return model, diffusion


def pick_ticker(manifest: pd.DataFrame) -> str:
    tickers = sorted(manifest["ticker"].unique())
    print(f"\n{len(tickers)} stocks available:\n")
    for i, t in enumerate(tickers):
        count = (manifest["ticker"] == t).sum()
        print(f"  [{i}] {t}  ({count} samples)")

    while True:
        choice = input("\nEnter a ticker symbol or its number: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(tickers):
            return tickers[int(choice)]
        if choice.upper() in tickers:
            return choice.upper()
        print("Not recognized - try again (type the exact ticker or its list number).")


def pick_sample(manifest: pd.DataFrame, ticker: str) -> pd.Series:
    rows = manifest[manifest["ticker"] == ticker].sort_values(["window_end", "horizon_days"]).reset_index(drop=True)
    print(f"\n{len(rows)} available windows/horizons for {ticker}:\n")
    for i, row in rows.iterrows():
        print(f"  [{i}] input {row.window_start} → {row.window_end}  |  "
              f"target {row.horizon_days}d ahead ({row.target_start} → {row.target_end})")

    while True:
        choice = input("\nPick a sample by number: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(rows):
            return rows.iloc[int(choice)]
        print("Not recognized - enter one of the listed numbers.")


def show_forecast(ticker, sample_row, X_u8, Y_generated, Y_true_u8):
    """Pops up one window with the three input channels, the generated
    forecast, the actual outcome, and a difference map - plus prints a
    real quantitative comparison (MSE/SSIM), not just two images to eyeball."""
    raw_gen = Y_generated[0].squeeze(0).cpu().numpy()  # still in [-1,1]-ish range, BEFORE clipping
    print(f"\n  Raw generated tensor stats (before display conversion):")
    print(f"    min={raw_gen.min():.3f}  max={raw_gen.max():.3f}  mean={raw_gen.mean():.3f}  std={raw_gen.std():.3f}")
    if raw_gen.std() < 0.05:
        print("    *** WARNING: near-zero variance - the model is producing a flat/blank output, ***")
        print("    *** not a plotting bug. This is a genuine model-quality issue (very likely undertraining). ***")

    gen_arr = ((Y_generated[0].squeeze(0).cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255)
    gen_u8 = gen_arr.astype(np.uint8)
    print(f"    after conversion to 0-255: min={gen_u8.min()}  max={gen_u8.max()}  std={gen_u8.std():.1f}")
    true_u8 = Y_true_u8.astype(np.uint8)

    mse = float(np.mean((gen_u8.astype(np.float32) - true_u8.astype(np.float32)) ** 2))
    similarity = float(ssim(true_u8, gen_u8, data_range=255))
    diff = np.abs(gen_u8.astype(np.float32) - true_u8.astype(np.float32))

    print(f"\n{'=' * 60}")
    print(f" Comparison: generated forecast vs. actual outcome")
    print(f"{'=' * 60}")
    print(f"  MSE (pixel-level, lower is better):        {mse:.1f}")
    print(f"  SSIM (structural similarity, 1.0 = perfect): {similarity:.3f}")
    if similarity > 0.7:
        print("  -> Strong structural match")
    elif similarity > 0.4:
        print("  -> Moderate match - general shape captured, details differ")
    else:
        print("  -> Weak match - treat this forecast with caution")

    panels = [
        (X_u8[:, :, 0], "Input: Price Chart", "gray"),
        (X_u8[:, :, 1], "Input: Fundamentals", "gray"),
        (X_u8[:, :, 2], "Input: News Tone", "gray"),
        (gen_u8, "GENERATED Forecast", "gray"),
        (true_u8, "Actual Outcome", "gray"),
        (diff, f"Difference\n(MSE={mse:.0f}, SSIM={similarity:.2f})", "hot"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(21, 4))
    checkpoint_label = "best" if USE_BEST_CHECKPOINT else "latest"
    fig.suptitle(
        f"[STABLE MODEL - {checkpoint_label} checkpoint]  {ticker}  |  "
        f"input {sample_row.window_start} → {sample_row.window_end}  |  "
        f"+{sample_row.horizon_days}d forecast ({sample_row.target_start} → {sample_row.target_end})  |  "
        f"SSIM={similarity:.2f}",
        fontsize=11,
    )
    for ax, (arr, title, cmap) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():  # visible border - a blank white
            spine.set_visible(True)        # panel must never look the same
            spine.set_color("red")          # as an empty/broken one
            spine.set_linewidth(1.5)
        if cmap == "hot":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def main():
    device = get_device()
    manifest = load_manifest()
    model, diffusion = load_model(device)

    ticker = pick_ticker(manifest)
    sample_row = pick_sample(manifest, ticker)

    print(f"\n{'=' * 60}")
    print(f" Inputs used for this forecast")
    print(f"{'=' * 60}")
    print(f"  Ticker:              {sample_row.ticker}")
    print(f"  Input window:        {sample_row.window_start}  to  {sample_row.window_end}  (1 year of history)")
    print(f"  Forecast horizon:    {sample_row.horizon_days} trading days ahead")
    print(f"  Target date range:   {sample_row.target_start}  to  {sample_row.target_end}")
    print(f"  Fundamentals as of:  {sample_row.fundamentals_as_of}")
    print(f"  Real SEC fundamentals available: {bool(sample_row.had_real_fundamentals)}")
    print(f"  News (8-K) days matched in window: {sample_row.news_articles_in_window}")

    data = np.load(config.SAMPLES_DIR / sample_row.file)
    X_u8, Y_true_u8 = data["X"], data["Y"]  # (128,128,3) and (128,128), uint8

    horizon_idx = sorted(config.HORIZON_DAYS_OPTIONS).index(int(sample_row.horizon_days))
    X_tensor = (torch.from_numpy(X_u8.astype(np.float32) / 255.0) * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    horizon_tensor = torch.tensor([horizon_idx], device=device)

    print(f"\nGenerating forecast (this runs the full {DIFFUSION_TIMESTEPS}-step reverse diffusion "
          f"process - may take a little while) ...")
    t0 = time.time()
    with torch.no_grad():
        Y_generated = diffusion.sample(model, X_tensor, horizon_tensor, img_size=config.IMG_HEIGHT, channels=1)
    print(f"Done in {time.time() - t0:.1f}s")
    print("\nOpening a window with the input channels, generated forecast, and actual outcome ...")
    show_forecast(ticker, sample_row, X_u8, Y_generated, Y_true_u8)


if __name__ == "__main__":
    main()
