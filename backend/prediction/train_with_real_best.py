"""
A SEPARATE training run using fixed, known-stable hyperparameters (your
original Optuna result, not the batch_size=32/lr=0.002 experiment that's
been showing instability). Writes to its own folder - checkpoints_stable/ -
completely separate from training_config.CHECKPOINT_DIR, so this is safe
to run at the same time as train.py, evaluate.py, or infer.py without any
file collision.

Hyperparameters are hardcoded below, not loaded from optuna/best_hyperparameters.json -
deliberately, so this run is unaffected by whatever that file currently
holds or gets overwritten with later.

SAVES THE BEST MODEL BY VAL LOSS: two files land in checkpoints_stable/ -
"latest.pt" (every epoch, for resuming) and "best.pt" (only overwritten
when val_loss actually improves - a bad epoch can never destroy it).

NOTE: checkpoints_stable/latest.pt currently shipped with this integration
(epoch 25, val_loss≈0.0026) is what forecast_runner.py serves to the live
API. Running this script again continues training from it (see
load_checkpoint below) and will keep overwriting latest.pt/best.pt in
place - that's by design (this IS the deployed model's own training run),
just be aware a bad subsequent epoch can move val_loss the wrong way; only
best.pt is protected against that.

Usage:
    python train_with_real_best.py
"""

import sys
import time


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _Step:
    def __init__(self, description):
        self.description = description
    def __enter__(self):
        log(f"[STEP] starting: {self.description}")
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.t0
        if exc_type is not None:
            log(f"[STEP] FAILED after {elapsed:.1f}s: {self.description} ({exc_type.__name__}: {exc_val})")
        else:
            log(f"[STEP] finished ({elapsed:.1f}s): {self.description}")


def step(description: str):
    return _Step(description)


log("Script started.")
log(f"Python: {sys.version}")

log("[IMPORT] starting: stdlib (csv, os)")
import csv
import os
from pathlib import Path
log("[IMPORT] finished: stdlib (csv, os)")

log("[IMPORT] starting: torch")
_t0 = time.time()
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
log(f"[IMPORT] finished: torch ({time.time() - _t0:.1f}s) - version {torch.__version__}")

log("[IMPORT] starting: config, dataset, model, diffusion")
import config
from dataset import StockChartDataset
from model import ConditionalUNet
from diffusion import GaussianDiffusion
log("[IMPORT] finished: config, dataset, model, diffusion")

log("All imports complete - starting main().")

# ---------------------------------------------------------------------
# FIXED hyperparameters for this run - your original stable Optuna result.
# Hardcoded on purpose, not loaded from optuna/best_hyperparameters.json.
# ---------------------------------------------------------------------
BATCH_SIZE = 8
LEARNING_RATE = 0.0009605405228012401
BASE_CHANNELS = 24
DIFFUSION_TIMESTEPS = 1000
GRAD_CLIP_NORM = 1.0        # same protection as the main train.py fix
NUM_EPOCHS = 100
NUM_WORKERS = 0              # see training_config.py's note on why 0 is the safe default
VAL_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 1

# Own, separate output folder - never touches checkpoints/ or optuna/
OUTPUT_DIR = Path(__file__).parent / "checkpoints_stable"
CHECKPOINT_PATH = OUTPUT_DIR / "latest.pt"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"
LOG_CSV = OUTPUT_DIR / "training_log.csv"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, optimizer, device):
    if not CHECKPOINT_PATH.exists():
        log("No existing checkpoint in checkpoints_stable/ - starting from epoch 0.")
        return 0
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    log(f"Resumed from checkpoints_stable/latest.pt at epoch {ckpt['epoch']}")
    return ckpt["epoch"] + 1


def get_best_val_loss_so_far(device) -> float:
    if not BEST_CHECKPOINT_PATH.exists():
        return float("inf")
    ckpt = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    return ckpt.get("val_loss", float("inf"))


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "hyperparameters": {
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "base_channels": BASE_CHANNELS, "diffusion_timesteps": DIFFUSION_TIMESTEPS,
        },
    }, tmp_path)
    os.replace(tmp_path, path)


def log_row(epoch, train_loss, val_loss, elapsed_s):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["epoch", "train_loss", "val_loss", "elapsed_seconds"])
        writer.writerow([epoch, train_loss, val_loss, f"{elapsed_s:.1f}"])


def run_epoch(model, diffusion, loader, optimizer, device, train: bool, desc: str = ""):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for X, Y, horizon_idx in pbar:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        horizon_idx = horizon_idx.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(model, Y, X, horizon_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        else:
            with torch.no_grad():
                loss = diffusion.training_loss(model, Y, X, horizon_idx)
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_batches, 1)


def main():
    import training_config as tc
    tc.refuse_rebuild_or_retrain()
    log(f"Output folder for this run: {OUTPUT_DIR.resolve()}")
    log(f"Fixed hyperparameters: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
        f"base_channels={BASE_CHANNELS}, diffusion_timesteps={DIFFUSION_TIMESTEPS}, "
        f"grad_clip_norm={GRAD_CLIP_NORM}")

    with step("get device"):
        device = get_device()
        log(f"  device: {device}")
        if device.type == "cpu":
            log("  WARNING: training on CPU - this will be extremely slow for a real run.")

    with step("load train dataset"):
        train_ds = StockChartDataset("train")
        log(f"  train samples: {len(train_ds)}")

    with step("load val dataset"):
        val_ds = StockChartDataset("val")
        log(f"  val samples: {len(val_ds)}")

    with step("sanity-check: load ONE actual sample from disk"):
        X0, Y0, h0 = train_ds[0]
        log(f"  sample 0 loaded OK: X={X0.shape} Y={Y0.shape} horizon_idx={h0}")

    with step(f"build DataLoaders (batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})"):
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    with step("fetch first training batch (tests actual file I/O)"):
        first_batch = next(iter(train_loader))
        log(f"  first batch shapes: X={first_batch[0].shape} Y={first_batch[1].shape}")

    with step(f"build model on {device} (base_channels={BASE_CHANNELS})"):
        model = ConditionalUNet(
            num_horizons=len(config.HORIZON_DAYS_OPTIONS),
            base_channels=BASE_CHANNELS,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"  model built - {n_params:,} parameters")

    with step(f"build diffusion schedule (timesteps={DIFFUSION_TIMESTEPS})"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        diffusion = GaussianDiffusion(timesteps=DIFFUSION_TIMESTEPS, device=device)
        for attr in ["betas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "sqrt_recip_alphas", "posterior_variance"]:
            setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    with step("load checkpoint (if any)"):
        start_epoch = load_checkpoint(model, optimizer, device)
        best_val_loss = get_best_val_loss_so_far(device)
        if best_val_loss < float("inf"):
            log(f"  best val_loss so far: {best_val_loss:.4f}")

    log(f"Starting training: epoch {start_epoch} through {NUM_EPOCHS - 1}")
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        with step(f"epoch {epoch}: train pass ({len(train_loader)} batches)"):
            train_loss = run_epoch(model, diffusion, train_loader, optimizer, device, train=True,
                                    desc=f"[stable] Epoch {epoch} train")

        val_loss = None
        if epoch % VAL_EVERY_N_EPOCHS == 0:
            with step(f"epoch {epoch}: val pass ({len(val_loader)} batches)"):
                val_loss = run_epoch(model, diffusion, val_loader, optimizer, device, train=False,
                                      desc=f"[stable] Epoch {epoch} val")

        elapsed = time.time() - t0
        log(f"Epoch {epoch}: train_loss={train_loss:.4f}"
            + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
            + f" ({elapsed:.1f}s)")
        log_row(epoch, train_loss, val_loss, elapsed)

        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0:
            with step(f"epoch {epoch}: save checkpoint"):
                save_checkpoint(model, optimizer, epoch, val_loss, path=CHECKPOINT_PATH)

        if val_loss is not None and val_loss < best_val_loss:
            improvement = "first checkpoint" if best_val_loss == float("inf") else f"improved from {best_val_loss:.4f}"
            log(f"  New best val_loss: {val_loss:.4f} ({improvement}) - saving to {BEST_CHECKPOINT_PATH}")
            save_checkpoint(model, optimizer, epoch, val_loss, path=BEST_CHECKPOINT_PATH)
            best_val_loss = val_loss

    log(f"Training complete. Latest: {CHECKPOINT_PATH}")
    log(f"Best (val_loss={best_val_loss:.4f}): {BEST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
