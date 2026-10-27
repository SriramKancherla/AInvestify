"""
Trains the conditional diffusion model to predict a stock's future chart,
given a year of fundamentals+news+price history and a requested horizon.

CHECKPOINTED: saves model/optimizer/epoch state every
training_config.CHECKPOINT_EVERY_N_EPOCHS epochs. If training is
interrupted (crash, closed terminal, ran out of time), just run this same
command again - it resumes from the last checkpoint instead of starting
over. This matters a lot here: a real run over the full dataset is likely
to take hours to days depending on your hardware, and interruption should
be expected, not treated as a failure.

If tune_hyperparameters.py has been run, its winning hyperparameters are
loaded automatically at startup (see load_tuned_hyperparameters below) -
you don't need to do anything manually for that.

Writes to training_config.CHECKPOINT_DIR (checkpoints/), separate from
checkpoints_stable/ (train_with_real_best.py's own output folder, and where
the currently-deployed model lives) - so running this script can never
overwrite the checkpoint the live API serves from. See forecast_runner.py's
docstring for why the live API uses the checkpoints_stable/ pipeline
specifically.

DEBUGGING: every import and every setup step prints a timestamped
"[STEP]"/"[IMPORT]" line immediately before and after it runs (flushed
instantly - never buffered). If this script ever appears to hang, the
LAST "starting: ..." line with no matching "finished" line tells you
exactly which operation is stuck.

Usage:
    python train.py
"""

import os
import sys
import time


def log(msg: str):
    """Timestamped, immediately-flushed print - never buffered, never silent."""
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


_VERBOSE_IMPORTS = __name__ == "__main__"  # only log import timings when run directly -
                                             # tune_hyperparameters.py imports get_device/run_epoch
                                             # from this file too, and would otherwise trigger a
                                             # second confusing "Script started" banner and re-log
                                             # every import it does.


def ilog(msg: str):
    if _VERBOSE_IMPORTS:
        log(msg)


ilog("Script started.")
ilog(f"Python: {sys.version}")
ilog(f"Executable: {sys.executable}")

ilog("[IMPORT] starting: stdlib (csv, json)")
import csv
import json
ilog("[IMPORT] finished: stdlib (csv, json)")

ilog("[IMPORT] starting: torch (often the slow one on a cold Windows start - "
     "antivirus scanning its DLLs, or GPU driver/CUDA conflicts)")
_t0 = time.time()
import torch
torch.backends.cudnn.benchmark = True  # auto-tune conv algorithms for our fixed 128x128
                                         # input size - safe since our shape never changes
                                         # between batches, standard speedup for conv-heavy models
from torch.utils.data import DataLoader
ilog(f"[IMPORT] finished: torch ({time.time() - _t0:.1f}s) - version {torch.__version__}")

ilog("[IMPORT] starting: tqdm")
from tqdm import tqdm
ilog("[IMPORT] finished: tqdm")

ilog("[IMPORT] starting: config")
import config
ilog("[IMPORT] finished: config")

ilog("[IMPORT] starting: training_config")
import training_config as tc
ilog("[IMPORT] finished: training_config")

ilog("[IMPORT] starting: dataset (StockChartDataset)")
from dataset import StockChartDataset
ilog("[IMPORT] finished: dataset")

ilog("[IMPORT] starting: model (ConditionalUNet)")
from model import ConditionalUNet
ilog("[IMPORT] finished: model")

ilog("[IMPORT] starting: diffusion (GaussianDiffusion)")
from diffusion import GaussianDiffusion
ilog("[IMPORT] finished: diffusion")

ilog("All imports complete - starting main().")


def load_tuned_hyperparameters():
    """
    If tune_hyperparameters.py has been run, its winning config is saved to
    training_config.BEST_HYPERPARAMS_PATH. Load it here and override the
    defaults in training_config.py - so running train.py after a search
    automatically uses the tuned values without you editing anything by
    hand. If no search has been run yet, this is a no-op and the defaults
    in training_config.py apply as-is.
    """
    if not tc.BEST_HYPERPARAMS_PATH.exists():
        log("No tuned hyperparameters found - using the defaults in training_config.py.")
        log("(Run tune_hyperparameters.py first if you want Optuna to search for better ones.)")
        return

    with open(tc.BEST_HYPERPARAMS_PATH) as f:
        best = json.load(f)

    log(f"Loaded tuned hyperparameters from {tc.BEST_HYPERPARAMS_PATH}:")
    for key, value in best.items():
        log(f"  {key}: {value}")

    if "batch_size" in best:
        tc.BATCH_SIZE = best["batch_size"]
    if "learning_rate" in best:
        tc.LEARNING_RATE = best["learning_rate"]
    if "base_channels" in best:
        tc.BASE_CHANNELS = best["base_channels"]
    if "diffusion_timesteps" in best:
        tc.DIFFUSION_TIMESTEPS = best["diffusion_timesteps"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, optimizer, device, scaler=None):
    if not tc.CHECKPOINT_PATH.exists():
        log("No existing checkpoint found - starting from epoch 0.")
        return 0  # start_epoch
    ckpt = torch.load(tc.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
        log("Restored AMP GradScaler state from checkpoint.")
    log(f"Resumed from checkpoint at epoch {ckpt['epoch']}")
    return ckpt["epoch"] + 1


def save_checkpoint(model, optimizer, epoch, scaler=None):
    tc.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then atomically rename into place. Without
    # this, a process reading the checkpoint (e.g. evaluate.py running
    # alongside training) could catch it mid-write and load a corrupted
    # file - os.replace() is atomic on both Windows and POSIX, so readers
    # only ever see either the old complete file or the new complete file.
    tmp_path = tc.CHECKPOINT_PATH.with_suffix(".tmp")
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, tmp_path)
    os.replace(tmp_path, tc.CHECKPOINT_PATH)


def log_row(epoch, train_loss, val_loss, elapsed_s):
    tc.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not tc.LOG_CSV.exists()
    with open(tc.LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["epoch", "train_loss", "val_loss", "elapsed_seconds"])
        writer.writerow([epoch, train_loss, val_loss, f"{elapsed_s:.1f}"])


def run_epoch(model, diffusion, loader, optimizer, device, train: bool, desc: str = "", scaler=None):
    """
    scaler: a torch.cuda.amp.GradScaler, only used (and only needed) when
    device is CUDA. Enables mixed-precision training - runs the forward
    pass in float16 on GPUs with Tensor Cores (essentially all modern
    NVIDIA cards, including yours), which is typically 1.5-3x faster with
    no meaningful quality loss. GradScaler exists specifically to prevent
    tiny gradients from underflowing to zero in float16 - it scales the
    loss up before backward(), then unscales before the optimizer step.
    On CPU, autocast/scaler are both no-ops (guarded by `enabled=`).
    """
    model.train() if train else model.eval()
    use_amp = device.type == "cuda"
    if use_amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=True)  # safe default for callers (e.g.
                                                             # tune_hyperparameters.py) that don't
                                                             # pass one explicitly
    total_loss, n_batches = 0.0, 0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for X, Y, horizon_idx in pbar:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        horizon_idx = horizon_idx.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = diffusion.training_loss(model, Y, X, horizon_idx, cfg_drop_prob=tc.CFG_DROP_PROB)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # must unscale BEFORE clipping, or the clip
                                              # threshold is meaningless (grads are still
                                              # scaled up by an arbitrary factor at this point)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.GRAD_CLIP_NORM)
                optimizer.step()
        else:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
                loss = diffusion.training_loss(model, Y, X, horizon_idx, cfg_drop_prob=0.0)
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_batches, 1)


def main():
    tc.refuse_rebuild_or_retrain()
    with step("load tuned hyperparameters (if any)"):
        load_tuned_hyperparameters()
        log(f"  ABLATE_CHANNELS={tc.ABLATE_CHANNELS!r} (None = full multimodal X)")

    with step("get device"):
        device = get_device()
        log(f"  device: {device}")
        if device.type == "cpu":
            log("  WARNING: training on CPU. Fine for smoke-testing, but a real run over "
                "the full dataset needs a GPU - this will be extremely slow otherwise.")

    with step("load train dataset (reads manifest_split.csv)"):
        train_ds = StockChartDataset("train")
        log(f"  train samples: {len(train_ds)}")

    with step("load val dataset"):
        val_ds = StockChartDataset("val")
        log(f"  val samples: {len(val_ds)}")

    with step("sanity-check: load ONE actual sample from disk right now"):
        X0, Y0, h0 = train_ds[0]
        log(f"  sample 0 loaded OK: X={X0.shape} Y={Y0.shape} horizon_idx={h0}")

    with step(f"build DataLoaders (batch_size={tc.BATCH_SIZE}, num_workers={tc.NUM_WORKERS})"):
        train_loader = DataLoader(
            train_ds, batch_size=tc.BATCH_SIZE, shuffle=True, num_workers=tc.NUM_WORKERS,
            pin_memory=(device.type == "cuda"), persistent_workers=(tc.NUM_WORKERS > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=tc.BATCH_SIZE, shuffle=False, num_workers=tc.NUM_WORKERS,
            pin_memory=(device.type == "cuda"), persistent_workers=(tc.NUM_WORKERS > 0),
        )

    with step("fetch first training batch (tests actual file I/O)"):
        first_batch = next(iter(train_loader))
        log(f"  first batch shapes: X={first_batch[0].shape} Y={first_batch[1].shape}")

    with step(f"build model on {device} (base_channels={tc.BASE_CHANNELS})"):
        model = ConditionalUNet(
            num_horizons=len(config.HORIZON_DAYS_OPTIONS),
            base_channels=tc.BASE_CHANNELS,
            film_x=tc.FILM_X,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"  model built - {n_params:,} parameters")

    with step(f"build diffusion schedule (timesteps={tc.DIFFUSION_TIMESTEPS})"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc.LEARNING_RATE)
        diffusion = GaussianDiffusion(timesteps=tc.DIFFUSION_TIMESTEPS, device=device)
        for attr in ["betas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "sqrt_recip_alphas", "posterior_variance"]:
            setattr(diffusion, attr, getattr(diffusion, attr).to(device))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        if device.type == "cuda":
            log("  mixed-precision training (AMP) enabled")

    with step("load checkpoint (if any)"):
        start_epoch = load_checkpoint(model, optimizer, device, scaler=scaler)

    log(f"Starting training: epoch {start_epoch} through {tc.NUM_EPOCHS - 1}")
    for epoch in range(start_epoch, tc.NUM_EPOCHS):
        t0 = time.time()
        with step(f"epoch {epoch}: train pass ({len(train_loader)} batches)"):
            train_loss = run_epoch(model, diffusion, train_loader, optimizer, device, train=True,
                                    desc=f"Epoch {epoch} train", scaler=scaler)

        val_loss = None
        if epoch % tc.VAL_EVERY_N_EPOCHS == 0:
            with step(f"epoch {epoch}: val pass ({len(val_loader)} batches)"):
                val_loss = run_epoch(model, diffusion, val_loader, optimizer, device, train=False,
                                      desc=f"Epoch {epoch} val", scaler=scaler)

        elapsed = time.time() - t0
        log(f"Epoch {epoch}: train_loss={train_loss:.4f}"
            + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
            + f" ({elapsed:.1f}s)")
        log_row(epoch, train_loss, val_loss, elapsed)

        if epoch % tc.CHECKPOINT_EVERY_N_EPOCHS == 0:
            with step(f"epoch {epoch}: save checkpoint"):
                save_checkpoint(model, optimizer, epoch, scaler=scaler)

    log(f"Training complete. Checkpoint: {tc.CHECKPOINT_PATH}, log: {tc.LOG_CSV}")


if __name__ == "__main__":
    main()
