"""
Hyperparameter search for the diffusion model, using Optuna.

Searches over learning rate, batch size, model width (base_channels), and
diffusion timestep count. Each trial trains for a SHORT budget
(training_config.OPTUNA_EPOCHS_PER_TRIAL epochs, not the full NUM_EPOCHS -
this is a search, not a final training run) and reports validation loss
back to Optuna after every epoch, so poorly-performing trials get pruned
(stopped early) instead of wasting the full budget.

RESUMABLE BY DESIGN: the study is stored in a SQLite file
(training_config.OPTUNA_STORAGE), not just in memory. If this script is
interrupted, re-running it reopens the same study and tops up to
OPTUNA_N_TRIALS TOTAL - already-completed trials are not re-run.

DEBUGGING: every step below prints a timestamped "[STEP] ..." line
immediately before and after it runs, flushed instantly (flush=True - not
relying on any buffering behavior). If this script ever appears to hang
again, look at the LAST "[STEP] starting: ..." line with no matching
"finished" line after it - that tells you exactly which operation is
stuck, instead of guessing. Common causes per step are noted inline.

Usage:
    python tune_hyperparameters.py
"""

import sys
import time


def log(msg: str):
    """Timestamped, immediately-flushed print - never buffered, never silent."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("Script started.")
log(f"Python: {sys.version}")
log(f"Executable: {sys.executable}")
log(f"Working directory: {sys.path[0] if sys.path else '?'}")

log("[IMPORT] starting: json, random, traceback (stdlib - should be instant; if this "
    "hangs, something is very wrong with the Python install itself)")
import json
import random
import traceback
log("[IMPORT] finished: json, random, traceback")

log("[IMPORT] starting: torch (often the slow one - can take 10s-2min+ on a cold start, "
    "or hang outright if there's a GPU driver / antivirus / DLL conflict on Windows)")
_t0 = time.time()
import torch
log(f"[IMPORT] finished: torch ({time.time() - _t0:.1f}s) - version {torch.__version__}")

log("[IMPORT] starting: torch.utils.data")
from torch.utils.data import DataLoader, Subset
log("[IMPORT] finished: torch.utils.data")

log("[IMPORT] starting: optuna")
_t0 = time.time()
import optuna
from optuna.pruners import MedianPruner
log(f"[IMPORT] finished: optuna ({time.time() - _t0:.1f}s) - version {optuna.__version__}")

from pathlib import Path

log("[IMPORT] starting: config")
import config
log("[IMPORT] finished: config")

log("[IMPORT] starting: training_config")
import training_config as tc
log("[IMPORT] finished: training_config")

log("[IMPORT] starting: dataset (StockChartDataset)")
from dataset import StockChartDataset
log("[IMPORT] finished: dataset")

log("[IMPORT] starting: model (ConditionalUNet)")
from model import ConditionalUNet
log("[IMPORT] finished: model")

log("[IMPORT] starting: diffusion (GaussianDiffusion)")
from diffusion import GaussianDiffusion
log("[IMPORT] finished: diffusion")

log("[IMPORT] starting: train (get_device, run_epoch)")
from train import get_device, run_epoch
log("[IMPORT] finished: train")

log("All imports complete - starting main().")


def step(description: str):
    """Wraps a block of work with 'starting'/'finished' log lines and a timer,
    so a hang shows up as a 'starting' line with nothing after it."""
    class _Step:
        def __enter__(self):
            log(f"[STEP] starting: {description}")
            self.t0 = time.time()
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.t0
            if exc_type is not None:
                log(f"[STEP] FAILED after {elapsed:.1f}s: {description}")
                log(f"[STEP] exception: {exc_type.__name__}: {exc_val}")
                traceback.print_exc()
            else:
                log(f"[STEP] finished ({elapsed:.1f}s): {description}")
    return _Step()


def objective(trial: optuna.Trial, device, train_ds, val_ds) -> float:
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    base_channels = trial.suggest_categorical("base_channels", [24, 32, 48, 64])
    diffusion_timesteps = trial.suggest_categorical("diffusion_timesteps", [500, 1000])
    log(f"Trial {trial.number}: batch_size={batch_size} lr={learning_rate:.2e} "
        f"base_channels={base_channels} diffusion_timesteps={diffusion_timesteps}")

    with step(f"trial {trial.number}: build train subset"):
        train_subset = train_ds
        if tc.OPTUNA_MAX_TRAIN_SAMPLES is not None and len(train_ds) > tc.OPTUNA_MAX_TRAIN_SAMPLES:
            idx = random.sample(range(len(train_ds)), tc.OPTUNA_MAX_TRAIN_SAMPLES)
            train_subset = Subset(train_ds, idx)
        log(f"  using {len(train_subset)} training samples this trial")

    with step(f"trial {trial.number}: build DataLoaders (num_workers={tc.NUM_WORKERS})"):
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=tc.NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=tc.NUM_WORKERS)

    with step(f"trial {trial.number}: fetch first training batch (tests actual file I/O, not just DataLoader setup)"):
        first_batch = next(iter(train_loader))
        log(f"  first batch shapes: X={first_batch[0].shape} Y={first_batch[1].shape}")
        # NOTE: if THIS step is the one that hangs, it means reading the
        # actual .npz sample files from disk is the bottleneck - not
        # PyTorch, not CUDA, not Optuna. Common Windows-specific causes:
        # antivirus real-time scanning many small files, the project
        # folder living inside a cloud-sync folder (OneDrive/Dropbox/etc)
        # where files are "cloud-only" placeholders that must be
        # downloaded on first access, or a slow/network drive.

    with step(f"trial {trial.number}: build model on {device}"):
        model = ConditionalUNet(num_horizons=len(config.HORIZON_DAYS_OPTIONS), base_channels=base_channels).to(device)

    with step(f"trial {trial.number}: build diffusion schedule"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        diffusion = GaussianDiffusion(timesteps=diffusion_timesteps, device=device)
        for attr in ["betas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "sqrt_recip_alphas", "posterior_variance"]:
            setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    val_loss = float("inf")
    for epoch in range(tc.OPTUNA_EPOCHS_PER_TRIAL):
        with step(f"trial {trial.number} epoch {epoch}: train pass"):
            run_epoch(model, diffusion, train_loader, optimizer, device, train=True,
                      desc=f"Trial {trial.number} epoch {epoch} train")
        with step(f"trial {trial.number} epoch {epoch}: val pass"):
            val_loss = run_epoch(model, diffusion, val_loader, optimizer, device, train=False,
                                  desc=f"Trial {trial.number} epoch {epoch} val")

        trial.report(val_loss, epoch)
        if trial.should_prune():
            log(f"Trial {trial.number}: pruned at epoch {epoch} (val_loss={val_loss:.4f})")
            raise optuna.TrialPruned()

    return val_loss


def main():
    tc.refuse_rebuild_or_retrain()
    log(f"Working directory: {Path.cwd()}")
    log(f"This script: {Path(__file__).resolve()}")

    with step("create optuna directory"):
        tc.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
        log(f"  Optuna dir: {tc.OPTUNA_DIR.resolve()}")
        db_path = tc.OPTUNA_DIR / "study.db"
        if db_path.exists():
            log(f"  Existing study.db found: {db_path.stat().st_size} bytes, "
                f"last modified {time.ctime(db_path.stat().st_mtime)}")
            log("  (If this script hangs at the NEXT step - 'open/create optuna study' - "
                "and this file is left over from a run you killed with Ctrl+C, the SQLite "
                "file may have a stale lock. Close this script, delete study.db and "
                "study.db-journal if present, and re-run - you'll lose search progress but "
                "not your dataset or anything else.)")

    with step("get device"):
        device = get_device()
        log(f"  device: {device}")

    with step("load train dataset (reads manifest_split.csv)"):
        train_ds = StockChartDataset("train")
        log(f"  train samples: {len(train_ds)}")

    with step("load val dataset"):
        val_ds = StockChartDataset("val")
        log(f"  val samples: {len(val_ds)}")

    with step("sanity-check: load ONE actual sample from disk right now"):
        # Fails fast and clearly if there's a problem reading .npz files at
        # all, rather than discovering it deep inside the first trial.
        X0, Y0, h0 = train_ds[0]
        log(f"  sample 0 loaded OK: X={X0.shape} Y={Y0.shape} horizon_idx={h0}")

    with step("open/create optuna study (SQLite)"):
        log(f"  storage: {tc.OPTUNA_STORAGE}")
        study = optuna.create_study(
            study_name=tc.OPTUNA_STUDY_NAME,
            storage=tc.OPTUNA_STORAGE,
            direction="minimize",
            pruner=MedianPruner(n_warmup_steps=1),
            load_if_exists=True,
        )

    already_done = len(study.trials)
    remaining = max(0, tc.OPTUNA_N_TRIALS - already_done)
    if already_done:
        log(f"Resuming study: {already_done} trials already completed, {remaining} remaining "
            f"toward the target of {tc.OPTUNA_N_TRIALS}.")
    if remaining == 0:
        log("Target trial count already reached. Nothing to do - see results below.")
    else:
        def trial_progress_callback(study, trial):
            state = trial.state.name
            value = f"{trial.value:.4f}" if trial.value is not None else "pruned/failed"
            log(f">>> Trial {trial.number + 1}/{tc.OPTUNA_N_TRIALS} finished ({state}): "
                f"val_loss={value} | params={trial.params} | best so far={study.best_value:.4f}")

        with step(f"run optuna search ({remaining} trials)"):
            study.optimize(
                lambda trial: objective(trial, device, train_ds, val_ds),
                n_trials=remaining,
                callbacks=[trial_progress_callback],
            )

    log(f"Best trial: {study.best_trial.number}")
    log(f"Best val loss: {study.best_value:.4f}")
    log(f"Best hyperparameters: {study.best_params}")

    with step("save best hyperparameters"):
        with open(tc.BEST_HYPERPARAMS_PATH, "w") as f:
            json.dump(study.best_params, f, indent=2)
        log(f"  saved -> {tc.BEST_HYPERPARAMS_PATH}")
    log("train.py will automatically pick these up on your next run.")


if __name__ == "__main__":
    main()
