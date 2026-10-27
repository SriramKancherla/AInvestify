"""
Training hyperparameters for the diffusion model. Separate from config.py
(which governs dataset building) since these are a different concern.

BATCH_SIZE / LEARNING_RATE / BASE_CHANNELS / DIFFUSION_TIMESTEPS below are
DEFAULTS. If tune_hyperparameters.py has been run, train.py automatically
loads BEST_HYPERPARAMS_PATH and overrides these four with the tuned
values - see the "Loading tuned hyperparameters" note in train.py.
"""

from pathlib import Path

BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
GRAD_CLIP_NORM = 1.0       # caps the total gradient norm per step - standard, well-established
                            # technique to prevent occasional large/unstable updates from
                            # causing loss spikes, without needing to lower the learning rate
DIFFUSION_TIMESTEPS = 1000
BASE_CHANNELS = 48         # small on purpose - see model.py's docstring on why
NUM_WORKERS = 0            # DataLoader worker processes. Defaulting to 0 (load data in the
                            # main process, no multiprocessing) because >0 workers on Windows
                            # spawn as full fresh processes (re-importing torch/optuna/etc from
                            # scratch each time) - and this is a known deadlock risk specifically
                            # when spawned from inside a nested subprocess, which is exactly what
                            # happens when train.py/tune_hyperparameters.py run via
                            # run_pipeline.py. If you're on Linux/Mac and running scripts
                            # directly (not through run_pipeline.py), 2-4 workers can speed up
                            # data loading meaningfully - safe to raise it in that case.

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "latest.pt"
LOG_CSV = CHECKPOINT_DIR / "training_log.csv"

VAL_EVERY_N_EPOCHS = 1
CHECKPOINT_EVERY_N_EPOCHS = 1   # checkpoint often - training runs can be long,
                                  # and this is what makes resume-after-interrupt possible

# ---------------------------------------------------------------------
# Hyperparameter search (Optuna) - see tune_hyperparameters.py
# ---------------------------------------------------------------------
OPTUNA_DIR = Path(__file__).parent / "optuna"
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DIR / 'study.db'}"   # persistent storage = resumable study
OPTUNA_STUDY_NAME = "diffusion_forecaster_hpo"
OPTUNA_N_TRIALS = 20            # TOTAL trials targeted across the whole study, not "N more
                                  # each time you run it" - re-running tops up to this total
OPTUNA_EPOCHS_PER_TRIAL = 3      # short budget per trial - this is a search, not a final
                                  # training run; the winning config gets the real NUM_EPOCHS
OPTUNA_MAX_TRAIN_SAMPLES = 5000   # HPO only needs enough data to compare configs fairly, not
                                   # the full training set - searching against 100k+ samples
                                   # per trial makes a 20-trial search take unreasonably long.
                                   # A random 5000-sample subset (re-sampled fresh each trial)
                                   # is plenty for relative comparison. Set to None to disable
                                   # and use the full training set (slow - not recommended).
BEST_HYPERPARAMS_PATH = OPTUNA_DIR / "best_hyperparameters.json"

# ---------------------------------------------------------------------
# Channel ablation — see old/prediction_old/README.md "Channel ablation"
# ---------------------------------------------------------------------
# Channel order in X is [price, fundamentals, news] (see render_channels.stack_channels).
# Use a distinct CHECKPOINT_PATH per variant so runs don't overwrite each other.
#   None         — all three channels (default)
#   "price_only" — keep price; zero fundamentals + news
#   "price_fund" — keep price + fundamentals; zero news
#   "price_news" — keep price + news; zero fundamentals
ABLATE_CHANNELS = None

# Classifier-free guidance is off. The submitted model was trained with X
# always present and sampled with guidance scale 1.
CFG_DROP_PROB = 0.0
GUIDANCE_SCALE = 1.0

# Do not inject a pooled encoding of X into the residual blocks. X enters
# only by concatenation at the stem, matching the shipped checkpoint.
FILM_X = False


def refuse_rebuild_or_retrain() -> None:
    """Stop a dataset rebuild or a new training run.

    The training images and the split manifest are not in the repo. The
    presentation uses checkpoints_stable/latest.pt and the published tables.
    A new dataset or a new architecture would be a different system, and
    those tables would no longer apply.
    """
    raise SystemExit(
        "Refusing to rebuild or retrain. The training images and split manifest "
        "are not in this repo. A new dataset or a new training run would no longer "
        "be the system in the paper, and the published tables would no longer apply. "
        "Demo checkpoints_stable/latest.pt and the volatility cone."
    )
