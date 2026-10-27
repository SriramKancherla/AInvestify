"""
Proves tune_hyperparameters.py actually works: builds a tiny synthetic
dataset, runs a real (tiny) Optuna search against it, confirms best
hyperparameters get saved correctly, and - critically - confirms resuming
an interrupted study tops up to the target trial count rather than
starting over or overshooting it.

Deliberately tiny (few trials, 1 epoch each, small model) so this runs
quickly on CPU. No network required.
"""

import shutil
import json
import numpy as np
import pandas as pd
import optuna

import config
import training_config as tc
from dataset import StockChartDataset
import tune_hyperparameters as hpo
from train import get_device

TEST_DIR = config.ROOT / "data" / "processed_hpo_smoketest"


def build_tiny_dataset(n_per_split: int = 10):
    samples_dir = TEST_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, sample_idx = [], 0
    for split in ["train", "val"]:
        for _ in range(n_per_split):
            X = rng.integers(0, 256, (config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8)
            Y = rng.integers(0, 256, (config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)
            fname = f"sample_{sample_idx:04d}.npz"
            np.savez_compressed(samples_dir / fname, X=X, Y=Y)
            rows.append({
                "sample_id": sample_idx, "file": fname, "ticker": "TEST",
                "horizon_days": int(rng.choice(config.HORIZON_DAYS_OPTIONS)),
                "split": split,
            })
            sample_idx += 1
    manifest_path = TEST_DIR / "manifest_split.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, samples_dir


def main():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    if tc.OPTUNA_DIR.exists():
        shutil.rmtree(tc.OPTUNA_DIR)

    print("=== Building tiny synthetic dataset ===")
    manifest_path, samples_dir = build_tiny_dataset()
    config.SAMPLES_DIR = samples_dir
    config.PROCESSED_DIR = TEST_DIR  # so hpo.main()'s internal StockChartDataset("train")
                                       # calls (which use the default manifest path) resolve here
    print(f"  Built at {TEST_DIR}\n")

    # Small, fast test settings - override the real config for this test only
    tc.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    tc.OPTUNA_STORAGE = f"sqlite:///{tc.OPTUNA_DIR / 'test_study.db'}"
    tc.OPTUNA_STUDY_NAME = "test_study"
    tc.OPTUNA_N_TRIALS = 4
    tc.OPTUNA_EPOCHS_PER_TRIAL = 1
    tc.NUM_WORKERS = 0

    train_ds = StockChartDataset("train", manifest_path=manifest_path)
    val_ds = StockChartDataset("val", manifest_path=manifest_path)
    device = get_device()

    print("=== Test 1: run a partial search (2 of 4 target trials) ===")
    study = optuna.create_study(
        study_name=tc.OPTUNA_STUDY_NAME, storage=tc.OPTUNA_STORAGE,
        direction="minimize", load_if_exists=True,
    )
    study.optimize(lambda t: hpo.objective(t, device, train_ds, val_ds), n_trials=2)
    assert len(study.trials) == 2, f"expected 2 trials, got {len(study.trials)}"
    print(f"  OK - {len(study.trials)} trials completed\n")

    print("=== Test 2: 'interrupt' and resume - should run exactly 2 MORE, not restart ===")
    hpo.main()  # this re-opens the same study via storage and tops up to OPTUNA_N_TRIALS=4
    reloaded_study = optuna.load_study(study_name=tc.OPTUNA_STUDY_NAME, storage=tc.OPTUNA_STORAGE)
    assert len(reloaded_study.trials) == 4, f"expected 4 total trials after resuming, got {len(reloaded_study.trials)}"
    print(f"  OK - resumed and topped up to {len(reloaded_study.trials)} total trials (not 2, not 6)\n")

    print("=== Test 3: running again with target already met does nothing ===")
    hpo.main()
    final_study = optuna.load_study(study_name=tc.OPTUNA_STUDY_NAME, storage=tc.OPTUNA_STORAGE)
    assert len(final_study.trials) == 4, "should NOT have run more trials once the target was already met"
    print(f"  OK - still {len(final_study.trials)} trials, no wasted work\n")

    print("=== Test 4: best hyperparameters were saved correctly ===")
    assert tc.BEST_HYPERPARAMS_PATH.exists()
    with open(tc.BEST_HYPERPARAMS_PATH) as f:
        best = json.load(f)
    expected_keys = {"batch_size", "learning_rate", "base_channels", "diffusion_timesteps"}
    assert expected_keys.issubset(best.keys()), f"missing expected keys, got {best.keys()}"
    print(f"  OK - best_hyperparameters.json contains: {best}\n")

    shutil.rmtree(TEST_DIR)
    shutil.rmtree(tc.OPTUNA_DIR)
    print("All hyperparameter tuning tests passed.")


if __name__ == "__main__":
    main()
