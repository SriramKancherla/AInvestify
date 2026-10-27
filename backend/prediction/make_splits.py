"""
Splits the dataset into train/val/test by TIME, not randomly - random
splitting would leak information, since consecutive samples for the same
ticker share up to ~90% of their input days (252-day window, 21-day
stride). A random split would very likely put near-duplicate windows in
both train and test, letting a model "cheat" by pattern-matching data it
has effectively already seen.

Approach: pick two cutoff dates. Every sample whose TARGET fully resolves
before the first cutoff goes to train. Every sample whose INPUT WINDOW
starts after the second cutoff goes to test. Everything in between is
dropped entirely (an "embargo" gap) - this guarantees zero date overlap
between any train sample and any test sample, for any ticker.

Usage:
    python make_splits.py
Reads data/processed/manifest.csv, writes data/processed/manifest_split.csv
(same content, plus a `split` column: "train", "val", "test", or "dropped_embargo").
"""

import pandas as pd

import config

# Fraction of the full date range allocated to train / val, before the test
# period. The remainder (after val) is test. Adjust to taste.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC


def compute_split_dates(manifest: pd.DataFrame) -> dict:
    """
    Cutoff dates computed from the actual data range present in the
    manifest (not hardcoded). Based on the range of window_start values
    specifically (not target_end) - window_start's achievable range is
    inherently smaller than target_end's (a window can't start so late
    that its horizon runs past the data's actual end), so basing cutoffs
    on target_end's fuller range risks leaving zero valid window_starts
    for the test split. Using window_start's own range guarantees all
    three splits get real samples whenever TRAIN_FRAC + VAL_FRAC < 1.
    """
    window_starts = pd.to_datetime(manifest["window_start"])
    full_start, usable_end = window_starts.min(), window_starts.max()
    total_days = (usable_end - full_start).days

    train_cutoff = full_start + pd.Timedelta(days=int(total_days * TRAIN_FRAC))
    val_cutoff = full_start + pd.Timedelta(days=int(total_days * (TRAIN_FRAC + VAL_FRAC)))

    full_end = pd.to_datetime(manifest["target_end"]).max()
    return {"full_start": full_start, "full_end": full_end, "train_cutoff": train_cutoff, "val_cutoff": val_cutoff}


def assign_split(manifest: pd.DataFrame, dates: dict) -> pd.Series:
    target_end = pd.to_datetime(manifest["target_end"])
    window_start = pd.to_datetime(manifest["window_start"])

    train_mask = target_end <= dates["train_cutoff"]
    # Embargo: anything whose target crosses the train cutoff, or whose
    # window starts before the val cutoff, is dropped rather than assigned -
    # this is what actually prevents overlap, not just the labels.
    val_mask = (~train_mask) & (window_start > dates["train_cutoff"]) & (target_end <= dates["val_cutoff"])
    test_mask = (~train_mask) & (~val_mask) & (window_start > dates["val_cutoff"])

    split = pd.Series("dropped_embargo", index=manifest.index)
    split[train_mask] = "train"
    split[val_mask] = "val"
    split[test_mask] = "test"
    return split


def main():
    import training_config as tc
    tc.refuse_rebuild_or_retrain()
    manifest = pd.read_csv(config.DATASET_MANIFEST)
    if manifest.empty:
        print(f"{config.DATASET_MANIFEST} is empty - run build_dataset.py first.")
        return

    dates = compute_split_dates(manifest)
    print(f"Data spans {dates['full_start'].date()} to {dates['full_end'].date()}")
    print(f"Train: target_end <= {dates['train_cutoff'].date()}")
    print(f"Val:   window_start > {dates['train_cutoff'].date()} and target_end <= {dates['val_cutoff'].date()}")
    print(f"Test:  window_start > {dates['val_cutoff'].date()}")

    manifest["split"] = assign_split(manifest, dates)

    counts = manifest["split"].value_counts()
    print(f"\nSplit sizes:\n{counts.to_string()}")
    print(f"\n(dropped_embargo = samples whose dates straddle a cutoff - excluded from all splits to guarantee zero leakage)")

    for split_name in ["train", "val", "test"]:
        n = counts.get(split_name, 0)
        if n == 0:
            print(f"\nERROR: the '{split_name}' split is EMPTY. This happens when your date range is too "
                  f"short relative to INPUT_WINDOW_DAYS + the largest horizon - there's no room left for "
                  f"that split's window_start values to exist. Fix: widen START_DATE/END_DATE in config.py, "
                  f"or adjust TRAIN_FRAC/VAL_FRAC in this file, then re-run.")
        elif n < 20:
            print(f"\nWARNING: the '{split_name}' split only has {n} samples - likely too few to draw any "
                  f"reliable conclusions from. Same fix as above: widen the date range or adjust the fractions.")

    out_path = config.PROCESSED_DIR / "manifest_split.csv"
    manifest.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
