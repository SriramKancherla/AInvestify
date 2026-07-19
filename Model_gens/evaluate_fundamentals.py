"""
Held-out evaluation harness for the fundamentals ensemble.

Why this exists
---------------
The three training scripts (`stockfundamentalanalysis.py`, `rfr_fundamentals_scorer.py`,
`stockscoreregression.py`) each call `.fit(...)` on the *entire* dataset and save a
production artifact. That is correct for shipping (use all the data), but it means none
of them report how well the models *generalize* — an in-sample R^2/AUC on ~500 rows is
close to memorization and tells you nothing about a live ticker.

This harness answers two questions honestly:

1. Per-model held-out faithfulness to the heuristic label (the target every model is
   trained to reproduce): regressors vs `fundamental_score`, classifier vs
   `good_fundamentals`.
2. What each ensemble member contributes — i.e. how held-out MAE / ranking change if you
   drop the RF regressor or the classifier from the mean.

It uses 5-fold **out-of-fold** cross-validation: every row receives a prediction from a
model trained on the *other* folds, so (a) there is no train/serve leakage and (b) all
ensemble configurations are compared on exactly the same held-out rows.

The model factories below mirror the hyperparameters in the training scripts. If you
change a training script's hyperparameters, update the matching factory here.

Note: the Keras NN was dropped from the shipped ensemble (weakest held-out faithfulness,
heaviest dependency), so it is not evaluated here.

Run:  python Model_gens/evaluate_fundamentals.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

from fundamentals_labels import (
    FEATURE_COLS,
    fundamental_score,
    good_fundamentals,
    prepare_numeric,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = _SCRIPT_DIR.parent / "datasets"

RANDOM_STATE = 42
N_SPLITS = 5


# --- Model factories (mirror the training scripts) ---------------------------

def make_classifier() -> RandomForestClassifier:
    # stockfundamentalanalysis.py
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)


def make_rf_regressor() -> RandomForestRegressor:
    # rfr_fundamentals_scorer.py
    return RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )


def make_xgb_regressor() -> XGBRegressor:
    # stockscoreregression.py
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


# --- Out-of-fold prediction --------------------------------------------------

MODEL_KEYS = ["classifier", "rf_regressor", "xgb_regressor"]


def out_of_fold_predictions(
    X: pd.DataFrame, y_score: pd.Series, y_good: pd.Series
) -> dict[str, np.ndarray]:
    """Return a full-length OOF prediction vector (in [0,1]) for each base model.

    Each model is trained on the training folds and predicts the held-out fold, so
    every row's prediction comes from a model that never saw that row.
    """
    n = len(X)
    oof = {k: np.zeros(n, dtype=float) for k in MODEL_KEYS}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr, te) in enumerate(skf.split(X, y_good), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        ys_tr = y_score.iloc[tr]
        yg_tr = y_good.iloc[tr]

        # Classifier -> probability of the "good" class, mirroring stock_insights.py.
        clf = make_classifier().fit(X_tr, yg_tr)
        classes = list(clf.classes_)
        pos_idx = classes.index(1) if 1 in classes else (1 if len(classes) > 1 else 0)
        oof["classifier"][te] = _clip01(clf.predict_proba(X_te)[:, pos_idx])

        # RF regressor.
        rf = make_rf_regressor().fit(X_tr, ys_tr)
        oof["rf_regressor"][te] = _clip01(rf.predict(X_te))

        # XGBoost regressor.
        xgb = make_xgb_regressor().fit(X_tr, ys_tr)
        oof["xgb_regressor"][te] = _clip01(xgb.predict(X_te))

        print(f"  fold {fold}/{N_SPLITS} done (test rows: {len(te)})")

    return oof


# --- Reporting ---------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a, b)
    return float(rho)


def report_per_model(
    oof: dict[str, np.ndarray], y_score: pd.Series, y_good: pd.Series
) -> None:
    ys = y_score.to_numpy()
    yg = y_good.to_numpy()

    print("\n=== Per-model held-out faithfulness (5-fold out-of-fold) ===")

    print("\nRegressors vs heuristic `fundamental_score`:")
    print(f"  {'model':<16}{'R^2':>8}{'MAE':>8}{'Spearman':>10}")
    for key in ["rf_regressor", "xgb_regressor"]:
        p = oof[key]
        print(
            f"  {key:<16}{r2_score(ys, p):>8.3f}{mean_absolute_error(ys, p):>8.3f}"
            f"{_spearman(ys, p):>10.3f}"
        )

    print("\nClassifier vs heuristic `good_fundamentals` (positive rate: "
          f"{yg.mean():.3f}):")
    clf_p = oof["classifier"]
    auc = roc_auc_score(yg, clf_p) if len(np.unique(yg)) > 1 else float("nan")
    hard = (clf_p >= 0.5).astype(int)
    print(f"  ROC-AUC={auc:.3f}  accuracy@0.5={accuracy_score(yg, hard):.3f}  "
          f"F1@0.5={f1_score(yg, hard, zero_division=0):.3f}")


def _ensemble(oof: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    return np.mean(np.vstack([oof[k] for k in keys]), axis=0)


def report_ensemble_configs(oof: dict[str, np.ndarray], y_score: pd.Series) -> None:
    ys = y_score.to_numpy()

    configs = {
        "full (all 3)": MODEL_KEYS,
        "drop RF regressor": ["classifier", "xgb_regressor"],
        "drop classifier": ["rf_regressor", "xgb_regressor"],
    }

    preds = {name: _ensemble(oof, keys) for name, keys in configs.items()}
    full = preds["full (all 3)"]

    print("\n=== Ensemble configurations on held-out predictions ===")
    print(f"  {'config':<22}{'MAE':>8}{'Spearman':>10}{'rank-corr vs full':>20}")
    for name, p in preds.items():
        mae = mean_absolute_error(ys, p)
        sp = _spearman(ys, p)
        rc = 1.0 if name.startswith("full") else _spearman(full, p)
        print(f"  {name:<22}{mae:>8.4f}{sp:>10.3f}{rc:>20.4f}")

    # Drop-RF verdict.
    full_mae = mean_absolute_error(ys, full)
    rf_drop_mae = mean_absolute_error(ys, preds["drop RF regressor"])
    rf_drop_rankcorr = _spearman(full, preds["drop RF regressor"])
    delta = rf_drop_mae - full_mae
    print("\n=== Drop-RF verdict (held-out) ===")
    print(f"  full MAE            : {full_mae:.4f}")
    print(f"  drop-RF MAE         : {rf_drop_mae:.4f}  (delta {delta:+.4f})")
    print(f"  drop-RF rank-corr   : {rf_drop_rankcorr:.4f} vs full ordering")
    if delta <= 0.0 and rf_drop_rankcorr >= 0.99:
        print("  -> Dropping the RF regressor does NOT hurt held-out MAE and barely "
              "changes ordering. Safe to drop.")
    elif delta <= 0.005 and rf_drop_rankcorr >= 0.98:
        print("  -> Dropping the RF regressor is roughly neutral on held-out data. "
              "Reasonable to drop for simplicity/latency.")
    else:
        print("  -> Dropping the RF regressor measurably changes held-out behavior. "
              "Keep it, or investigate before dropping.")


def main() -> None:
    df = prepare_numeric(pd.read_csv(DATASETS_DIR / "financials_cleaned.csv"))
    X = df[FEATURE_COLS].reset_index(drop=True)
    y_score = fundamental_score(df).reset_index(drop=True)
    y_good = good_fundamentals(df).reset_index(drop=True)

    print(f"Rows: {len(X)}  Features: {len(FEATURE_COLS)}  "
          f"positive-class rate: {y_good.mean():.3f}")
    print(f"Running {N_SPLITS}-fold out-of-fold CV...")

    oof = out_of_fold_predictions(X, y_score, y_good)
    report_per_model(oof, y_score, y_good)
    report_ensemble_configs(oof, y_score)


if __name__ == "__main__":
    main()
