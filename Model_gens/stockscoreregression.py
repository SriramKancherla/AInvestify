import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from pathlib import Path

from fundamentals_labels import FEATURE_COLS, fundamental_score, prepare_numeric

_SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = _SCRIPT_DIR.parent / "datasets"

print("Welcome to fundamentals stock regression")

data = prepare_numeric(pd.read_csv(DATASETS_DIR / "financials_cleaned.csv", sep=","))

X = data[FEATURE_COLS]
Y = fundamental_score(data)

modell = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

# Held-out sanity check. Does NOT affect the saved artifact (fit on all data below);
# for a rigorous, leakage-free view use Model_gens/evaluate_fundamentals.py.
X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
_pred = clone(modell).fit(X_tr, y_tr).predict(X_te)
print(
    f"[holdout] R2={r2_score(y_te, _pred):.3f} "
    f"MAE={mean_absolute_error(y_te, _pred):.3f}"
)

modell.fit(X, Y)

joblib.dump(modell, _SCRIPT_DIR / "stock_score_regression.pkl")
print("Model saved as stock_score_regression.pkl")
