from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from pathlib import Path

from fundamentals_labels import FEATURE_COLS, fundamental_score, prepare_numeric

_SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = _SCRIPT_DIR.parent / "datasets"

data = prepare_numeric(pd.read_csv(DATASETS_DIR / "financials_cleaned.csv"))

X = data[FEATURE_COLS]
Y = fundamental_score(data)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

# Held-out sanity check. Does NOT affect the saved artifact (fit on all data below);
# for a rigorous, leakage-free view use Model_gens/evaluate_fundamentals.py.
X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
_pred = clone(model).fit(X_tr, y_tr).predict(X_te)
print(
    f"[holdout] R2={r2_score(y_te, _pred):.3f} "
    f"MAE={mean_absolute_error(y_te, _pred):.3f}"
)

model.fit(X, Y)
joblib.dump(model, _SCRIPT_DIR / "rfr_stockfundamentalsscorer.pkl")
print("model dumped")
