import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

from fundamentals_labels import FEATURE_COLS, good_fundamentals, prepare_numeric

_SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = _SCRIPT_DIR.parent / "datasets"

print("Welcome to fundamentals stock classification")

data = prepare_numeric(pd.read_csv(DATASETS_DIR / "financials_cleaned.csv", sep=","))

X = data[FEATURE_COLS]
Y = good_fundamentals(data)

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Held-out sanity check. Does NOT affect the saved artifact (fit on all data below);
# for a rigorous, leakage-free view use Model_gens/evaluate_fundamentals.py.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
_eval = clone(model).fit(X_tr, y_tr)
_proba = _eval.predict_proba(X_te)[:, list(_eval.classes_).index(1)]
print(
    f"[holdout] ROC-AUC={roc_auc_score(y_te, _proba):.3f} "
    f"accuracy={accuracy_score(y_te, (_proba >= 0.5).astype(int)):.3f}"
)

model.fit(X, Y)

joblib.dump(model, _SCRIPT_DIR / "fundamentals_stock_model.joblib")
print(" Model saved as fundamentals_stock_model.joblib")
