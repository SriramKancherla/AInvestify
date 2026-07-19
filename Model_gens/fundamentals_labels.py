"""Single source of truth for fundamentals features and labels.

Previously the criteria, the graded ``fundamental_score`` and the classifier's
``good_fundamentals`` label were copy-pasted across four training scripts, and the
two labels had drifted into *different* definitions of "good" (the classifier used a
strict 6-way AND with ``Price/Book < 3`` promoted to a mandatory term, while the
regressors used 5 graded OR-based terms). That inconsistency is why the classifier
predicted a much rarer target than the regressors it was averaged with.

All fundamentals training scripts now import from here so the definition can only
live in one place. The classifier label is defined as a threshold on the graded
score, so all four models measure the same underlying concept.
"""
from __future__ import annotations

import pandas as pd

# Order MUST match FEATURE_COLS in stock_insights.py (inference feeds columns positionally by name).
FEATURE_COLS = [
    "Market_Cap",
    "Price",
    "52w_high",
    "52w_low",
    "Book_Value",
    "Price/Earnings",
    "Dividend_Yield",
    "EBITDA",
    "Price/Sales",
    "Price/Book",
]

# Columns coerced to numeric before scoring.
NUMERIC_COLS = [
    "Price", "Price/Earnings", "Dividend_Yield", "52w_low", "52w_high",
    "Market_Cap", "EBITDA", "Price/Sales", "Price/Book", "Book_Value",
]

# The graded score awards 0.20 per satisfied criterion (5 criteria => score in [0, 1]).
# A stock is labelled "good" when it meets at least 4 of the 5 criteria (score >= 0.8).
# On the current dataset this yields ~35% positives (vs. ~1% under the old AND label),
# giving the classifier a balanced, learnable target aligned with the regressors.
GOOD_FUNDAMENTALS_THRESHOLD = 0.8


def prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the numeric feature columns and drop rows with missing values."""
    df = df.copy()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=NUMERIC_COLS).reset_index(drop=True)


def _criteria(df: pd.DataFrame) -> list[pd.Series]:
    """The 5 boolean fundamentals criteria (shared by score and classifier label)."""
    selling_zone = ((df["52w_high"] - df["Price"]) / df["52w_high"]) <= 0.10
    ebitda_to_mcap = df["EBITDA"] / df["Market_Cap"]
    return [
        df["Price/Earnings"].between(10, 25),
        df["Market_Cap"] >= 10_000_000_000,
        ebitda_to_mcap.between(0.05, 0.15),
        (df["Price/Sales"] < 1) | (df["Price/Sales"].between(1, 2)),
        (df["Dividend_Yield"] > 3.5) | selling_zone | (df["Price/Book"] < 3),
    ]


def fundamental_score(df: pd.DataFrame) -> pd.Series:
    """Graded fundamentals score in [0, 1] (regressor target)."""
    score = sum(0.20 * c.astype(int) for c in _criteria(df))
    return score.rename("fundamental_score")


def good_fundamentals(df: pd.DataFrame, threshold: float = GOOD_FUNDAMENTALS_THRESHOLD) -> pd.Series:
    """Binary classifier target: graded score at or above ``threshold``."""
    return (fundamental_score(df) >= threshold).astype(int).rename("good_fundamentals")
