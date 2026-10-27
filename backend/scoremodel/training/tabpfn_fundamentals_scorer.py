# Carried from scoremodel_new (AInvestify-main/AI_PART/Model_gens/tabpfn_fundamentals_scorer.py)
# as-is, except the data-read path was fixed for this script's new location at
# backend/scoremodel/training/ (run with cwd=this directory).
#
# NOT part of the live serving ensemble: scoremodel_new shipped no pre-trained
# tabpfn_fundamentals_scorer.pkl artifact, and its own working_scripts/main.py never
# loads/calls this model either - so engine.py's _predict_fundamentals (4-model
# ensemble: RF classifier, RF regressor, XGB regressor, Keras NN) does not reference
# it. Kept here for reference / future use. Requires the `tabpfn` package, which is
# NOT in requirements.txt since nothing currently imports it at serving time - add it
# yourself if you want to train and wire this in as a 5th ensemble member.
import pandas as pd
import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Training TabPFN fundamentals scorer")

data = pd.read_csv("../../../data/financials_cleaned.csv", sep=",")

numeric_cols = [
    "Price", "Price/Earnings", "Dividend_Yield", "52w_low", "52w_high",
    "Market_Cap", "EBITDA", "Price/Sales", "Price/Book", "Book_Value"
]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
data = data.dropna(subset=numeric_cols)

data["selling_zone"] = np.where(
    ((data["52w_high"] - data["Price"]) / data["52w_high"]) <= 0.10,
    1, 0
)
data["ebitda_to_mcap"] = data["EBITDA"] / data["Market_Cap"]

data["fundamental_score"] = (
    0.20 * (data["Price/Earnings"].between(10, 25)).astype(int) +
    0.20 * (data["Market_Cap"] >= 10000000000).astype(int) +
    0.20 * (data["ebitda_to_mcap"].between(0.05, 0.15)).astype(int) +
    0.20 * ((data["Price/Sales"] < 1) | (data["Price/Sales"].between(1, 2))).astype(int) +
    0.20 * (
        (data["Dividend_Yield"] > 3.5) |
        (data["selling_zone"] == 1) |
        (data["Price/Book"] < 3)
    ).astype(int)
)

X = data[
    [
        "Market_Cap",
        "Price",
        "52w_high",
        "52w_low",
        "Book_Value",
        "Price/Earnings",
        "Dividend_Yield",
        "EBITDA",
        "Price/Sales",
        "Price/Book"
    ]
]
Y = data["fundamental_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = TabPFNRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"TabPFN — MSE: {mse:.4f} | R2: {r2:.4f}")

joblib.dump(model, "../artifacts/tabpfn_fundamentals_scorer.pkl")
print("Model saved as tabpfn_fundamentals_scorer.pkl")
