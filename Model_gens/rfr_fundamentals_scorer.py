from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import numpy as np

data=pd.read_csv('../datasets/financials_cleaned.csv')
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


model=RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X,Y)
joblib.dump(model,'rfr_stockfundamentalsscorer.pkl')
print('model dumped')
