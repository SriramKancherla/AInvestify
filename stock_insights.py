import argparse
import csv
import os
import subprocess
import sys
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Downgrade sklearn noise: estimators were fit on named DataFrames, but some code paths
# (and sklearn internals on certain versions) still pass ndarray and trigger this warning.
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "Model_gens"
DATA_DIR = REPO_ROOT / "datasets"

FINANCIALS_PATH = DATA_DIR / "financials_cleaned.csv"
TWEETS_PATH = DATA_DIR / "stock_tweets.csv"

# Must match the column selection used in the model training scripts.
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


def _clean_tweet(text: str) -> str:
    """
    Tweet cleaning logic copied to match Model_gens/sentiment_logreg.py.
    """
    import re

    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"[^a-z\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _is_symbol_like(query: str) -> bool:
    """
    Heuristic: if the user types something that looks like a ticker symbol
    (no spaces, short, alnum plus . or -), treat it as a symbol.
    """
    import re

    q = (query or "").strip()
    if not q or " " in q:
        return False
    if len(q) > 8:
        return False
    return re.fullmatch(r"[A-Za-z0-9\.\-]+", q) is not None


def _resolve_symbol_from_df(
    df: pd.DataFrame,
    query: str,
    symbol_col: str,
    name_col: str,
) -> tuple[str | None, str | None]:
    """
    Resolve user input to a symbol that exists in the given dataframe.

    - If input looks like a ticker symbol, we only accept exact matches.
    - If input looks like a company name, we try contains match, then fuzzy.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Input cannot be empty.")

    if _is_symbol_like(q):
        sym = q.upper()
        symbols = df[symbol_col].astype(str).str.upper()
        mask = symbols == sym
        if mask.any():
            row = df.loc[mask].iloc[0]
            return sym, str(row[name_col])
        return None, None

    # Name-like resolution.
    names = df[name_col].astype(str)
    q_low = q.lower()
    contains_mask = names.str.lower().str.contains(q_low, na=False)
    if contains_mask.any():
        row = df.loc[contains_mask].iloc[0]
        return str(row[symbol_col]).upper(), str(row[name_col])

    # Fuzzy fallback.
    best_sym: str | None = None
    best_name: str | None = None
    best_score = -1.0
    for _, row in df.iterrows():
        name = str(row[name_col])
        score = SequenceMatcher(None, q_low, name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_sym = str(row[symbol_col]).upper()
            best_name = name

    return best_sym, best_name


def _truncate(s: str, max_len: int = 140) -> str:
    s = str(s).replace("\\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _score_to_label(score_0_1: float) -> str:
    if score_0_1 >= 0.66:
        return "Bullish"
    if score_0_1 >= 0.56:
        return "Mildly Bullish"
    if score_0_1 >= 0.44:
        return "Mixed / Uncertain"
    if score_0_1 >= 0.34:
        return "Mildly Bearish"
    return "Bearish"


def _clamp_01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _compute_confidence(
    *,
    fundamentals_score: float | None,
    fundamentals_out: dict | None,
    fundamentals_meta: dict,
    sentiment_score: float | None,
    sentiment_out: dict,
    sentiment_weight: float,
) -> dict:
    # Fundamentals confidence.
    if fundamentals_score is None:
        fundamentals_conf = 0.0
    else:
        source = fundamentals_meta.get("source")
        model_scores = (fundamentals_out or {}).get("model_scores") or {}
        vals = [float(v) for v in model_scores.values()] if model_scores else [float(fundamentals_score)]
        spread = float(np.std(vals)) if vals else 0.0  # lower spread => higher confidence
        spread_conf = _clamp_01(1.0 - min(1.0, spread / 0.22))

        if source == "dataset":
            source_conf = 0.86
        elif source == "live_yfinance":
            raw_count = int(fundamentals_meta.get("raw_available_count") or 0)
            imputed_count = len(fundamentals_meta.get("imputed_fields") or [])
            source_conf = 0.45 + 0.03 * raw_count - 0.02 * imputed_count
        else:
            source_conf = 0.35

        fundamentals_conf = 0.55 * source_conf + 0.45 * spread_conf
    fundamentals_conf = _clamp_01(float(fundamentals_conf))

    # Sentiment confidence.
    if sentiment_score is None:
        sentiment_conf = 0.0
    else:
        item_count = int(sentiment_out.get("item_count") or 0)
        confidence_meta = sentiment_out.get("confidence_meta") or {}
        dispersion = float(confidence_meta.get("dispersion") or 0.0)  # lower better
        balance = float(confidence_meta.get("balance") or 0.0)  # lower better
        strength = float(confidence_meta.get("strength") or 0.0)  # higher better

        count_conf = _clamp_01(min(item_count, 20) / 20.0)
        dispersion_conf = _clamp_01(1.0 - min(1.0, dispersion / 0.25))
        # Make one-sided distributions a caution signal, not a hard collapse.
        balance_conf = _clamp_01(1.0 - 0.55 * min(1.0, balance))
        strength_conf = _clamp_01(min(1.0, strength / 0.35))

        # Prevent overconfidence when sentiment is one-sided.
        # Apply only a mild penalty and mostly when confidence strength is weak.
        one_sided_penalty = 0.0
        if balance >= 0.9 and strength < 0.18:
            one_sided_penalty = 0.10
        elif balance >= 0.85 and strength < 0.12:
            one_sided_penalty = 0.06

        source_bonus = 0.07 if sentiment_out.get("source") == "dataset_tweets" else 0.0
        sentiment_conf = (
            0.32 * count_conf
            + 0.25 * dispersion_conf
            + 0.18 * balance_conf
            + 0.25 * strength_conf
            + source_bonus
            - one_sided_penalty
        )
        sentiment_conf = _clamp_01(float(sentiment_conf))

    # Overall confidence uses same blend logic as final score.
    if fundamentals_score is None:
        overall_conf = sentiment_conf
    elif sentiment_score is None:
        overall_conf = fundamentals_conf
    else:
        fw = 1.0 - sentiment_weight
        overall_conf = fw * fundamentals_conf + sentiment_weight * sentiment_conf

    return {
        "overall": _clamp_01(float(overall_conf)),
        "fundamentals": fundamentals_conf,
        "sentiment": sentiment_conf,
    }


def _calibrate_final_score(raw_score: float, overall_confidence: float) -> float:
    # Low confidence -> shrink toward neutral (0.5). High confidence -> preserve signal.
    shrink = 0.72 + 0.28 * _clamp_01(float(overall_confidence))
    calibrated = 0.5 + (float(raw_score) - 0.5) * shrink
    return _clamp_01(calibrated)


def _maybe_train_models(train_missing: bool) -> None:
    """
    Training is expensive for the Keras model (custom_nn_scorer.py uses 500 epochs).
    By default, we do not train; we only train if --train-missing is provided.
    """
    required_artifacts = {
        "fundamentals_stock_model.joblib": ["stockfundamentalanalysis.py"],
        "rfr_stockfundamentalsscorer.pkl": ["rfr_fundamentals_scorer.py"],
        "stock_score_regression.pkl": ["stockscoreregression.py"],
        "sentiment_logreg.pkl": ["sentiment_logreg.py"],
        "tfidf_vectorizer.pkl": ["sentiment_logreg.py"],
        "keras_stockfundamentalsscorer.h5": ["custom_nn_scorer.py"],
        "keras_X_scaler.pkl": ["custom_nn_scorer.py"],
        "keras_Y_scaler.pkl": ["custom_nn_scorer.py"],
    }

    missing_by_script: dict[str, list[str]] = {}
    missing_any = False
    for artifact, scripts in required_artifacts.items():
        artifact_path = MODEL_DIR / artifact
        if not artifact_path.exists():
            missing_any = True
            for script in scripts:
                missing_by_script.setdefault(script, []).append(artifact)

    if not missing_any:
        return

    if not train_missing:
        missing = [a for a in required_artifacts.keys() if not (MODEL_DIR / a).exists()]
        raise FileNotFoundError(
            "Model artifacts are missing and --train-missing was not provided. "
            f"Missing: {', '.join(missing)}. "
            "Run with --train-missing to generate them (Keras training is very slow)."
        )

    for script, missing_artifacts in missing_by_script.items():
        print(f"[train] Running {script} (needed: {', '.join(sorted(set(missing_artifacts)))})")
        script_path = MODEL_DIR / script
        if not script_path.exists():
            raise FileNotFoundError(f"Training script not found: {script_path}")
        subprocess.check_call([sys.executable, str(script_path.name)], cwd=str(MODEL_DIR))


def _load_models():
    fundamentals_clf = joblib.load(MODEL_DIR / "fundamentals_stock_model.joblib")
    fundamentals_rfr = joblib.load(MODEL_DIR / "rfr_stockfundamentalsscorer.pkl")
    fundamentals_xgb = joblib.load(MODEL_DIR / "stock_score_regression.pkl")

    sentiment_model = joblib.load(MODEL_DIR / "sentiment_logreg.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

    keras_model = None
    keras_X_scaler = joblib.load(MODEL_DIR / "keras_X_scaler.pkl")
    keras_Y_scaler = joblib.load(MODEL_DIR / "keras_Y_scaler.pkl")
    # Import tensorflow lazily so the program can still run without Keras if artifacts are absent.
    from tensorflow.keras.models import load_model  # type: ignore

    # Keras 3+ can fail when loading legacy H5 models that include compile config.
    # We only need `predict()`, so skip deserializing loss/metrics.
    keras_model = load_model(MODEL_DIR / "keras_stockfundamentalsscorer.h5", compile=False)

    return {
        "fundamentals_clf": fundamentals_clf,
        "fundamentals_rfr": fundamentals_rfr,
        "fundamentals_xgb": fundamentals_xgb,
        "sentiment_model": sentiment_model,
        "vectorizer": vectorizer,
        "keras_model": keras_model,
        "keras_X_scaler": keras_X_scaler,
        "keras_Y_scaler": keras_Y_scaler,
    }


def _predict_fundamentals(models: dict, financials_row: pd.Series) -> dict:
    # Models were saved from pipelines trained on named DataFrame columns; pass a DataFrame
    # so sklearn does not emit "X does not have valid feature names" on every request.
    X_df = pd.DataFrame(
        [financials_row[FEATURE_COLS].astype(float).to_numpy()],
        columns=FEATURE_COLS,
    )

    # Classifier: output probability for class 1.
    clf = models["fundamentals_clf"]
    proba = clf.predict_proba(X_df)[0]
    if 1 in clf.classes_:
        pos_idx = list(clf.classes_).index(1)
    else:
        # Fallback: assume the second class is "good".
        pos_idx = 1 if len(clf.classes_) > 1 else 0
    clf_score = float(proba[pos_idx])

    # Regressors: already trained on a [0,1] style "fundamental_score".
    rfr_score = float(models["fundamentals_rfr"].predict(X_df)[0])
    xgb_score = float(models["fundamentals_xgb"].predict(X_df)[0])

    # Keras NN: works on scaled X; Y_scaled is inverted back to [0,1].
    X_scaled = models["keras_X_scaler"].transform(X_df)
    y_scaled_pred = float(models["keras_model"].predict(X_scaled, verbose=0)[0][0])
    y_pred = float(models["keras_Y_scaler"].inverse_transform(np.array([[y_scaled_pred]]))[0][0])

    # Keep everything in [0,1] so the ensemble is comparable.
    def clip_01(v: float) -> float:
        return float(min(1.0, max(0.0, v)))

    clf_score = clip_01(clf_score)
    rfr_score = clip_01(rfr_score)
    xgb_score = clip_01(xgb_score)
    y_pred = clip_01(y_pred)

    fundamentals_score = float(np.mean([clf_score, rfr_score, xgb_score, y_pred]))
    return {
        "fundamentals_score": fundamentals_score,
        "model_scores": {
            "fundamentals_classifier_p1": clf_score,
            "random_forest_regressor": rfr_score,
            "xgb_regressor": xgb_score,
            "keras_nn_regressor": y_pred,
        },
    }


def _predict_sentiment(models: dict, tweets: pd.DataFrame, top_n_examples: int) -> dict:
    if tweets.empty:
        return {
            "sentiment_score": None,
            "source": "dataset_tweets",
            "item_label": "Tweets",
            "item_count": 0,
            "positive_item_count": 0,
            "negative_item_count": 0,
            # Backwards-compatible keys (used by earlier versions of the CLI).
            "tweet_count": 0,
            "positive_tweet_count": 0,
            "negative_tweet_count": 0,
            "top_positive_examples": [],
            "top_negative_examples": [],
        }

    vectorizer = models["vectorizer"]
    sentiment_model = models["sentiment_model"]

    cleaned = tweets["Tweet"].apply(_clean_tweet)
    X_tfidf = vectorizer.transform(cleaned)
    proba = sentiment_model.predict_proba(X_tfidf)
    if 1 in sentiment_model.classes_:
        pos_idx = list(sentiment_model.classes_).index(1)
    else:
        pos_idx = 1 if proba.shape[1] > 1 else 0

    pos_prob = proba[:, pos_idx]
    sentiment_score = float(np.mean(pos_prob))
    dispersion = float(np.std(pos_prob))
    strength = float(np.mean(np.abs(pos_prob - 0.5)))

    pred_label = (pos_prob >= 0.5).astype(int)
    positive_tweet_count = int(pred_label.sum())
    negative_tweet_count = int(len(pred_label) - positive_tweet_count)
    denom = max(1, positive_tweet_count + negative_tweet_count)
    balance = abs(positive_tweet_count - negative_tweet_count) / denom

    # Pick examples for user interpretability.
    order_pos = np.argsort(-pos_prob)
    order_neg = np.argsort(pos_prob)

    def take_examples(order_idx: np.ndarray) -> list[str]:
        examples = []
        for i in order_idx[:top_n_examples]:
            examples.append(_truncate(tweets.iloc[i]["Tweet"], 160))
        return examples

    return {
        "sentiment_score": sentiment_score,
        "source": "dataset_tweets",
        "item_label": "Tweets",
        "item_count": int(len(tweets)),
        "positive_item_count": positive_tweet_count,
        "negative_item_count": negative_tweet_count,
        # Backwards-compatible keys (used by earlier versions of the CLI).
        "tweet_count": int(len(tweets)),
        "positive_tweet_count": positive_tweet_count,
        "negative_tweet_count": negative_tweet_count,
        "top_positive_examples": take_examples(order_pos),
        "top_negative_examples": take_examples(order_neg),
        "confidence_meta": {
            "dispersion": dispersion,
            "balance": float(balance),
            "strength": strength,
        },
    }


def _predict_sentiment_from_news_items(
    models: dict, news_items: list[dict], top_n_examples: int
) -> dict:
    if not news_items:
        return {
            "sentiment_score": None,
            "source": "live_news",
            "item_label": "News",
            "item_count": 0,
            "positive_item_count": 0,
            "negative_item_count": 0,
            "top_positive_examples": [],
            "top_negative_examples": [],
        }

    vectorizer = models["vectorizer"]
    sentiment_model = models["sentiment_model"]

    texts = [str(item.get("text_for_model", "")) for item in news_items]
    X_tfidf = vectorizer.transform(texts)
    proba = sentiment_model.predict_proba(X_tfidf)
    if 1 in sentiment_model.classes_:
        pos_idx = list(sentiment_model.classes_).index(1)
    else:
        pos_idx = 1 if proba.shape[1] > 1 else 0

    pos_prob = proba[:, pos_idx]
    sentiment_score = float(np.mean(pos_prob))
    dispersion = float(np.std(pos_prob))
    strength = float(np.mean(np.abs(pos_prob - 0.5)))

    pred_label = (pos_prob >= 0.5).astype(int)
    positive_count = int(pred_label.sum())
    negative_count = int(len(pred_label) - positive_count)
    denom = max(1, positive_count + negative_count)
    balance = abs(positive_count - negative_count) / denom

    order_pos = np.argsort(-pos_prob)
    order_neg = np.argsort(pos_prob)

    def format_item(idx: int) -> str:
        item = news_items[int(idx)]
        # User-facing output: headlines only.
        # We intentionally omit links/summaries here; they can be used later for the website.
        title = str(item.get("title", "")).strip()
        return _truncate(title, 160)

    def take_examples(order_idx: np.ndarray) -> list[str]:
        examples: list[str] = []
        for i in order_idx[:top_n_examples]:
            examples.append(format_item(int(i)))
        return examples

    return {
        "sentiment_score": sentiment_score,
        "source": "live_news",
        "item_label": "News",
        "item_count": int(len(news_items)),
        "positive_item_count": positive_count,
        "negative_item_count": negative_count,
        "top_positive_examples": take_examples(order_pos),
        "top_negative_examples": take_examples(order_neg),
        "confidence_meta": {
            "dispersion": dispersion,
            "balance": float(balance),
            "strength": strength,
        },
    }


def _fetch_live_news_items(news_query: str, max_articles: int) -> list[dict]:
    """
    Fetch recent news from Google News RSS.
    Returns structured items from `Model_gens/webscrapernews.py`.
    """
    try:
        from Model_gens.webscrapernews import scrape_google_news_rss  # type: ignore

        return scrape_google_news_rss(news_query, max_articles=max_articles)
    except Exception as e:
        print(f"[warn] Live news fetch failed: {e}")
        return []


def _fetch_live_fundamentals_row(symbol: str, financials: pd.DataFrame) -> tuple[pd.Series | None, dict]:
    """
    Fetch fundamentals from Yahoo Finance and map to model feature columns.
    Missing live values are median-imputed from the training dataset.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        return None, {"source": "none", "error": f"yfinance import failed: {e}"}

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception as e:
        return None, {"source": "none", "error": f"yfinance fetch failed: {e}"}

    # Price fallback order for stability.
    live_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or info.get("open")
    )

    raw_live = {
        "Market_Cap": info.get("marketCap"),
        "Price": live_price,
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "Book_Value": info.get("bookValue"),
        "Price/Earnings": info.get("trailingPE"),
        "Dividend_Yield": info.get("dividendYield"),
        "EBITDA": info.get("ebitda"),
        "Price/Sales": info.get("priceToSalesTrailing12Months"),
        "Price/Book": info.get("priceToBook"),
    }

    # yfinance dividendYield is usually decimal (0.02), while training data is often percentage (2.0).
    dy = raw_live["Dividend_Yield"]
    if dy is not None:
        try:
            dy_f = float(dy)
            raw_live["Dividend_Yield"] = dy_f * 100.0 if dy_f <= 1 else dy_f
        except Exception:
            raw_live["Dividend_Yield"] = None

    # How much real data we got from Yahoo before any imputation.
    raw_available_count = 0
    missing_fields: list[str] = []
    for k in FEATURE_COLS:
        v = raw_live.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            missing_fields.append(k)
        else:
            raw_available_count += 1

    # If Yahoo gives almost nothing, avoid producing a score from mostly imputed values.
    if raw_available_count < 4:
        return None, {
            "source": "none",
            "error": "insufficient live fundamentals from yfinance",
            "raw_available_count": raw_available_count,
            "missing_fields": missing_fields,
        }

    medians = (
        financials[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .median(numeric_only=True)
        .to_dict()
    )
    row_values: dict[str, float] = {}
    imputed_fields: list[str] = []
    for col in FEATURE_COLS:
        val = raw_live.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            row_values[col] = float(medians[col])
            imputed_fields.append(col)
        else:
            row_values[col] = float(val)

    company_name = info.get("longName") or info.get("shortName") or symbol
    return pd.Series(row_values), {
        "source": "live_yfinance",
        "symbol": symbol.upper(),
        "company": str(company_name),
        "raw_available_count": raw_available_count,
        "missing_fields": missing_fields,
        "imputed_fields": imputed_fields,
    }


# Cache loaded models so backend requests are fast.
_MODELS_CACHE: dict | None = None


def _load_models_cached(*, force_reload: bool = False) -> dict:
    global _MODELS_CACHE
    if force_reload or _MODELS_CACHE is None:
        _MODELS_CACHE = _load_models()
    return _MODELS_CACHE


def get_insights(
    input_value: str,
    *,
    train_missing: bool = False,
    top_items: int = 3,
    sentiment_weight: float = 0.3,
    news_source: str = "auto",
    max_news: int = 10,
) -> dict:
    """
    Core inference function for the website/backend.

    Returns a JSON-serializable dict with:
    - final score + label
    - fundamentals score (+ sub model scores if available)
    - sentiment score + top positive/negative items
    """
    if not input_value:
        raise ValueError("Input cannot be empty.")
    if not (0.0 <= sentiment_weight <= 1.0):
        raise ValueError("sentiment_weight must be between 0 and 1.")

    # Load datasets.
    if not FINANCIALS_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {FINANCIALS_PATH}")
    if not TWEETS_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {TWEETS_PATH}")

    financials = pd.read_csv(FINANCIALS_PATH)
    tweets_all = pd.read_csv(
        TWEETS_PATH,
        sep="\t",
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )

    financial_symbol, financial_company_name = _resolve_symbol_from_df(
        financials, input_value, symbol_col="Symbol", name_col="Name"
    )
    tweet_symbol, tweet_company_name = _resolve_symbol_from_df(
        tweets_all, input_value, symbol_col="Stock Name", name_col="Company Name"
    )

    if tweet_symbol is not None:
        tweets = tweets_all.loc[tweets_all["Stock Name"].astype(str) == tweet_symbol].copy()
    else:
        tweets = tweets_all.iloc[0:0].copy()

    _maybe_train_models(train_missing=train_missing)
    models = _load_models_cached(force_reload=train_missing)

    fundamentals_out = None
    fundamentals_score: float | None = None
    fundamentals_meta: dict = {"source": "none"}
    if financial_symbol is not None:
        fin_mask = financials["Symbol"].astype(str).str.upper() == financial_symbol
        if fin_mask.any():
            fin_row = financials.loc[fin_mask].iloc[0].copy()
            fundamentals_out = _predict_fundamentals(models, fin_row)
            fundamentals_score = float(fundamentals_out["fundamentals_score"])
            fundamentals_meta = {"source": "dataset"}
    else:
        live_symbol_candidate = None
        if _is_symbol_like(input_value):
            live_symbol_candidate = input_value.strip().upper()
        elif tweet_symbol is not None:
            live_symbol_candidate = tweet_symbol

        if live_symbol_candidate is not None:
            live_fin_row, live_meta = _fetch_live_fundamentals_row(live_symbol_candidate, financials)
            if live_fin_row is not None:
                fundamentals_out = _predict_fundamentals(models, live_fin_row)
                fundamentals_score = float(fundamentals_out["fundamentals_score"])
                financial_symbol = live_meta.get("symbol")
                financial_company_name = live_meta.get("company")
            fundamentals_meta = live_meta

    sentiment_out_dataset = _predict_sentiment(models, tweets, top_n_examples=top_items)
    sentiment_out = sentiment_out_dataset
    sentiment_score = sentiment_out_dataset["sentiment_score"]
    news_query_used = None
    live_news_attempted = False
    live_news_item_count: int | None = None

    fetch_live = news_source == "live" or (news_source == "auto" and sentiment_score is None)
    if fetch_live:
        live_news_attempted = True
        news_query_used = tweet_company_name or financial_company_name or input_value
        news_items = _fetch_live_news_items(news_query_used, max_articles=max_news)
        live_news_item_count = len(news_items)
        sentiment_out_live = _predict_sentiment_from_news_items(
            models, news_items, top_n_examples=top_items
        )
        if sentiment_out_live["sentiment_score"] is not None:
            sentiment_out = sentiment_out_live
            sentiment_score = sentiment_out["sentiment_score"]

    if fundamentals_score is None and sentiment_score is None:
        raise ValueError(
            "Could not resolve the input to any available data in this repo. "
            "Try a different stock symbol/company name."
        )

    if fundamentals_score is None:
        final_score_raw = float(sentiment_score)
    elif sentiment_score is None:
        final_score_raw = float(fundamentals_score)
    else:
        fw = 1.0 - sentiment_weight
        final_score_raw = fw * float(fundamentals_score) + sentiment_weight * float(sentiment_score)

    confidence = _compute_confidence(
        fundamentals_score=fundamentals_score,
        fundamentals_out=fundamentals_out,
        fundamentals_meta=fundamentals_meta,
        sentiment_score=sentiment_score,
        sentiment_out=sentiment_out,
        sentiment_weight=sentiment_weight,
    )
    final_score = _calibrate_final_score(final_score_raw, confidence["overall"])
    label = _score_to_label(final_score)

    return {
        "input": input_value,
        "resolved": {
            "fundamentals": {
                "symbol": financial_symbol,
                "company": financial_company_name,
            },
            "sentiment": {
                "symbol": tweet_symbol,
                "company": tweet_company_name,
            },
        },
        "final": {
            "score": float(final_score),
            "raw_score": float(final_score_raw),
            "label": label,
            "confidence": confidence,
        },
        "fundamentals": {
            "score": fundamentals_score,
            "model_scores": fundamentals_out["model_scores"] if fundamentals_out is not None else None,
            "source": fundamentals_meta.get("source"),
            "meta": fundamentals_meta,
        },
        "sentiment": {
            "source": sentiment_out.get("source"),
            "score": sentiment_score,
            "item_label": sentiment_out.get("item_label"),
            "item_count": sentiment_out.get("item_count"),
            "positive_item_count": sentiment_out.get("positive_item_count"),
            "negative_item_count": sentiment_out.get("negative_item_count"),
            "top_positive_examples": sentiment_out.get("top_positive_examples"),
            "top_negative_examples": sentiment_out.get("top_negative_examples"),
        },
        "meta": {
            "news_source": news_source,
            "live_news_attempted": live_news_attempted,
            "live_news_item_count": live_news_item_count,
            "news_query_used": news_query_used,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Stock insights: final score + news sentiment")
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help="Stock symbol (e.g., TSLA) or company name (e.g., Tesla, Inc.)",
    )
    parser.add_argument(
        "--train-missing",
        action="store_true",
        help="If model artifacts are missing, train them (can be slow, especially Keras).",
    )
    parser.add_argument(
        "--top-tweets",
        type=int,
        default=3,
        help="How many top positive/negative tweet examples to display.",
    )
    parser.add_argument(
        "--sentiment-weight",
        type=float,
        default=0.3,
        help="Weight of news sentiment in final score (0-1). Fundamentals weight = 1 - sentiment-weight.",
    )
    parser.add_argument(
        "--news-source",
        choices=["auto", "dataset", "live"],
        default="auto",
        help="Sentiment source: dataset tweets, live scraped news, or auto fallback.",
    )
    parser.add_argument(
        "--max-news",
        type=int,
        default=10,
        help="Max number of live news items to fetch (when using --news-source live/auto).",
    )
    args = parser.parse_args()

    if not args.input:
        # Interactive prompt mode (useful when later embedding in a website).
        args.input = input("Enter stock symbol or company name: ").strip()
    if not args.input:
        raise ValueError("No input provided.")

    if not (0.0 <= args.sentiment_weight <= 1.0):
        raise ValueError("--sentiment-weight must be between 0 and 1.")
    insights = get_insights(
        args.input,
        train_missing=args.train_missing,
        top_items=args.top_tweets,
        sentiment_weight=args.sentiment_weight,
        news_source=args.news_source,
        max_news=args.max_news,
    )

    financial_symbol = insights["resolved"]["fundamentals"]["symbol"]
    financial_company_name = insights["resolved"]["fundamentals"]["company"]
    tweet_symbol = insights["resolved"]["sentiment"]["symbol"]
    tweet_company_name = insights["resolved"]["sentiment"]["company"]

    sentiment_out = insights["sentiment"]
    sentiment_score = sentiment_out["score"]

    fundamentals_score = insights["fundamentals"]["score"]
    if insights["fundamentals"]["model_scores"] is not None:
        fundamentals_out = {"model_scores": insights["fundamentals"]["model_scores"]}
    else:
        fundamentals_out = None

    final_score = insights["final"]["score"]
    label = insights["final"]["label"]

    news_query_used = insights["meta"]["news_query_used"]
    live_news_attempted = insights["meta"]["live_news_attempted"]
    live_news_item_count = insights["meta"]["live_news_item_count"]

    print("=== Stock Insights ===")
    print(f"Input: {args.input}")
    if financial_symbol is None:
        print("Resolved Fundamentals: N/A (not present in financials dataset)")
    else:
        print(f"Resolved Fundamentals: {financial_symbol} - {financial_company_name}")
    if sentiment_out.get("source") == "live_news":
        print(f"Resolved Sentiment: Live News (query: {news_query_used})")
    else:
        if tweet_symbol is None:
            print("Resolved Sentiment: N/A (not present in tweets dataset)")
        else:
            print(f"Resolved Sentiment: {tweet_symbol} - {tweet_company_name}")
    print("")
    print(f"Final Score: {final_score:.3f} / 1.000 ({label})")
    if fundamentals_score is None:
        print("Fundamentals Score: N/A")
    else:
        print(f"Fundamentals Score: {fundamentals_score:.3f}")
    if sentiment_score is None:
        if live_news_attempted and (live_news_item_count == 0):
            print("Sentiment Score: N/A (no live news items found)")
        else:
            item_label = sentiment_out.get("item_label", "Items")
            print(f"Sentiment Score: N/A (no suitable {item_label.lower()} found)")
    else:
        print(f"Sentiment Score: {float(sentiment_score):.3f}")
        print(
            f"{sentiment_out.get('item_label', 'Items')}: {sentiment_out.get('item_count', 0)} | Positive: {sentiment_out.get('positive_item_count', 0)} | Negative: {sentiment_out.get('negative_item_count', 0)}"
        )
        print("")
        print("Top Positive Examples:")
        for ex in sentiment_out["top_positive_examples"]:
            print(f"- {ex}")
        print("")
        print("Top Negative Examples:")
        for ex in sentiment_out["top_negative_examples"]:
            print(f"- {ex}")
    print("")
    if fundamentals_out is not None:
        print("Fundamentals Model Sub-scores:")
        for k, v in fundamentals_out["model_scores"].items():
            print(f"- {k}: {v:.3f}")


if __name__ == "__main__":
    main()

