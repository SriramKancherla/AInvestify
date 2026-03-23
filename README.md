# AInvestify

AInvestify is a stock insights web app that combines:

- multi-model fundamentals scoring,
- sentiment analysis (dataset + live news),
- chart visualization,
- compare mode,
- and an AI chatbot.

It is built with **FastAPI (backend)** and **vanilla HTML/CSS/JS (frontend)**.

---

## 1) Project Structure

```text
backend/                    # FastAPI routes + service layer
frontend/                   # Web UI (single page)
Model_gens/                 # Model training scripts + helper scripts
datasets/                   # Input datasets
stock_insights.py           # Core scoring/inference orchestration
tests/                      # API reliability tests
Dockerfile                  # Container build
render.yaml                 # Render Blueprint deploy config
requirements.txt            # Python dependencies
```

---

## 2) Prerequisites

- Python 3.10+ (recommended 3.11)
- pip
- (Optional) Docker
- (Optional) cloudflared (for temporary public sharing)

---

## 3) Setup

### 3.1 Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Configure environment variables

```bash
cp .env.example .env
```

Set values in `.env`:

- `GEMINI_API_KEY` (optional, chatbot falls back to local when missing or quota-hit)
- You can keep the rest as defaults for local dev.

---

## 4) Generate/Train Models (Important)

Run these scripts **from project root** in the order below.

### 4.1 Fundamentals classifier

```bash
.venv/bin/python Model_gens/stockfundamentalanalysis.py
```

Expected output file:

- `Model_gens/fundamentals_stock_model.joblib`

### 4.2 Fundamentals regressor (Random Forest)

```bash
.venv/bin/python Model_gens/rfr_fundamentals_scorer.py
```

Expected output file:

- `Model_gens/rfr_stockfundamentalsscorer.pkl`

### 4.3 Fundamentals regressor (XGBoost)

```bash
.venv/bin/python Model_gens/stockscoreregression.py
```

Expected output file:

- `Model_gens/stock_score_regression.pkl`

### 4.4 Sentiment model + vectorizer

```bash
.venv/bin/python Model_gens/sentiment_logreg.py
```

Expected output files:

- `Model_gens/sentiment_logreg.pkl`
- `Model_gens/tfidf_vectorizer.pkl`

### 4.5 Keras NN fundamentals scorer

```bash
.venv/bin/python Model_gens/custom_nn_scorer.py
```

Expected output files:

- `Model_gens/keras_stockfundamentalsscorer.h5`
- `Model_gens/keras_X_scaler.pkl`
- `Model_gens/keras_Y_scaler.pkl`

### 4.6 Verify all required artifacts exist

You should have these 8+ artifacts in `Model_gens/`:

- `fundamentals_stock_model.joblib`
- `rfr_stockfundamentalsscorer.pkl`
- `stock_score_regression.pkl`
- `sentiment_logreg.pkl`
- `tfidf_vectorizer.pkl`
- `keras_stockfundamentalsscorer.h5`
- `keras_X_scaler.pkl`
- `keras_Y_scaler.pkl`

If any file is missing, rerun that model script.

---

## 5) Run the App

```bash
.venv/bin/uvicorn backend.app:app --reload --port 8000
```

Open:

- `http://localhost:8000`

Health:

- `http://localhost:8000/health`

---

## 6) Core API Endpoints

- `GET /health`
- `GET /` (serves frontend)
- `GET /tickers`
- `POST /api/insights`
- `GET /chart/{ticker}`
- `GET /fundamentals/{ticker}`
- `GET /news/{stock_name}`
- `POST /chatbot`
- `GET /chatbot/status`

---

## 7) Testing

Run API reliability tests:

```bash
.venv/bin/python -m pytest -q tests/test_api.py
```

Expected:

- all tests pass

---

## 8) Environment Variables

### Required for full features

- `GEMINI_API_KEY` (optional but recommended)

### App/security

- `APP_ENV=development|production`
- `ALLOWED_HOSTS=<comma-separated-hosts>`
- `CORS_ALLOW_ORIGINS=<comma-separated-origins>`
- `ENABLE_PROXY_HEADERS=true|false`
- `CORS_ALLOW_CREDENTIALS=true|false`
- `CORS_ALLOW_METHODS=GET,POST,OPTIONS`
- `CORS_ALLOW_HEADERS=Content-Type,Authorization`

### Reliability/rate limits

- `RATE_LIMIT_INSIGHTS_COUNT`
- `RATE_LIMIT_INSIGHTS_WINDOW_SECONDS`
- `RATE_LIMIT_CHATBOT_COUNT`
- `RATE_LIMIT_CHATBOT_WINDOW_SECONDS`
- `HTTP_RETRY_COUNT`
- `HTTP_RETRY_BACKOFF_MS`
- `GEMINI_TIMEOUT_SECONDS`

> Never commit `.env` to GitHub. Keep secrets in `.env` (local) or provider dashboard (production).

---

## 9) Docker (Local Container Run)

### Build

```bash
docker build -t ainvestify:latest .
```

### Run

```bash
docker run --rm -p 8000:8000 --env-file .env ainvestify:latest
```

Open:

- `http://localhost:8000`

---

## 10) Share Website With Others

### 10.1 Temporary public URL (quickest)

```bash
cloudflared tunnel --url http://localhost:8000
```

Send the generated `trycloudflare.com` link.

### 10.2 Same Wi-Fi sharing

```bash
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Share:

- `http://<your-local-ip>:8000`

---

## 11) Permanent Deployment (Render)

This repo includes `render.yaml` for Blueprint deployment.

### Steps

1. Push your full repo to GitHub.
2. In Render: **New + -> Blueprint**.
3. Select repository and apply.
4. Set env vars in Render:
   - `ALLOWED_HOSTS=<your-service>.onrender.com`
   - `CORS_ALLOW_ORIGINS=https://<your-service>.onrender.com`
   - `GEMINI_API_KEY=<your_key>`
5. Deploy.
6. Verify:
   - `https://<your-service>.onrender.com/health` -> `{"status":"ok"}`
7. Share:
   - `https://<your-service>.onrender.com`

---

## 12) Troubleshooting

### Invalid host header

- Ensure `ALLOWED_HOSTS` includes your deployment host.
- For quick tunnels, dev default includes `*.trycloudflare.com`.

### Chart unavailable

- App still shows fundamentals/sentiment; this is expected graceful fallback behavior.

### Gemini unavailable / quota exceeded

- Chatbot auto-falls back to local assistant.

### Missing model artifacts at startup

- Run missing training scripts from Section 4.

---

## 13) Creators

- **Sriram Kancherla**
  - LinkedIn: [https://www.linkedin.com/in/sriram-kancherla-80a7b028a/](https://www.linkedin.com/in/sriram-kancherla-80a7b028a/)
  - Gmail: [Kancherlasriram2006@gmail.com](mailto:Kancherlasriram2006@gmail.com)

- **Viswanath Parashuram Yadavalli**
  - LinkedIn: [https://www.linkedin.com/in/vishwa-yadavalli-65503628b/](https://www.linkedin.com/in/vishwa-yadavalli-65503628b/)
  - Gmail: [vishwanaathh4@gmail.com](mailto:vishwanaathh4@gmail.com)
