# AInvestify



# AInvestify

AInvestify is a stock insights web app built with **FastAPI + vanilla HTML/CSS/JS**.
It combines:

- Multi-model fundamentals scoring
- Sentiment analysis (dataset tweets + live news fallback)
- Price chart visualization
- Compare mode for 2 stocks
- AI chatbot (Gemini when configured, local fallback otherwise)

---

## Features

- Stock lookup by ticker/company from dropdown
- Final score with interpretation (fundamentals + sentiment blend)
- Live chart support with graceful fallback handling
- Compare mode (side-by-side KPI, chart, and sentiment examples)
- Confidence indicator for model-output reliability
- Light/Dark theme toggle
- Recent ticker chips (primary/compare), remove individual or clear all
- Branded UI (logo + favicon)
- Chatbot with quota-safe local fallback messaging

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **ML/Data:** scikit-learn, XGBoost, TensorFlow/Keras, pandas, numpy
- **Data sources:** local CSV datasets, Yahoo Finance (`yfinance`), Google News RSS (`feedparser`)
- **NLP:** Logistic Regression + TF-IDF + VADER
- **Frontend:** Vanilla HTML/CSS/JS + Chart.js

---

## Project Structure

```text
backend/                # FastAPI app + service layer
frontend/               # Single-page UI
Model_gens/             # Training/util scripts + saved model artifacts
datasets/               # Input datasets
stock_insights.py       # Core scoring and inference orchestration
tests/                  # API reliability tests
Dockerfile              # Container packaging
render.yaml             # Render Blueprint config
```

---

## Local Setup

### 1) Create and activate venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure env

```bash
cp .env.example .env
```

Set at least:

- `GEMINI_API_KEY` (optional, chatbot uses local fallback if missing)

### 4) Run app

```bash
.venv/bin/uvicorn backend.app:app --reload --port 8000
```

Open: [http://localhost:8000](http://localhost:8000)

---

## Environment Variables

### Core

- `GEMINI_API_KEY`
- `APP_ENV` (`development` / `production`)
- `ALLOWED_HOSTS` (comma-separated hostnames)
- `CORS_ALLOW_ORIGINS` (comma-separated allowed origins)

### Security/CORS

- `ENABLE_PROXY_HEADERS`
- `CORS_ALLOW_CREDENTIALS`
- `CORS_ALLOW_METHODS`
- `CORS_ALLOW_HEADERS`

### Reliability/Rate Limits

- `RATE_LIMIT_INSIGHTS_COUNT`
- `RATE_LIMIT_INSIGHTS_WINDOW_SECONDS`
- `RATE_LIMIT_CHATBOT_COUNT`
- `RATE_LIMIT_CHATBOT_WINDOW_SECONDS`
- `HTTP_RETRY_COUNT`
- `HTTP_RETRY_BACKOFF_MS`
- `GEMINI_TIMEOUT_SECONDS`

> Keep secrets in `.env` only. Do **not** commit `.env`.

---

## API Endpoints

- `GET /health`
- `GET /`
- `GET /tickers`
- `POST /api/insights`
- `GET /chart/{ticker}`
- `GET /fundamentals/{ticker}`
- `GET /news/{stock_name}`
- `POST /chatbot`
- `GET /chatbot/status`

---

## Tests

Run reliability API tests:

```bash
.venv/bin/python -m pytest -q tests/test_api.py
```

---

## Docker

### Build

```bash
docker build -t ainvestify:latest .
```

### Run

```bash
docker run --rm -p 8000:8000 --env-file .env ainvestify:latest
```

App: [http://localhost:8000](http://localhost:8000)

---

## Share with Others (Quick)

Temporary public link using Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

Send the generated `trycloudflare.com` URL.

---

## Permanent Deployment (Render)

This repo includes `render.yaml` for Blueprint deploy.

### Steps

1. Push project to GitHub
2. In Render: **New +** -> **Blueprint**
3. Select repo and apply
4. Set env vars in Render:
   - `ALLOWED_HOSTS=<your-service>.onrender.com`
   - `CORS_ALLOW_ORIGINS=https://<your-service>.onrender.com`
   - `GEMINI_API_KEY=<your_key>`
5. Deploy and verify:
   - `https://<your-service>.onrender.com/health`

---

## Notes

- Startup checks validate required datasets/model artifacts
- Favicon routes support both `GET` and `HEAD`
- In development, `*.trycloudflare.com` is allowed in host validation for quick sharing

---

## Creators

- **Sriram Kancherla**
  - LinkedIn: [https://www.linkedin.com/in/sriram-kancherla-80a7b028a/](https://www.linkedin.com/in/sriram-kancherla-80a7b028a/)
  - Gmail: [Kancherlasriram2006@gmail.com](mailto:Kancherlasriram2006@gmail.com)

- **Viswanath Parashuram Yadavalli**
  - LinkedIn: [https://www.linkedin.com/in/vishwa-yadavalli-65503628b/](https://www.linkedin.com/in/vishwa-yadavalli-65503628b/)
  - Gmail: [vishwanaathh4@gmail.com](mailto:vishwanaathh4@gmail.com)
