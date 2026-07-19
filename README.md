# AInvestify

AInvestify is a stock insights web app that combines:

- multi-model fundamentals scoring (Random Forest classifier + Random Forest regressor + XGBoost ensemble),
- sentiment analysis (dataset tweets + live Google News, blended logistic regression + VADER),
- interactive chart visualization,
- compare mode,
- per-user portfolio & watchlists (Supabase-backed),
- PDF report export,
- and an AI chatbot (Gemini with automatic local fallback).

It is built with a **FastAPI backend** and a **React + Vite + TypeScript SPA** (`frontend-2/`, shadcn-ui + Tailwind). Authentication and per-user data are powered by **Supabase**.

---

## 1) Project Structure

```text
backend/                    # FastAPI app (app.py) + service layer
  app.py                    # Entrypoint: routes, middleware, startup checks
  insights_service.py       # Insights orchestration, news sentiment, chatbot
  feature_services.py       # Events, backtest, guest sync
  user_data.py              # Portfolio/watchlist CRUD (Supabase or in-memory)
  report_pdf.py             # PDF report: WeasyPrint (HTML->PDF) primary, fpdf2 fallback
  chat_style_service.py     # Optional Gemini "style rewrite" of chatbot replies
  supabase_jwt.py           # Supabase JWT verification (JWKS)
  shared_utils.py           # TTL cache + HTTP retry helpers
frontend-2/                 # Active React/Vite/TS SPA (built to frontend-2/dist)
frontend/                   # Legacy single-file HTML UI (favicon fallback only)
Model_gens/                 # Model training scripts + saved artifacts + news scraper
  fundamentals_labels.py    # Single source of truth: features, label + score definitions
  text_cleaning.py          # Shared sentiment text cleaning (train == serve) + mojibake repair
  evaluate_fundamentals.py  # Held-out (5-fold OOF) evaluation harness for the ensemble
datasets/                   # Input CSV datasets
supabase/                   # SQL migrations: portfolio/watchlists (+ alerts tables, future-scoped, no API yet)
tests/                      # Pytest API reliability tests
stock_insights.py           # Core scoring/inference orchestration + CLI
Dockerfile                  # Multi-stage build (Node SPA build -> Python runtime)
render.yaml                 # Render Blueprint deploy config
requirements.txt            # Python dependencies
pytest.ini                  # Pytest config
```

---

## 2) Prerequisites

- Python 3.11 (matches the Docker runtime; 3.10+ works)
- pip
- Node.js 20+ and npm (to build/run the `frontend-2` SPA)
- (Optional) Docker
- (Optional) A Supabase project (for auth, portfolio, watchlists, sync, report export)
- (Optional) A Gemini API key (chatbot upgrades from local fallback to Gemini)
- (Optional) cloudflared (for temporary public sharing)

---

## 3) Setup

### 3.1 Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Configure environment variables

```bash
cp .env.example .env
```

Set values in `.env` (see Section 8 for the full list). For basic local dev you can keep defaults; set `GEMINI_API_KEY` for Gemini chatbot responses, and the `SUPABASE_*` keys if you want auth-gated features (portfolio, watchlists, report export, sync).

---

## 4) Generate/Train Models (Important)

The backend performs a **startup readiness check** and will refuse to start if any dataset or model artifact is missing. The training scripts use paths relative to the script file (`Path(__file__)`), so they can be run from **any working directory** — the examples below use the project root.

All four fundamentals models share one label definition in `Model_gens/fundamentals_labels.py` (feature columns, the continuous `fundamental_score`, and the binary `good_fundamentals` classifier target derived by thresholding that score). This is the single source of truth: the classifier and the regressors are guaranteed to train against the same concept, and there is no copy-pasted label logic to drift. Each training script also prints a `[holdout]` metric (an 80/20 split for a quick generalization read) before refitting on all data to save the shipped artifact.

### 4.1 Fundamentals classifier

```bash
.venv/bin/python Model_gens/stockfundamentalanalysis.py
```

Output: `Model_gens/fundamentals_stock_model.joblib` (RandomForest classifier)

### 4.2 Fundamentals regressor (Random Forest)

```bash
.venv/bin/python Model_gens/rfr_fundamentals_scorer.py
```

Output: `Model_gens/rfr_stockfundamentalsscorer.pkl`

### 4.3 Fundamentals regressor (XGBoost)

```bash
.venv/bin/python Model_gens/stockscoreregression.py
```

Output: `Model_gens/stock_score_regression.pkl`

### 4.4 Sentiment model + vectorizer

```bash
.venv/bin/python Model_gens/sentiment_logreg.py
```

Outputs:

- `Model_gens/sentiment_logreg.pkl`
- `Model_gens/tfidf_vectorizer.pkl`

### 4.5 Verify all required artifacts exist

You should have these 5 artifacts in `Model_gens/`:

- `fundamentals_stock_model.joblib`
- `rfr_stockfundamentalsscorer.pkl`
- `stock_score_regression.pkl`
- `sentiment_logreg.pkl`
- `tfidf_vectorizer.pkl`

If any file is missing, rerun that model script. (`Model_gens/webscrapernews.py` is a Google News RSS scraper helper, `fundamentals_labels.py` / `text_cleaning.py` are shared library modules, and `evaluate_fundamentals.py` is an evaluation tool — none of these produce an artifact.)

### 4.6 (Optional) Evaluate the fundamentals ensemble

```bash
.venv/bin/python Model_gens/evaluate_fundamentals.py
```

This runs 5-fold **out-of-fold** cross-validation and reports each model's held-out faithfulness to the heuristic label, plus how ensemble MAE / ranking change if a member is dropped. It's a leakage-free way to justify ensemble composition decisions (it's what showed the RF regressor is worth keeping and the Keras NN was not).

---

## 5) Build the Frontend (SPA)

The active UI is the React/Vite app in `frontend-2/`. FastAPI serves `frontend-2/dist/` when it exists.

```bash
cd frontend-2
npm ci
# Configure Supabase for the browser client (build-time):
#   create frontend-2/.env with:
#   VITE_SUPABASE_URL=...
#   VITE_SUPABASE_ANON_KEY=...
npm run build
cd ..
```

### Frontend dev server (optional, hot reload)

```bash
cd frontend-2
npm run dev
```

The Vite dev server runs on port **8080** and proxies `/api`, `/chart`, `/fundamentals`, `/news`, `/tickers`, `/chatbot`, and `/stock` to `http://127.0.0.1:8000` (so run the backend too).

---

## 6) Run the App

```bash
.venv/bin/uvicorn backend.app:app --reload --port 8000
```

Open:

- `http://localhost:8000` (serves the built SPA)

Health:

- `http://localhost:8000/health` -> `{"status":"ok"}`

If `frontend-2/dist` is not built, `/` returns a small JSON info payload instead of the SPA.

---

## 7) API Endpoints

All routes are defined in `backend/app.py`. Endpoints marked **Auth** require an `Authorization: Bearer <Supabase access token>` header.

### Pages / static (serve SPA)

- `GET /` — SPA index (JSON info fallback if no build)
- `GET /app`, `GET /portfolio`, `GET /watchlists`, `GET /stock/{symbol}` — SPA deep-link routes
- `GET|HEAD /favicon.ico`, `GET|HEAD /favicon.svg`

### Core insights & data

- `GET /health`
- `POST /api/insights` — fundamentals + sentiment + final score (cached, coalesced, rate limited)
- `GET /api/insights/explain/{ticker}` — explainability breakdown (driver weights/contributions)
- `GET /chart/{ticker}` — OHLCV chart data (`period`, `interval` query params)
- `GET /fundamentals/{ticker}` — fundamentals score + metrics
- `GET /news/{stock_name}` — live news + sentiment (`max_articles` query param)
- `GET /tickers` — ticker dropdown options
- `GET /api/events/{ticker}` — upcoming earnings/dividends (yfinance)
- `GET /api/backtest/{ticker}` — simple momentum backtest (`period` query param)

### Chatbot

- `POST /chatbot` — chatbot reply (Gemini + local fallback, rate limited)
- `GET /chatbot/status` — reports whether Gemini is configured

### Auth-gated user data (**Auth**)

- `GET /api/portfolio` — list holdings
- `POST /api/portfolio` — add holding (`ticker`, `quantity`, `avg_buy_price`)
- `DELETE /api/portfolio/{holding_id}` — delete holding
- `GET /api/watchlists` — list watchlists
- `POST /api/watchlists` — upsert watchlist (`name`, `tickers`)
- `POST /api/report/export` — generate PDF report (`application/pdf`)
- `POST /api/sync/push` — save a cross-device sync snapshot (`token`, `state`)
- `GET /api/sync/pull/{token}` — load a sync snapshot by `token`

> **On the sync endpoints:** despite the "guest" naming of the token, `/api/sync/push` and `/api/sync/pull/{token}` both call `_require_auth_user`, so they **do** require the `Authorization: Bearer` header (hence their placement here). The `token` in the body/path is a separate *sync-session identifier* — not an auth credential — issued by `POST /api/auth/guest`. In other words, a logged-in user obtains a sync token, then pushes/pulls snapshots under it; the token alone is not sufficient.

### Public auth helpers (no Bearer required)

- `POST /api/auth/guest` — issue a sync-session token used by the sync endpoints above
- `POST /api/auth/unstick-email-confirm` — admin email-confirm fix via Supabase (rate limited)

---

## 8) Environment Variables

Copy `.env.example` to `.env`. `.env` is gitignored — never commit it.

### Chatbot (Gemini)

- `GEMINI_API_KEY` — enables Gemini chatbot; empty falls back to the local assistant
- `GEMINI_TIMEOUT_SECONDS` (default `12`)
- `CHAT_STYLE_REWRITE` (default `false`) — optional Gemini fact-preserving style rewrite
- `CHAT_STYLE_PROVIDER` (default `gemini`)

### App / security

- `APP_ENV=development|production` (default `development`)
- `FRONTEND_DIR` (default `frontend-2`) — which frontend directory to serve
- `ALLOWED_HOSTS` — comma-separated trusted hosts (dev default `*`, prod default `localhost`)
- `CORS_ALLOW_ORIGINS` — comma-separated origins
- `ENABLE_PROXY_HEADERS=true|false` (default: true in production)
- `CORS_ALLOW_CREDENTIALS=true|false`
- `CORS_ALLOW_METHODS` (e.g. `GET,POST,OPTIONS`)
- `CORS_ALLOW_HEADERS` (e.g. `Content-Type,Authorization`)

### Reliability / rate limits

- `RATE_LIMIT_INSIGHTS_COUNT` (default `60`)
- `RATE_LIMIT_INSIGHTS_WINDOW_SECONDS` (default `60`)
- `RATE_LIMIT_CHATBOT_COUNT` (default `30`)
- `RATE_LIMIT_CHATBOT_WINDOW_SECONDS` (default `60`)
- `RATE_LIMIT_UNSTICK_EMAIL_COUNT` (default `8`)
- `RATE_LIMIT_UNSTICK_EMAIL_WINDOW_SECONDS` (default `3600`)
- `INSIGHTS_CACHE_TTL_SECONDS` (default `8`)
- `HTTP_RETRY_COUNT` (default `2`)
- `HTTP_RETRY_BACKOFF_MS` (default `300`)
- `DEBUG_INSIGHTS_CALLS` (default `false`)

### Supabase (auth, portfolio, watchlists, report export, sync)

Server-side (backend):

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` — server-side access (bypasses RLS); if unset, backend uses in-memory dev stores
- `SUPABASE_ANON_KEY` — used by the email-confirm helper
- `SUPABASE_JWT_ISSUER` (optional; defaults to `{SUPABASE_URL}/auth/v1`)
- `AUTH_JWT_INSECURE_SKIP_VERIFY` — **tests only**, skips JWT signature verification

Browser client (build-time, injected into the SPA):

- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`

> If `SUPABASE_*` server keys are not set, auth-gated endpoints fall back to in-memory stores for local development.

---

## 9) Testing

Run the API reliability tests:

```bash
.venv/bin/python -m pytest -q tests/test_api.py
```

The suite (7 tests) covers insights response shape, chart 404 fallback, chatbot local fallback on Gemini quota, insights rate limiting (429), and auth-gated route behavior (401 without a token, 200 with one). Tests set `AUTH_JWT_INSECURE_SKIP_VERIFY=1` and monkeypatch heavy startup/model calls.

---

## 10) Docker (Local Container Run)

The `Dockerfile` is multi-stage: it builds the `frontend-2` SPA with Node 20, then runs the FastAPI app on Python 3.11 (with WeasyPrint system libraries installed).

### Build

```bash
docker build -t ainvestify:latest \
  --build-arg VITE_SUPABASE_URL=<your_supabase_url> \
  --build-arg VITE_SUPABASE_ANON_KEY=<your_supabase_anon_key> .
```

### Run

```bash
docker run --rm -p 8000:8000 --env-file .env ainvestify:latest
```

The container starts `uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}`.

Open:

- `http://localhost:8000`

---

## 11) Share Website With Others

### 11.1 Temporary public URL (quickest)

```bash
cloudflared tunnel --url http://localhost:8000
```

Send the generated `trycloudflare.com` link.

### 11.2 Same Wi-Fi sharing

```bash
.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Share:

- `http://<your-local-ip>:8000`

---

## 12) Permanent Deployment (Render)

This repo includes `render.yaml` for Blueprint deployment as a Docker web service (`healthCheckPath: /health`, free plan, auto-deploy).

### Steps

1. Push your full repo to GitHub.
2. In Render: **New + -> Blueprint**.
3. Select the repository and apply (Render reads `render.yaml`).
4. Set/confirm env vars in Render:
   - `ALLOWED_HOSTS` includes `*.onrender.com` (already set in `render.yaml`)
   - `APP_ENV=production`, `ENABLE_PROXY_HEADERS=true` (already set)
   - `GEMINI_API_KEY` (optional, for Gemini chatbot)
   - `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` (build-time, `sync: false`)
   - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY` (for auth-gated features)
5. Deploy.
6. Verify: `https://<your-service>.onrender.com/health` -> `{"status":"ok"}`
7. Share: `https://<your-service>.onrender.com`

---

## 13) How It Works (Core Inference)

`stock_insights.py` orchestrates scoring via `get_insights(...)`:

1. Resolves the input to a ticker (exact symbol match, else company-name fuzzy match) using `datasets/financials_cleaned.csv`.
2. **Fundamentals**: ensembles three models — RandomForest classifier probability, RandomForest regressor, and XGBoost regressor — each clipped to `[0,1]` and averaged. Uses the dataset row, or live yfinance fundamentals with median imputation when the ticker isn't in the dataset.
3. **Sentiment**: logistic-regression + TF-IDF over dataset tweets, or live Google News items (blended `0.7*LR + 0.3*VADER`) when configured. Both the training pipeline and both serving paths clean text through the single shared `Model_gens/text_cleaning.py::clean_text_for_model` so the TF-IDF vocabulary matches at inference time (no train/serve skew).
4. **Final score**: `(1 - sentiment_weight) * fundamentals + sentiment_weight * sentiment`, then confidence-calibrated and mapped to a label (Bullish / Mildly Bullish / Mixed / Mildly Bearish / Bearish).

### Data & model integrity notes

A few deliberate correctness decisions are baked into the current models — worth knowing before retraining or extending:

- **Ensemble = 3 models (Keras NN removed).** Held-out (5-fold OOF) evaluation showed the NN was the weakest member (R² ≈ 0.40 vs 0.71–0.76 for the tree models) while being the heaviest dependency (TensorFlow) and slowest per row. It was dropped; the small ensemble-diversity cost (held-out MAE ≈ 0.106 → 0.117) was judged worth the large complexity/latency/image-size win. Removing it also eliminated the Keras `MinMaxScaler` extrapolation risk, since that scaler existed only to feed the NN.
- **Unified label.** The classifier and regressors previously used subtly different label logic; they now share `fundamentals_labels.py`, with the classifier target defined as `good_fundamentals = (fundamental_score >= 0.8)`.
- **Corrected 52-week orientation.** `datasets/financials_cleaned.csv` had `52w_low` and `52w_high` systematically swapped (every row had low > high). The column values were corrected in place, and all models were retrained on the fixed data.
- **Centralized, corrected text cleaning.** A doubled-backslash bug in the serving-side tweet/news cleaners silently mangled input; all cleaning now goes through one shared function, verified byte-identical to the training cleaner across the full tweet corpus (so no vectorizer retrain was needed). Display text is also repaired for double-encoded UTF-8 (mojibake) via `text_cleaning.py::fix_mojibake`.

The chatbot (`POST /chatbot`) grounds replies in the computed insights; it tries Gemini models first and falls back to a deterministic local assistant (`provider: "local"`) when Gemini is unavailable or quota-limited.

**Chatbot call chain:** `backend/app.py` (`POST /chatbot`) → `backend/insights_service.py` (`get_chatbot_reply`: infers the target symbol, calls `compute_insights` → `stock_insights.get_insights`, then tries Gemini and otherwise builds the local reply) → `backend/chat_style_service.py` (`rewrite_chat_reply`: optional fact-preserving Gemini rewrite, gated by `CHAT_STYLE_REWRITE`). So `insights_service.py` owns the chatbot orchestration; `stock_insights.py` supplies the grounding data, and `chat_style_service.py` is only a post-processing polish step.

### PDF report generation

`POST /api/report/export` → `backend/report_pdf.py` `build_report_pdf_bytes`, which renders a styled HTML document with **WeasyPrint (`HTML(...).write_pdf()`) as the primary path**, and automatically falls back to an **fpdf2** layout if WeasyPrint (or its native libraries) is unavailable or a render fails. This is why the Docker image installs WeasyPrint's system libraries — they are used, not leftover.

---

## 14) Dependencies

Python dependencies are pinned in `requirements.txt` (FastAPI, uvicorn, pandas, numpy, scikit-learn, xgboost, yfinance, feedparser, vaderSentiment, PyJWT, cryptography, httpx, requests, python-dotenv, pytest). Both `weasyprint` and `fpdf2` are intentional: `weasyprint` is the primary PDF renderer (HTML→PDF) and `fpdf2` is the pure-Python fallback used when WeasyPrint's native libraries aren't present. (TensorFlow was removed along with the Keras NN — see Section 13's integrity notes.)

Frontend dependencies are in `frontend-2/package.json` (React 18, Vite 5, TypeScript, Tailwind, shadcn-ui/Radix, `@supabase/supabase-js`, `@tanstack/react-query`, `recharts`, `react-router-dom`).

---

## 15) Troubleshooting

### App won't start / readiness check fails

- The backend hard-fails at startup if any dataset or model artifact is missing. Run the training scripts in Section 4 and confirm all 5 artifacts exist in `Model_gens/`.

### Invalid host header

- Ensure `ALLOWED_HOSTS` includes your deployment host (e.g. `*.onrender.com`).

### SPA not loading (JSON shown at `/`)

- Build the frontend: `cd frontend-2 && npm ci && npm run build`. FastAPI serves `frontend-2/dist` when present.

### Auth-gated features return 401 / not working

- Set the `SUPABASE_*` server keys and `VITE_SUPABASE_*` build args. Without them, auth-gated endpoints use in-memory dev stores.

### Chart unavailable

- App still shows fundamentals/sentiment; this is expected graceful fallback behavior.

### Gemini unavailable / quota exceeded

- Chatbot auto-falls back to the local assistant (`provider: "local"`).

---

## 16) Creators

- **Sriram Kancherla**
  - LinkedIn: [https://www.linkedin.com/in/sriram-kancherla-80a7b028a/](https://www.linkedin.com/in/sriram-kancherla-80a7b028a/)
  - Gmail: [Kancherlasriram2006@gmail.com](mailto:Kancherlasriram2006@gmail.com)

- **Viswanath Parashuram Yadavalli**
  - LinkedIn: [https://www.linkedin.com/in/vishwa-yadavalli-65503628b/](https://www.linkedin.com/in/vishwa-yadavalli-65503628b/)
  - Gmail: [vishwanaathh4@gmail.com](mailto:vishwanaathh4@gmail.com)
