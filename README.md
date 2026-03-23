# AInvestify

FastAPI backend + vanilla frontend for stock insights, sentiment, charts, compare view, and chatbot.

## Local Run

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy env template and set values:
   - `cp .env.example .env`
   - set `GEMINI_API_KEY` if you want Gemini responses
4. Start server:
   - `.venv/bin/uvicorn backend.app:app --reload --port 8000`
5. Open:
   - `http://localhost:8000`

## Deploy Readiness

- CORS is now controlled by `CORS_ALLOW_ORIGINS` (comma-separated list).
- Startup performs readiness checks for required datasets and model files.
- Keep secrets only in `.env` (ignored by git). Use `.env.example` for shared defaults.
- Step 1.1 hardening is enabled: trusted hosts middleware, optional proxy-header middleware, and configurable CORS methods/headers/credentials.
- Step 1.2 hardening is enabled: in-memory rate limits on insights/chatbot endpoints and retry/timeout guards for external HTTP calls.

### Production env example

```
APP_ENV=production
CORS_ALLOW_ORIGINS=https://your-frontend-domain.com
ALLOWED_HOSTS=api.your-domain.com
ENABLE_PROXY_HEADERS=true
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,OPTIONS
CORS_ALLOW_HEADERS=Content-Type,Authorization
```

If you are behind Nginx/Cloudflare/Render/Fly/any reverse proxy, keep `ENABLE_PROXY_HEADERS=true`.

### Reliability and rate-limit envs

```
RATE_LIMIT_INSIGHTS_COUNT=60
RATE_LIMIT_INSIGHTS_WINDOW_SECONDS=60
RATE_LIMIT_CHATBOT_COUNT=30
RATE_LIMIT_CHATBOT_WINDOW_SECONDS=60
HTTP_RETRY_COUNT=2
HTTP_RETRY_BACKOFF_MS=300
GEMINI_TIMEOUT_SECONDS=12
```

## API Endpoints

- `GET /health`
- `POST /api/insights`
- `GET /tickers`
- `GET /chart/{ticker}`
- `GET /fundamentals/{ticker}`
- `GET /news/{stock_name}`
- `POST /chatbot`
- `GET /chatbot/status`

## Deploy Packaging (Docker)

1. Build image:
   - `docker build -t ainvestify:latest .`
2. Run container:
   - `docker run --rm -p 8000:8000 --env-file .env ainvestify:latest`
3. Open:
   - `http://localhost:8000`

## Share Website With Friend

### Quick share (temporary link, easiest)

Use Cloudflare Tunnel from your laptop while app runs on port `8000`:

- `cloudflared tunnel --url http://localhost:8000`

It gives a public URL you can send to your friend.

### Same Wi-Fi share (local network only)

Run server on all interfaces:

- `.venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000`

Then share:

- `http://<your-laptop-local-ip>:8000`

### Permanent share (recommended)

Deploy Docker image to Render/Railway/Fly.io and share that HTTPS URL.

## Permanent Deploy (Render)

1. Push this project to a GitHub repository.
2. In Render dashboard, choose **New +** -> **Blueprint**.
3. Select your repository (Render reads `render.yaml` automatically).
4. Set required environment values in Render:
   - `ALLOWED_HOSTS=<your-render-service>.onrender.com`
   - `CORS_ALLOW_ORIGINS=https://<your-render-service>.onrender.com`
   - `GEMINI_API_KEY=<your_key>`
5. Deploy and wait for health check `GET /health` to pass.
6. Share your permanent URL:
   - `https://<your-render-service>.onrender.com`

### Notes

- If you later host frontend on another domain, update:
  - `CORS_ALLOW_ORIGINS` to that frontend domain.
  - keep `ALLOWED_HOSTS` as your backend domain.
- Do not commit `.env`; set secrets only in Render dashboard.
