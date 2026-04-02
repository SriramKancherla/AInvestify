# --- Build React SPA (dist/ is gitignored; image must include a real Vite build) ---
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend-2
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY
COPY frontend-2/package.json frontend-2/package-lock.json ./
RUN npm ci
COPY frontend-2/ ./
RUN npm run build

FROM python:3.11.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FRONTEND_DIR=frontend-2

WORKDIR /app

# WeasyPrint (HTML → PDF): see https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#debian-11
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libffi-dev \
    libjpeg62-turbo \
    libopenjp2-7 \
    fonts-dejavu-core \
    shared-mime-info \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
COPY --from=frontend-build /app/frontend-2/dist /app/frontend-2/dist

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
