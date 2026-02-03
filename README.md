# Potato Disease Classification

## Run API locally

From the repo root:

- Install deps: `pip install -r api/requirements.txt`
- Run: `uvicorn api.main:app --reload --port 8080`

Ping: `http://localhost:8080/ping`

## Run Streamlit UI locally

- Install deps: `pip install -r requirements.txt`
- Point UI at API (choose one):
  - Local: set `POTATO_API_URL=http://localhost:8080`
  - Fly: set `POTATO_API_URL=https://potato-disease-classification.fly.dev`
- Run: `streamlit run ui/app.py`

## Deploy API to Fly.io

This repo includes a `fly.toml` that builds from `api/Dockerfile`.

1. Install Fly CLI: `flyctl` (Fly.io docs)
2. Login: `fly auth login`
3. Set your app name in `fly.toml` (the `app = "..."` line)
4. Deploy from repo root: `fly deploy`
