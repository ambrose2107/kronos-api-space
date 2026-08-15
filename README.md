# Kronos Forecast API

A small REST wrapper around [Kronos](https://github.com/shiyu-coder/Kronos)
(a foundation model for financial K-line forecasting), meant to run as its
own free web service so your trading bot can call it over HTTP instead of
running torch/Kronos in-process.

## Deploy on Render (free)

1. Push this folder to its own GitHub repo (or point Render's "Root
   Directory" setting at this folder if it's a subfolder of a bigger repo).
2. Render dashboard -> New -> Web Service -> connect the repo.
3. Environment: **Docker** (Render auto-detects the Dockerfile here).
4. Instance type: **Free**.
5. No start command needed -- it's set by the Dockerfile's CMD.
6. Deploy. You'll get a URL like `https://kronos-api-xxxx.onrender.com`.

## Endpoints

- `GET /health`
- `POST /predict` -- see `app.py` docstring for the request/response shape.

## Notes on the free tier

- Render's free plan gives 512MB RAM, much tighter than a GPU host. Default
  model is `"mini"` (4.1M params) to fit comfortably. `"small"` (24.7M
  params, better quality) may also fit but is riskier -- try it and watch
  for out-of-memory restarts in Render's logs; downgrade back to `"mini"`
  if you see that.
- First request after the service wakes from sleep will be slow (model
  download + load, ~10-30s). Subsequent requests are fast.
- Free services sleep after 15 min idle. Add this service's `/health` URL
  to your existing UptimeRobot monitor (ping every 5 min) to keep it warm.
