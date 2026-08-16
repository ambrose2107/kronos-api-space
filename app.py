"""
Kronos Forecast API — a small, free-hostable REST wrapper around the Kronos
foundation model (https://github.com/shiyu-coder/Kronos), meant to run as a
Hugging Face Space (free CPU tier) so your trading bot can call it over HTTP
instead of running torch/Kronos in-process.

Endpoints:
  GET  /health            -> {"status": "ok", "model_loaded": bool}
  POST /predict            -> forecast future OHLC bars from bars you send

Request body for /predict:
{
  "bars": [
     {"t": "2026-07-01T00:00:00Z", "o": 100.1, "h": 101.0, "l": 99.5, "c": 100.8, "v": 120000},
     ...
  ],
  "pred_len": 20,
  "period": "1D",          // used only to space future timestamps: 1D/1h/15m/5m
  "model": "small",        // "mini" (fast/tiny) or "small" (default, better quality)
  "sample_count": 1
}

Response:
{
  "last_close": 100.8,
  "forecast": [{"t": "...", "o":.., "h":.., "l":.., "c":..}, ...],
  "implied_move_pct": 3.2,
  "model": "NeoQuasar/Kronos-small"
}
"""
import os
import logging
from datetime import timedelta
from typing import List, Optional, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kronos_model import Kronos, KronosTokenizer, KronosPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kronos-api")

app = FastAPI(title="Kronos Forecast API")

MODEL_CONFIGS = {
    "mini":  {"model_id": "NeoQuasar/Kronos-mini",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",  "max_context": 2048},
    "small": {"model_id": "NeoQuasar/Kronos-small", "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base", "max_context": 512},
}
DEVICE = os.environ.get("KRONOS_DEVICE", "cpu")

_predictors = {}  # lazy cache, one per model size requested


def _get_predictor(size: str) -> KronosPredictor:
    if size not in MODEL_CONFIGS:
        raise HTTPException(400, f"Unknown model '{size}', choose one of {list(MODEL_CONFIGS)}")
    if size not in _predictors:
        cfg = MODEL_CONFIGS[size]
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        model = Kronos.from_pretrained(cfg["model_id"])
        _predictors[size] = KronosPredictor(model, tokenizer, device=DEVICE, max_context=cfg["max_context"])
    return _predictors[size]


class Bar(BaseModel):
    t: str
    o: float
    h: float
    l: float
    c: float
    v: Optional[float] = 0


class PredictRequest(BaseModel):
    bars: List[Bar]
    pred_len: int = 20
    period: Literal["5m", "15m", "1h", "4h", "1D"] = "1D"
    model: Literal["mini", "small"] = "mini"
    sample_count: int = 1


@app.get("/")
def root():
    # Render's own health check hits "/" by default. Keep this trivially
    # fast and dependency-free so it never fails even while the model is
    # still loading elsewhere.
    return {"service": "kronos-forecast-api", "status": "ok", "docs": "/health, /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": list(_predictors.keys())}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.bars) < 50:
        raise HTTPException(400, f"Need at least 50 bars, got {len(req.bars)}")

    df = pd.DataFrame([{
        "timestamps": b.t, "open": b.o, "high": b.h, "low": b.l,
        "close": b.c, "volume": b.v or 0, "amount": (b.v or 0) * b.c,
    } for b in req.bars])
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamps", "close"]).reset_index(drop=True)

    try:
        predictor = _get_predictor(req.model)
        max_ctx = MODEL_CONFIGS[req.model]["max_context"]
        if len(df) > max_ctx:
            df = df.iloc[-max_ctx:].reset_index(drop=True)

        step = {"5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
                "1h": timedelta(hours=1), "4h": timedelta(hours=4)}.get(req.period, timedelta(days=1))
        last_ts = df["timestamps"].iloc[-1]
        y_timestamp = pd.Series([last_ts + step * (i + 1) for i in range(req.pred_len)])

        pred_df = predictor.predict(
            df=df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=df["timestamps"],
            y_timestamp=y_timestamp,
            pred_len=req.pred_len,
            T=1.0, top_p=0.9, sample_count=req.sample_count,
        )
    except Exception as e:
        logger.exception("Kronos inference failed")
        raise HTTPException(500, f"Kronos inference failed: {type(e).__name__}: {e}")

    last_close = float(df["close"].iloc[-1])
    forecast = [{
        "t": ts.isoformat(),
        "o": round(float(row.open), 4), "h": round(float(row.high), 4),
        "l": round(float(row.low), 4), "c": round(float(row.close), 4),
    } for ts, row in zip(y_timestamp, pred_df.itertuples(index=False))]

    final_close = forecast[-1]["c"] if forecast else last_close
    implied_pct = round((final_close - last_close) / last_close * 100, 2) if last_close else 0

    return {
        "last_close": round(last_close, 4),
        "forecast": forecast,
        "implied_move_pct": implied_pct,
        "model": MODEL_CONFIGS[req.model]["model_id"],
    }
