"""Real-time lead-scoring API (Cloud Run).

Loads the per-segment artifacts produced by the Vertex pipeline from GCS at
startup, routes each incoming lead to the right segment model, applies the SAME
``leadscoring.preprocess`` + the saved ``ColumnTransformer`` (no train/serve
skew), and returns a score in [0, 1] for ranking commercial calls.

Env:
  GCS_MODEL_PREFIX   gs://<bucket>/models   (where the pipeline wrote the joblibs)
  PORT               provided by Cloud Run (default 8080)
"""
from __future__ import annotations

import os
import tempfile

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from leadscoring import config, preprocess

app = FastAPI(title="OBS lead scoring", version="1.0")
MODELS: dict[str, dict] = {}


def _download(uri: str) -> str:
    from google.cloud import storage

    bkt, _, blob = uri[len("gs://"):].partition("/")
    fd, local = tempfile.mkstemp(suffix=".joblib")
    os.close(fd)
    storage.Client(project=config.PROJECT_ID).bucket(bkt).blob(blob).download_to_filename(local)
    return local


@app.on_event("startup")
def load_models() -> None:
    for segment in config.SEGMENTS:
        # Always serve the promoted 'live' artifact (config.MODELS_PREFIX already
        # honours GCS_MODEL_PREFIX + ENV); never the unvalidated 'candidate'.
        uri = config.model_uri(segment, stage="live")
        try:
            local = _download(uri) if uri.startswith("gs://") else uri
            MODELS[segment] = joblib.load(local)
            print(f"loaded {segment} model from {uri}")
        except Exception as e:  # don't crash all segments if one is missing
            print(f"WARNING: could not load {segment} model from {uri}: {e}")


class ScoreRequest(BaseModel):
    # Free-form lead/form payload; only the model's features are used, the rest ignored.
    model_config = {"extra": "allow"}


@app.get("/")
def root():
    return {
        "service": "lead-scoring",
        "segments_loaded": list(MODELS.keys()),
        "models": {
            s: {
                "features": m["features"],
                "base_rate": m.get("base_rate"),
                "metrics": m.get("metrics"),
            }
            for s, m in MODELS.items()
        },
    }


@app.get("/health")
def health():
    if not MODELS:
        raise HTTPException(503, "no models loaded")
    return {"status": "ok", "segments": list(MODELS.keys())}


@app.post("/score")
def score(payload: dict):
    if not MODELS:
        raise HTTPException(503, "no models loaded")
    segment = config.route_segment(payload)
    art = MODELS.get(segment) or next(iter(MODELS.values()))

    row = preprocess.derive_columns(pd.DataFrame([payload]))  # page_path/utm_campaign from page_location
    X = preprocess.transform(art["preprocessor"], row, art["num"], art["cat"])
    proba = float(art["model"].predict_proba(X)[0, 1])

    return {
        "segmento": segment,
        "score": proba,
        "base_rate": art.get("base_rate"),
        "lift_vs_base": (proba / art["base_rate"]) if art.get("base_rate") else None,
        "features_used": art["features"],
        "schema_version": art.get("schema_version"),
    }
