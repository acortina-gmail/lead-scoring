"""Real-time lead-scoring API (Cloud Run).

Loads the per-segment artifacts produced by the Vertex pipeline from GCS at
startup, routes each incoming lead to the right segment model, applies the SAME
``leadscoring.preprocess`` + the saved ``ColumnTransformer`` (no train/serve
skew), and returns a score in [0, 1] for ranking commercial calls.

The live model is also hot-swapped without a redeploy: requests trigger a throttled
GCS re-check (MODEL_RELOAD_CHECK_SECONDS) and any freshly-promoted artifact is loaded
in place. POST /reload forces it immediately (e.g. right after a retrain).

Env:
  GCS_MODEL_PREFIX            gs://<bucket>/models  (where the pipeline wrote the joblibs)
  MODEL_RELOAD_CHECK_SECONDS  min seconds between live-model re-checks (default 300; 0 disables)
  PORT                        provided by Cloud Run (default 8080)
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from leadscoring import config, preprocess

app = FastAPI(title="OBS lead scoring", version="1.0")

# In-memory artifacts + the GCS object generation each was loaded from, so we can
# detect a freshly-promoted model and skip re-downloading an unchanged one.
MODELS: dict[str, dict] = {}
_GENERATIONS: dict[str, int] = {}

# Self-refresh. The model loads at startup, but a (e.g. monthly) retrain promotes
# a new joblib to GCS while this container keeps running. Instead of needing a
# redeploy, we re-check GCS at most once per CHECK_INTERVAL and hot-swap any
# changed segment. The check is driven by incoming requests (see maybe_reload),
# not a background timer, so it still fires under Cloud Run's idle-CPU throttling.
CHECK_INTERVAL = int(os.environ.get("MODEL_RELOAD_CHECK_SECONDS", "300"))
_reload_lock = threading.Lock()
_last_check = 0.0


def _live_blob(uri: str):
    """The GCS blob behind a live-model URI (fetches metadata), or None if absent."""
    from google.cloud import storage

    bkt, _, name = uri[len("gs://"):].partition("/")
    return storage.Client(project=config.PROJECT_ID).bucket(bkt).get_blob(name)


def _load_segment(segment: str, *, force: bool) -> bool:
    """(Re)load one segment's LIVE artifact if its GCS generation changed.

    Returns True if a (new) model was loaded. Always the promoted 'live' artifact
    (config.MODELS_PREFIX honours GCS_MODEL_PREFIX + ENV); never 'candidate'.
    """
    uri = config.model_uri(segment, stage="live")
    try:
        if not uri.startswith("gs://"):  # local path (tests / dev)
            if not force and segment in MODELS:
                return False
            MODELS[segment] = joblib.load(uri)
            return True

        blob = _live_blob(uri)
        if blob is None:
            if force:
                print(f"WARNING: {segment} model not found at {uri}")
            return False
        if not force and _GENERATIONS.get(segment) == blob.generation:
            return False  # already serving this exact object

        fd, local = tempfile.mkstemp(suffix=".joblib")
        os.close(fd)
        blob.download_to_filename(local)
        MODELS[segment] = joblib.load(local)  # atomic rebind of the in-memory ref
        _GENERATIONS[segment] = blob.generation
        os.remove(local)
        print(f"loaded {segment} model from {uri} (generation {blob.generation})")
        return True
    except Exception as e:  # one bad/missing segment must not take down the others
        print(f"WARNING: could not load {segment} model from {uri}: {e}")
        return False


def reload_models(*, force: bool) -> list[str]:
    """Check every segment; return the list that was (re)loaded."""
    return [s for s in config.SEGMENTS if _load_segment(s, force=force)]


def maybe_reload() -> None:
    """Throttled, request-driven refresh: at most one GCS check per CHECK_INTERVAL,
    hot-swapping any segment whose live model changed. A no-op metadata check when
    nothing changed (~ms); only downloads on an actual new generation."""
    global _last_check
    if CHECK_INTERVAL <= 0:
        return
    now = time.monotonic()
    if now - _last_check < CHECK_INTERVAL:
        return
    if not _reload_lock.acquire(blocking=False):
        return  # another request is already checking
    try:
        _last_check = now
        changed = reload_models(force=False)
        if changed:
            print(f"auto-reload: refreshed {changed}")
    finally:
        _reload_lock.release()


@app.on_event("startup")
def _startup() -> None:
    reload_models(force=True)


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
    maybe_reload()
    if not MODELS:
        raise HTTPException(503, "no models loaded")
    return {"status": "ok", "segments": list(MODELS.keys())}


@app.post("/reload")
def reload_endpoint():
    """Force an immediate reload of all live models — e.g. pinged right after a
    retrain promotes new artifacts, so the live API refreshes without a redeploy.
    Returns which segments were (re)loaded."""
    return {"reloaded": reload_models(force=True), "segments": list(MODELS.keys())}


@app.post("/score")
def score(payload: dict):
    maybe_reload()
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
