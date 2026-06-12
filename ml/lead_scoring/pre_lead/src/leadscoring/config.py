"""Central configuration for the lead-scoring train + serve pipeline.

Everything that depends on the GCP project / dataset / naming lives here so the
rest of the code is environment-agnostic. Values can be overridden via env vars
(handy for Cloud Run / Vertex without rebuilding).
"""
from __future__ import annotations

import os

# --- Environment -------------------------------------------------------------
# Logical environment (dev | prod). Namespaces the GCS model paths and the Cloud
# Run service so dev experiments never touch the prod model that's serving real
# calls — WITHOUT requiring a second GCP project (just a different prefix). If the
# client later provides a prod project, only PROJECT_ID/BUCKET change; the rest is
# identical. Default 'dev' so an un-set ENV can never overwrite prod by accident.
ENV = os.environ.get("ENV", "dev")

# --- Deployment target (THE single place to retarget) ------------------------
# Everything project-specific lives in THIS block. deploy/config.sh derives its
# values from here (no second copy), Terraform reads them via the TF_VAR_* that
# config.sh exports, and the Vertex components read this same config.py baked into
# the training image. To point at another GCP project, edit here only. Every value
# stays env-overridable for one-off runs (e.g. PROJECT_ID=other ./deploy/...).
PROJECT_ID = os.environ.get("PROJECT_ID", "bq-pfu-ga4")
REGION = os.environ.get("REGION", "europe-west1")  # must match the BQ data location's continent
BUCKET = os.environ.get("BUCKET", "bq-pfu-ga4-leadscoring")  # gs://<BUCKET> — GLOBALLY unique
AR_REPO = os.environ.get("AR_REPO", "lead-scoring")          # Artifact Registry repo

# --- BigQuery source ---------------------------------------------------------
BQ_DATASET = os.environ.get("BQ_DATASET", "dataset")
BQ_TABLE = os.environ.get("BQ_TABLE", "lead_scoring_train")
BQ_TABLE_REF = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
# BigQuery job location. The client defaults to "US"; an EU dataset MUST be queried
# with location="EU" (else "Dataset ... was not found in location US"). Keep this in
# step with REGION (EU data -> a europe-* region).
BQ_LOCATION = os.environ.get("BQ_LOCATION", "EU")

# --- Schema contract ---------------------------------------------------------
# Columns that must NEVER be used as features (identifiers + target + routing).
# The pipeline derives the feature list dynamically as
#   (all columns) - ID_COLS - {TARGET, SEGMENT_COL}
# so adding/removing columns in BigQuery never breaks training again.
TARGET = "y"
SEGMENT_COL = "segmento"
ID_COLS = [
    "event_timestamp",
    "user_pseudo_id",
    "ga_session_id",
    "transaction_id",
]

SEGMENTS = ["landing", "main"]

# Frozen feature list per segment (the default the pipeline trains on).
# A segment without an entry falls back to fully dynamic (all non-ID/target/segment
# columns). Only columns present after `derive_columns` are kept (resolve_features
# intersects), so a missing/renamed column is skipped rather than crashing.
# NOTE: `page_path` and `utm_campaign` are DERIVED from `page_location`
# (see preprocess.derive_columns), not raw table columns.
FEATURE_OVERRIDES: dict[str, list[str]] = {
    "landing": ["ga_session_number", "user_studies", "language_site", "utm_campaign", "page_path"],
    "main": ["ga_session_number", "product_id", "user_country", "user_province", "user_studies", "form_name", "page_name"],
}

# --- GCS layout --------------------------------------------------------------
# Models are namespaced by ENV and by STAGE:
#   {MODELS_PREFIX}/candidate/lead_scoring_<segment>.joblib   fresh retrain output
#   {MODELS_PREFIX}/live/lead_scoring_<segment>.joblib        promoted; what serving loads
# The monthly retrain writes 'candidate'; the in-pipeline gate promotes it to
# 'live' only if it doesn't regress vs the current live model (see PROMOTION).
MODELS_PREFIX = os.environ.get("GCS_MODEL_PREFIX", f"gs://{BUCKET}/models/{ENV}")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", f"gs://{BUCKET}/pipeline-root")

# --- Promotion gate (candidate -> live) --------------------------------------
# A retrained 'candidate' is promoted to 'live' only if it doesn't clearly regress.
# We gate on the honest multi-seed `lift_top` mean (already in `stability`):
#   * sanity: candidate lift >= min_abs (beats random),
#   * no-regression: candidate lift >= live lift - max_regression.
# SOFT behaviour: failing the gate keeps the current live model and emits a
# 'promoted=0' metric + HTML reason in the Vertex UI; it never fails the pipeline.
PROMOTION = {
    "metric": "lift_top",
    "min_abs": 1.0,
    "max_regression": 0.15,
}


def model_uri(segment: str, stage: str = "live") -> str:
    """GCS URI of a segment artifact at a given stage ('live' or 'candidate')."""
    return f"{MODELS_PREFIX}/{stage}/lead_scoring_{segment}.joblib"


# --- Score grades (A/B/C) ----------------------------------------------------
# Grades are PERCENTILE BANDS of the model's OWN score distribution (scores are
# uncalibrated/ranking-only, so absolute cutoffs would be meaningless):
#   A = top 25% (best converting), B = 25–50%, C = bottom 50%.
# The cutoffs are fitted per model at train time (evaluate.grade_thresholds) and
# stored in the artifact; grade_of() maps a live score to its grade at serve time.
GRADE_BANDS = [("A", 75), ("B", 50)]  # (grade, lower percentile); below B -> GRADE_FALLBACK
GRADE_FALLBACK = "C"


def grade_of(score, thresholds):
    """Map a raw score to its letter grade using fitted per-model thresholds.

    ``thresholds`` is ``{"A": q75, "B": q50}`` (score cutoffs). Returns ``None`` when
    the artifact carries no thresholds (e.g. an older model), so the caller degrades
    gracefully instead of crashing.
    """
    if not thresholds:
        return None
    for g, _ in GRADE_BANDS:
        if score >= thresholds[g]:
            return g
    return GRADE_FALLBACK


def route_segment(payload: dict) -> str:
    """Decide which segment model scores this lead.

    Prefer an explicit `segmento` field; otherwise derive from `form_name`
    (unbounce landings -> 'landing', everything else -> 'main'), matching how the
    BigQuery `segmento` column is built upstream.
    """
    seg = payload.get(SEGMENT_COL)
    if isinstance(seg, str) and seg.strip():
        seg = seg.strip().lower()
        return seg if seg in SEGMENTS else "main"
    form = str(payload.get("form_name", "") or "")
    return "landing" if form.lower().startswith("unbounce") else "main"
