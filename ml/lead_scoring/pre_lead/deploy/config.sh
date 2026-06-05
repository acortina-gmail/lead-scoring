#!/usr/bin/env bash
# Shared settings for the deploy scripts. Edit here, sourced by the others.
# Keep in sync with src/leadscoring/config.py.
set -euo pipefail

# ENV (dev|prod) namespaces the GCS model paths + Cloud Run service. Default dev
# so an un-set ENV can never overwrite prod. Run e.g.  ENV=prod ./deploy/03_...sh
export ENV="${ENV:-dev}"
export PROJECT_ID="${PROJECT_ID:-test-ml-flow-484314}"
export REGION="${REGION:-us-central1}"
export BUCKET="${BUCKET:-bq-pfu-ga4-leadscoring}"
export AR_REPO="${AR_REPO:-lead-scoring}"
export BQ_DATASET="${BQ_DATASET:-dataset}"
export SERVICE="${SERVICE:-lead-scoring-${ENV}}"

export AR_HOST="${REGION}-docker.pkg.dev"
export TRAINING_IMAGE="${AR_HOST}/${PROJECT_ID}/${AR_REPO}/training-base:latest"
export SERVING_IMAGE="${AR_HOST}/${PROJECT_ID}/${AR_REPO}/lead-scoring-serve:latest"
# Base (env-namespaced) prefix; serving appends /live, the pipeline writes /candidate.
export GCS_MODEL_PREFIX="gs://${BUCKET}/models/${ENV}"
