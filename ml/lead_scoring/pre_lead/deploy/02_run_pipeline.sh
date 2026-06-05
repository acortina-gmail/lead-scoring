#!/usr/bin/env bash
# Compile + submit the Vertex AI training pipeline. Writes the two serving
# joblibs to gs://${BUCKET}/models/ and shows metrics/HTML in the Vertex UI.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

PY="${PYTHON:-./.venv/bin/python}"
echo ">> Installing pipeline submit deps"
"${PY}" -m pip install -q -r requirements-pipeline.txt

echo ">> Compiling + submitting pipeline (env ${ENV}, image ${TRAINING_IMAGE})"
TRAINING_IMAGE="${TRAINING_IMAGE}" \
ENV="${ENV}" \
PROJECT_ID="${PROJECT_ID}" REGION="${REGION}" BUCKET="${BUCKET}" AR_REPO="${AR_REPO}" \
BQ_DATASET="${BQ_DATASET}" \
"${PY}" pipelines/compile_and_run.py --env "${ENV}" "$@"

echo ">> Submitted (${ENV}). Writes candidate -> gates -> live under gs://${BUCKET}/models/${ENV}/."
echo ">> Watch it in the Vertex AI > Pipelines console (region ${REGION})."
