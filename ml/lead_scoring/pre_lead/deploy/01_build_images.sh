#!/usr/bin/env bash
# Build + push the training-base and serving images to Artifact Registry (Cloud Build).
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

echo ">> Building images via Cloud Build (context = repo root)"
echo "   training: ${TRAINING_IMAGE}"
echo "   serving : ${SERVING_IMAGE}"

gcloud builds submit . \
  --project "${PROJECT_ID}" \
  --config deploy/cloudbuild.yaml \
  --substitutions "_TRAINING_IMAGE=${TRAINING_IMAGE},_SERVING_IMAGE=${SERVING_IMAGE}"

echo ">> Done. Next: deploy/02_run_pipeline.sh (train) then deploy/03_deploy_serving.sh"
