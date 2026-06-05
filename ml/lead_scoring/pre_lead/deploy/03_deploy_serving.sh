#!/usr/bin/env bash
# Deploy the scoring API to Cloud Run (scale-to-zero). Auth required by default.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

echo ">> Deploying ${SERVICE} to Cloud Run (${REGION}, env ${ENV}, serving the LIVE model)"
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${SERVING_IMAGE}" \
  --set-env-vars "ENV=${ENV},GCS_MODEL_PREFIX=${GCS_MODEL_PREFIX},PROJECT_ID=${PROJECT_ID},REGION=${REGION}" \
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 5 \
  --no-allow-unauthenticated

URL=$(gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')
echo ">> Deployed: ${URL}"
echo "   Test (authenticated):"
echo "   curl -s -X POST ${URL}/score -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
echo "        -H 'Content-Type: application/json' -d '{\"form_name\":\"unbounce_x\",\"product_id\":123,\"user_province\":\"Barcelona\"}'"
