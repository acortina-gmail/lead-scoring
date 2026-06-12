# lead scoring — `pre_lead` (stage 1)

Segmented lead-scoring rankers (segments **`landing`** / unbounce and **`main`** / web)
to prioritize commercial calls, scored **at form submit** (the first funnel stage).
Trained on Vertex AI Pipelines, served real-time on Cloud Run. Scores are used for
**ranking** (lift / precision@decile), not as calibrated probabilities.

> This is one model of the funnel — see the [repo-root README](../../../README.md) for
> the overall structure. **All commands below run from this directory**
> (`ml/lead_scoring/pre_lead/`), which is this model's self-contained root.

## Layout

```
src/leadscoring/      shared library (baked into BOTH the training and serving images)
  config.py           project/region/bucket + schema contract + segment routing
  preprocess.py       DYNAMIC feature parsing + ColumnTransformer  (the no-skew core)
  data.py             BigQuery -> DataFrame
  train.py            leak-free tuning (RandomizedSearchCV) + production refit
  evaluate.py         multi-seed stability + ROC/lift + HTML report for the Vertex UI
pipelines/            KFP v2 pipeline (ingest -> per segment: split -> preprocess -> train -> evaluate -> package)
serving/              FastAPI scoring API + Dockerfile  (Cloud Run)
training/             base image for the KFP components
deploy/               00 setup -> 01 build -> 02 train -> 03 serve
```

## Key design choices

- **Dynamic schema.** Features are derived as *(all columns) − IDs − target − segment*,
  so a BigQuery schema change never crashes training again. Freeze a validated subset
  per segment via `FEATURE_OVERRIDES` in `src/leadscoring/config.py` when ready.
- **No train/serve skew.** The same `leadscoring.preprocess` module is in both images,
  and the *fitted* `ColumnTransformer` is saved and re-applied at serve time (never re-fit).
  `scikit-learn`/`xgboost` are pinned identically in both images so the joblib unpickles.
- **Nulls are information.** Categoricals → `MISSING` category; numerics → `NaN` kept
  (XGBoost native). Never mean/median-imputed.
- **Preprocess is a separate pipeline step from train** (`fit_preprocess` → `train_model`).
- **Monthly retrain can't silently break prod.** Each retrain writes a *candidate*; an
  in-pipeline gate promotes it to *live* only if it doesn't regress. See
  [Environments & model promotion](#environments--model-promotion).

## Deploy (run from repo root)

> Full step-by-step guide (prerequisites, verification, troubleshooting): **[DEPLOY.md](DEPLOY.md)**.

```bash
# 0) one-time: APIs, GCS bucket, Artifact Registry repo
#    Terraform (preferred):
cd terraform && terraform init && terraform apply && cd ..
#    ...or the gcloud equivalent:  ./deploy/00_setup_gcp.sh

# 1) build + push training-base and serving images
./deploy/01_build_images.sh

# 2) train on Vertex. Writes a CANDIDATE, then the gate promotes it to LIVE:
#    gs://<bucket>/models/<env>/live/lead_scoring_{landing,main}.joblib
#    Add --compile-only to just validate the pipeline locally.
ENV=dev ./deploy/02_run_pipeline.sh

# 3) deploy the scoring API to Cloud Run (serves the LIVE model of that env)
ENV=dev ./deploy/03_deploy_serving.sh
```

`ENV` (default `dev`) selects the environment — see
[Environments & model promotion](#environments--model-promotion). Other defaults
(project `bq-pfu-ga4`, region `europe-west1`, bucket `bq-pfu-ga4-leadscoring`, AR repo
`lead-scoring`, `BQ_LOCATION=EU`) live in **one place**, the "Deployment target" block of
`src/leadscoring/config.py`. `deploy/config.sh` and Terraform read from it (no duplication)
— override any value via env vars, or edit that block to retarget another project.

## Scoring

`POST /score` with the raw form/lead JSON; the service routes by `segmento`
(or derives it from `form_name`; in the **real deploy** this switches to `platform`),
applies the saved preprocessing and returns:

```json
{ "segmento": "landing", "score": 0.071, "grade": "A", "lift_vs_base": 3.0, "features_used": [...] }
```

`grade` is an A/B/C ranking band (A = top 25% by score, B = 25–50%, C = bottom 50%).

Only the model's features are read from the payload; missing keys become `MISSING`/`NaN`,
extra keys are ignored — robust to schema drift.

### Examples

The Cloud Run service is private, so every call needs an identity token. Grab the URL once:

```bash
URL=$(gcloud run services describe lead-scoring --region us-central1 \
      --format='value(status.url)')
TOK=$(gcloud auth print-identity-token)

curl -s "$URL/health" -H "Authorization: Bearer $TOK"
# {"status":"ok","segments":["landing","main"]}
```

**Landing** (routed from `form_name` starting with `unbounce` — in the real deploy,
from `platform`; `utm_campaign` is *derived from* `page_location` and `page_path` from
`page_name`, so send `page_name` + `page_location`):

```bash
curl -s -X POST "$URL/score" -H "Authorization: Bearer $TOK" \
  -H 'Content-Type: application/json' \
  -d '{"form_name":"unbounce_master","platform":"WEB",
       "page_name":"unbounce/mba",
       "page_location":"https://obs.edu/landing/mba?utm_campaign=brand",
       "user_studies":"es-2","language_site":"es","ga_session_number":2}'
# {"segmento":"landing","score":0.498,"grade":"A","base_rate":0.0237,"lift_vs_base":21.0,
#  "features_used":["ga_session_number","user_studies","language_site","utm_campaign","page_path"],
#  "schema_version":2}
```

**Main** (segment given explicitly via `segmento`; uses `form_name` + `page_name` as
features, so send them):

```bash
curl -s -X POST "$URL/score" -H "Authorization: Bearer $TOK" \
  -H 'Content-Type: application/json' \
  -d '{"form_name":"web_contacto","platform":"WEB","segmento":"main","product_id":"mba-full",
       "page_name":"producto/detalle/mba","user_country":"ES","user_province":"Barcelona",
       "user_studies":"es-3","ga_session_number":4}'
# {"segmento":"main","score":0.534,"grade":"B","base_rate":0.1229,"lift_vs_base":4.34,
#  "features_used":["ga_session_number","product_id","user_country","user_province","user_studies","form_name","page_name"],
#  "schema_version":2}
```

> ⚠️ **`score` is a ranking score, not a calibrated probability.** `scale_pos_weight`
> centers raw scores near 0.5, so the `lift_vs_base` field (= `score / base_rate`) is **not**
> a real lift — don't read it as "converts 21× more". Use the score to **rank leads and call
> the top ones**; the validated lift is the *decile* lift from the pipeline (~1.5–2.4×).

## Environments & model promotion

The model **retrains monthly**, so the central risk is a *bad* retrain (data drift, a
BigQuery schema change, a degenerate `best_iter` like the 11.67×→2.4× fluke we hit early)
**silently replacing the live model** that's scoring real commercial calls. Two
independent safeguards prevent that — keep them distinct:

| Gate | Scope | Who decides | Built? |
|---|---|---|---|
| **candidate → live** | within one env | the pipeline, automatically | ✅ here |
| **dev → prod** | across envs | a human (GitHub Environments approval) | 🔜 future CI |

### `ENV` — dev/prod without needing two GCP projects

Everything is parameterized by `ENV` (`dev` \| `prod`, default **`dev`**). It namespaces
the GCS model paths and the Cloud Run service, so dev experiments never touch prod —
**on a single GCP project**. You climb this ladder as the client provides more:

1. **One project, namespaced** *(current)* — `models/dev/…` vs `models/prod/…`, services
   `lead-scoring-dev` / `lead-scoring-prod`.
2. **One project + the promotion gate** — the part that actually protects prediction quality.
3. **Two projects** *(if the client gives one)* — just point `PROJECT_ID`/`BUCKET` at the
   prod project; **no code changes**, same scripts.

```bash
ENV=dev  ./deploy/02_run_pipeline.sh && ENV=dev  ./deploy/03_deploy_serving.sh
ENV=prod ./deploy/02_run_pipeline.sh && ENV=prod ./deploy/03_deploy_serving.sh
```

### GCS layout — candidate vs live

```
gs://<bucket>/models/<env>/candidate/lead_scoring_<segment>.joblib   # fresh retrain output
gs://<bucket>/models/<env>/live/lead_scoring_<segment>.joblib        # promoted; what serving loads
```

The serving API **only ever loads `live/`** (`config.model_uri(segment, stage="live")`).

### The candidate → live gate (automatic, in the pipeline)

```
… → evaluate → package_artifact (→ candidate/) → validate_and_promote
                                                       │
        compare candidate vs current live on lift_top (multi-seed mean):
          sanity:        candidate ≥ 1.0                 (beats random)
          no-regression: candidate ≥ live − 0.15
                                                       │
                              ┌── PASS ──────────────► copy candidate → live ✅
                              └── FAIL ──────────────► keep live, promoted=0 + HTML reason
                                                       (pipeline stays GREEN — SOFT gate)
```

- **Metric & tolerance** live in `config.PROMOTION` (`lift_top`, `min_abs=1.0`,
  `max_regression=0.15`) and are passed to the pipeline as parameters — tune without code edits.
- **First run bootstraps**: no `live/` yet → promote if sanity passes.
- **SOFT** (your choice): a failed gate never fails the pipeline. The run is green,
  `live/` is untouched, and the `validate-and-promote-<segment>` step shows a
  `promoted=0` metric + an HTML report with the reason in the Vertex UI. A human reviews it.

### The dev → prod gate (manual — future CI)

When the repo lands on GitHub: two **GitHub Environments** (`dev`, `prod`), `prod` with a
**required-reviewer** rule. `dev` deploys on push; promoting to `prod` waits for a click.
Those environments hold the per-env variables, so they can point at one project (namespaced)
today and two projects later — same workflow. We'll use **GitHub Actions** (not Cloud Build)
because it authenticates to **both GCP and Azure**, keeping later funnel-stage models portable.

### Monthly retrain flow

```bash
ENV=prod ./deploy/02_run_pipeline.sh   # retrain → candidate → gate → (maybe) live
ENV=prod ./deploy/03_deploy_serving.sh # redeploy so serving RELOADS the new live model
```

> The API loads models **at container startup**, so a freshly-promoted `live/` model is
> **not** picked up until the service restarts. Re-running step 3 forces a new revision that
> reloads from GCS. (Same note as in [DEPLOY.md](DEPLOY.md).)

## ⚠️ Modelling note

The current BigQuery table dropped `page_path` / `utm_campaign` (strong features in the
earlier analysis). The pipeline trains on the available columns dynamically, but the
previously-reported lifts won't hold as-is — **re-run variable selection on the new
schema** and pin the result in `FEATURE_OVERRIDES`.
