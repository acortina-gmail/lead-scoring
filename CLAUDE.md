# CLAUDE.md — working context for the lead-scoring repo

Context for AI assistants (and humans) working in this repo. The user-facing docs
are `README.md` (repo root) and `ml/lead_scoring/pre_lead/README.md` + `DEPLOY.md`
(per model) — read those first for *how to use/deploy*. This file captures the
*why* and the non-obvious decisions so you don't re-derive or undo them.

## What this is

ML rankers that score sales leads along OBS's commercial **funnel**, so reps call
the most promising leads first. **One self-contained model per funnel stage** under
`ml/lead_scoring/<stage>/`. Today only **`pre_lead`** exists (stage 1: scores at form
submit). Future: `contact/`, `interview/`. They share the lead, not the code.

Scores are for **ranking** (lift / precision@decile) — **not** calibrated
probabilities. `scale_pos_weight` centers raw scores near 0.5, so an absolute score
or `score/base_rate` is *not* a real lift. The honest metric is multi-seed **decile
lift** (~1.5–2.4×).

## Architecture (`pre_lead`)

BigQuery → **Vertex AI Pipeline (KFP v2)** trains two segment models
(`landing`/unbounce, `main`/web) → packages joblibs to GCS → **Cloud Run** (FastAPI)
serves real-time. The shared `leadscoring` library is baked into both the training
and serving images.

Per segment the pipeline is: `ingest → prepare → fit_preprocess → train_model →
evaluate_model → package_artifact(→candidate) → validate_and_promote(→live)`.

## Non-negotiable design rules (don't undo these)

- **No train/serve skew.** Same `leadscoring.preprocess` in both images **and** the
  *fitted* `ColumnTransformer` is saved and re-applied at serve time (never re-fit).
  `scikit-learn`/`xgboost` are **pinned identically** in both images so the joblib
  unpickles. Don't bump one without the other.
- **Nulls are information.** Categoricals → `MISSING` category; numerics keep `NaN`
  (XGBoost native). **Never** mean/median-impute.
- **Always multi-seed.** Single-split lift is unreliable (we saw a degenerate
  `best_iter=4` give 11.67× that collapsed to ~2.4× over 5 seeds). `evaluate.py` reports
  mean±std; the production refit uses `median(best_iteration)`.
- **Permutation / leave-one-out are fooled by correlated features.** Validate trimmed
  feature subsets directly in CV — don't trust the importance ranking alone.
- **Dynamic schema.** Features = (all columns) − `ID_COLS` − {`TARGET`, `SEGMENT_COL`},
  unless pinned in `config.FEATURE_OVERRIDES`. A BQ schema change must not crash training.
- **`page_path` + `utm_campaign` are DERIVED** from `page_location` in
  `preprocess.derive_columns`, run identically in training (`data.load`) and serving
  (`app.py`). Keep that parity.
- **KFP components: no `from __future__ import annotations`** in `train_pipeline.py`
  (it stringizes annotations and breaks KFP artifact typing).

## Environments & the promotion gate

- **`ENV`** (`dev`|`prod`, default `dev`) namespaces GCS paths + the Cloud Run service —
  dev/prod **without a second GCP project**. Climbs to two projects later by changing
  `PROJECT_ID`/`BUCKET` only.
- GCS layout: `models/<env>/{candidate,live}/lead_scoring_<segment>.joblib`. Serving
  loads **`live`** only.
- **`validate_and_promote`** (in `train_pipeline.py`): a retrain writes `candidate`,
  then promotes to `live` **only if it doesn't regress** — gate on multi-seed
  `lift_top` mean: `cand ≥ 1.0` (sanity) and `cand ≥ live − 0.15`. First run bootstraps.
  Thresholds in `config.PROMOTION`, passed as pipeline params.
- **SOFT gate** (user's explicit choice): a failed gate keeps `live`, emits
  `promoted=0` + an HTML reason in the Vertex UI, and the pipeline **stays green**.
  Never make it raise without asking.
- Serving loads models **at container startup** → a freshly-promoted model needs a
  **redeploy** (step 3) to be picked up.
- **Second gate (future CI):** dev→prod is a *manual* GitHub Environments approval.
  CI will be **GitHub Actions** (not Cloud Build) — chosen for multi-cloud portability
  (later funnel stages may run on Azure if their data lives in Dynamics/CRM).

## Deploy targets (current)

Project `test-ml-flow-484314`, region `us-central1`, bucket `bq-pfu-ga4-leadscoring`,
Artifact Registry repo `lead-scoring`, BQ table `dataset.lead_scoring_train`. The
deployer account (`acortina@knowmadmood.com`) has `roles/editor` but **can't set IAM
policy** → Terraform is **infra-only** (bucket + AR repo + APIs); workloads run as the
default compute SA. **Run deploy commands from `ml/lead_scoring/pre_lead/`.**

## Gotchas already hit (don't rediscover)

- **sklearn ≥ 1.3 `roc_curve` returns `thr[0]=inf`** → serializes to `Infinity`
  (invalid JSON) and Vertex's metadata store rejects it ("Failed to parse the output
  metadata payload"). Fixed by clamping to finite in `evaluate.roc_points` **and**
  inside the KFP component (the component source is embedded in the compiled pipeline,
  so the in-component fix applies without rebuilding the image).
- **`kfp==2.16.1` needs `protobuf>=6.31.1`** → `google-cloud-aiplatform` must be recent
  (pinned `1.156.0`); the old `1.79.0` capped protobuf `<6` and broke the resolver.
- **`gh` has two accounts** on this machine; the repo owner is `acortina-gmail`.

## Conventions

- This repo was migrated from the `planeta` repo (Bitbucket). Don't reintroduce
  `planeta` naming or absolute paths — everything is relative to each model's root.
- Generated `pipeline.json` and Terraform state are gitignored; don't commit them.
