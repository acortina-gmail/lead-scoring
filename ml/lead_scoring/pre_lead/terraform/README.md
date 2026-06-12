# Terraform — lead-scoring infra (infra only)

Creates the durable infra: enables APIs, the GCS bucket (models + pipeline-root)
and the Artifact Registry repo. Workloads run as the project **default** service
accounts (no custom SA / IAM here — project Editor can't set IAM policy, and the
default compute SA already has the access the pipeline/serving need).

Replaces `deploy/00_setup_gcp.sh`.

```bash
# from the model root (ml/lead_scoring/pre_lead):
./deploy/tf.sh init
./deploy/tf.sh plan      # review
./deploy/tf.sh apply     # create bucket + AR repo + enable APIs

# then:  ./deploy/01_build_images.sh -> 02 -> 03
```

State is **local** (`terraform.tfstate`, gitignored). To share/lock later, add a
GCS backend in `versions.tf` and `terraform init -migrate-state`.

`project_id` / `region` / `bucket` / `ar_repo` are NOT in `terraform.tfvars` — they
come from the single source of truth (`src/leadscoring/config.py`) via the `TF_VAR_*`
env vars that `deploy/config.sh` exports (`deploy/tf.sh` sources it for you). Only
`alert_emails` lives in `terraform.tfvars`. Region must equal the BigQuery data
location (EU → `europe-west1`). Running bare `terraform` without sourcing
`config.sh` first will just prompt for the missing vars (safe, not wrong values).
