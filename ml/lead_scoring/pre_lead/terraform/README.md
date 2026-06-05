# Terraform — lead-scoring infra (infra only)

Creates the durable infra: enables APIs, the GCS bucket (models + pipeline-root)
and the Artifact Registry repo. Workloads run as the project **default** service
accounts (no custom SA / IAM here — project Editor can't set IAM policy, and the
default compute SA already has the access the pipeline/serving need).

Replaces `deploy/00_setup_gcp.sh`.

```bash
cd terraform
terraform init
terraform plan      # review
terraform apply     # create bucket + AR repo + enable APIs

# then back at repo root:  ./deploy/01_build_images.sh -> 02 -> 03
```

State is **local** (`terraform.tfstate`, gitignored). To share/lock later, add a
GCS backend in `versions.tf` and `terraform init -migrate-state`.

Values in `terraform.tfvars` must match `src/leadscoring/config.py` /
`deploy/config.sh` (project, region, bucket, ar_repo). Region must equal the
BigQuery data location (`us-central1`).
