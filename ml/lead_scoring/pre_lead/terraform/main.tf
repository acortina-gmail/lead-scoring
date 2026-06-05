# Infra-only: APIs + GCS bucket + Artifact Registry repo.
# Workloads (Vertex pipeline, Cloud Build, Cloud Run) run as the project DEFAULT
# service accounts, which already carry the access they need — so no custom SA /
# IAM bindings here (and none required: project Editor can't set IAM policy).

locals {
  apis = [
    "serviceusage.googleapis.com",
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "compute.googleapis.com",
  ]
}

resource "google_project_service" "enabled" {
  for_each = toset(local.apis)
  project  = var.project_id
  service  = each.value

  disable_on_destroy = false # don't tear down APIs (other things may use them)
}

resource "google_storage_bucket" "models" {
  name                        = var.bucket
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.force_destroy_bucket

  depends_on = [google_project_service.enabled]
}

resource "google_artifact_registry_repository" "images" {
  repository_id = var.ar_repo
  project       = var.project_id
  location      = var.region
  format        = "DOCKER"
  description   = "Lead scoring training + serving images"

  depends_on = [google_project_service.enabled]
}
