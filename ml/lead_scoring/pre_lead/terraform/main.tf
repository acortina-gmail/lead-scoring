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

  # Keep the registry from growing unbounded. Every deploy pushes :latest, which
  # leaves the previous digest UNTAGGED but still stored — those pile up. Policy:
  # keep the most recent few of each image for rollback, prune old untagged ones.
  # KEEP rules take precedence over DELETE, so recent versions are always safe.
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count            = var.ar_keep_recent
      package_name_prefixes = ["lead-scoring-serve", "training-base"]
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "${var.ar_untagged_ttl_days * 24 * 60 * 60}s"
    }
  }

  # Set true to preview deletions in the AR logs without actually deleting.
  cleanup_policy_dry_run = var.ar_cleanup_dry_run

  depends_on = [google_project_service.enabled]
}
