variable "project_id" {
  type        = string
  description = "GCP project where the lead-scoring infra lives."
}

variable "region" {
  type        = string
  description = "Region for the bucket, Artifact Registry, Vertex and Cloud Run. Must match the BigQuery data location."
}

variable "bucket" {
  type        = string
  description = "GCS bucket name (globally unique) for models + pipeline-root."
}

variable "ar_repo" {
  type        = string
  default     = "lead-scoring"
  description = "Artifact Registry (Docker) repository id."
}

variable "force_destroy_bucket" {
  type        = bool
  default     = true
  description = "Allow `terraform destroy` to delete the bucket even if it has objects (handy for a test env)."
}
