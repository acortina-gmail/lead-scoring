# CI/CD — GitHub Actions

GitHub Actions runs the quality gate, builds images, and deploys serving. It does
**not** replace the manual deploy scripts — it *calls* them (`deploy/01_*.sh`,
`deploy/03_*.sh`, `deploy/02_run_pipeline.sh`), so local and CI deploys stay identical.

## Workflows (`.github/workflows/`)

| File         | Trigger                       | Does                                                              |
|--------------|-------------------------------|------------------------------------------------------------------|
| `ci.yml`     | every PR + push to `main`     | `ruff check` + `pytest` (no GCP creds)                            |
| `deploy.yml` | push to `main` (or manual)    | quality gate → build images → deploy **dev** → deploy **prod** (approval) |
| `train.yml`  | manual (`workflow_dispatch`)  | compile + submit the Vertex training pipeline for `dev`/`prod`    |

**The dev→prod gate.** The repo is **public**, so GitHub's required-reviewer Environment
protection is available. `deploy.yml`'s `deploy-prod` job targets the **`prod` Environment**,
which has a required reviewer — so prod deploys **pause until approved** in the Actions UI.
A prod `train.yml` run hits the same environment. The candidate→live *model* gate is
separate and lives inside the Vertex pipeline.

Retraining is intentionally **not** wired into merge — a Vertex run is slow/costly. Run
`train.yml` by hand when you want a new model, then re-run `deploy.yml` (or
`deploy/03_deploy_serving.sh`) so serving picks up the freshly-promoted `live` model.

## One-time setup (requires GCP + GitHub console)

These can't be automated from this repo (the deployer account can't set IAM policy, so
Workload Identity Federation isn't an option yet — we use a key instead).

1. **Create a service-account key.** Use the default compute SA (what the workloads
   already run as) or a dedicated `gha-deployer` SA with equivalent roles:
   ```bash
   gcloud iam service-accounts keys create key.json \
     --iam-account <SA_EMAIL> --project test-ml-flow-484314
   ```
   The SA needs: Cloud Build (`roles/cloudbuild.builds.editor`), Artifact Registry
   write, Storage object admin (models bucket), Vertex AI user, and Cloud Run admin —
   the project's editor-role compute SA already covers these.

2. **Store it as a GitHub secret.** Repo → Settings → Secrets and variables → Actions →
   New repository secret, name **`GCP_SA_KEY`**, value = full contents of `key.json`.

3. **Prod gate.** Already handled — the `prod` Environment has a required reviewer
   (`acortina-gmail`), so `deploy-prod` pauses for approval. To change reviewers:
   repo → Settings → Environments → `prod`.

4. **Delete the local key** — `rm key.json`. It's gitignored (`**/key.json`) but never
   commit one. Rotate the key periodically.

5. *(Optional)* Override `PROJECT_ID` / `REGION` / `BUCKET` as repo **Variables** if they
   ever differ from the `deploy/config.sh` defaults.

Steps 1–3 are already done for this repo (key uploaded as `GCP_SA_KEY`, `prod`
Environment created). They're documented here for re-setup / key rotation.

## Teardown

`terraform destroy` alone is **not** enough — Terraform here is infra-only (bucket + AR
repo + APIs), so it leaves the Cloud Run services and the CI key behind. Use the script,
which does it in the right order (Cloud Run → bucket + AR repo → optional SA key):

```bash
./deploy/99_teardown.sh                  # interactive: type the project id to confirm
./deploy/99_teardown.sh --yes            # non-interactive
./deploy/99_teardown.sh --delete-sa-keys # also delete the CI key (BREAKS GitHub deploy)
```

It prefers `terraform destroy` when TF state exists, else deletes the bucket + AR repo
directly with `gcloud` (infra may have been created via `00_setup_gcp.sh`). It never
touches the BigQuery source table, the default compute SA, or the enabled APIs. If you
pass `--delete-sa-keys`, also remove the `GCP_SA_KEY` secret from the GitHub repo.

## Migrating to keyless later

Once an account with `setIamPolicy` is available, switch to **Workload Identity
Federation**: create a pool + provider, grant the SA `roles/iam.workloadIdentityUser`
for the repo, and replace `credentials_json` with `workload_identity_provider` +
`service_account` in each `google-github-actions/auth@v2` step. Then delete `GCP_SA_KEY`.
