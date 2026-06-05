"""Compile the KFP pipeline and (optionally) submit it to Vertex AI Pipelines.

    python pipelines/compile_and_run.py --compile-only        # just produce pipeline.json
    python pipelines/compile_and_run.py                       # compile + submit to Vertex

The training image tag is injected via the TRAINING_IMAGE env var BEFORE importing
the pipeline module, because @dsl.component captures base_image at decoration time.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `import leadscoring` work when running from the repo without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from leadscoring import config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--output", default="pipeline.json")
    p.add_argument("--n-iter", type=int, default=60)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--training-image", default=None)
    p.add_argument("--env", default=os.environ.get("ENV", config.ENV),
                   help="logical environment (dev|prod) — namespaces the GCS model paths")
    args = p.parse_args()

    env = args.env
    # Base (env-namespaced) model prefix; the pipeline appends candidate/ and live/.
    models_prefix = f"gs://{config.BUCKET}/models/{env}"

    image = args.training_image or os.environ.get("TRAINING_IMAGE") or (
        f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.AR_REPO}/training-base:latest"
    )
    os.environ["TRAINING_IMAGE"] = image  # consumed at import time below

    from kfp import compiler

    import train_pipeline  # noqa: E402  (same dir; imported after env is set)

    compiler.Compiler().compile(train_pipeline.lead_scoring_pipeline, args.output)
    print(f"compiled -> {args.output}  (base image: {image})")

    if args.compile_only:
        return

    from google.cloud import aiplatform

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION,
                    staging_bucket=f"gs://{config.BUCKET}")
    job = aiplatform.PipelineJob(
        display_name=f"lead-scoring-train-{env}",
        template_path=args.output,
        pipeline_root=config.PIPELINE_ROOT,
        parameter_values={
            "table": config.BQ_TABLE_REF,
            "project": config.PROJECT_ID,
            "models_prefix": models_prefix,
            "n_iter": args.n_iter,
            "n_seeds": args.n_seeds,
            "gate_metric": config.PROMOTION["metric"],
            "gate_min_abs": config.PROMOTION["min_abs"],
            "gate_max_regression": config.PROMOTION["max_regression"],
        },
        enable_caching=True,
    )
    job.submit()
    print(f"submitted Vertex pipeline job ({env}):", job.resource_name)


if __name__ == "__main__":
    main()
