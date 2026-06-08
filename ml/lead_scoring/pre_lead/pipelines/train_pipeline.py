"""Vertex AI Pipeline (KFP v2) — train both segment models with rich UI metrics.

DAG (built once per segment so each emits its own metrics/HTML in the Vertex UI):

    ingest ─► [ prepare_segment ─► fit_preprocess ─► train_model ─► evaluate_model ─► package_artifact ]

All components run the custom ``training-base`` image (which has the ``leadscoring``
package installed), so they just ``import leadscoring``. Preprocessing
(``fit_preprocess``) is a SEPARATE component from training (``train_model``); the
fitted ``ColumnTransformer`` it outputs is the exact object reused at serve time.

Compile/submit with ``pipelines/compile_and_run.py``.

NOTE: do NOT add ``from __future__ import annotations`` here — KFP v2 component
introspection needs real (non-stringized) annotations.
"""
import os

from kfp import dsl
from kfp.dsl import (
    HTML,
    Artifact,
    ClassificationMetrics,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
)

# Base image with the leadscoring library. Overridable so compile_and_run can inject
# the Artifact Registry path it just built.
BASE_IMAGE = os.environ.get(
    "TRAINING_IMAGE",
    "europe-west1-docker.pkg.dev/bq-pfu-ga4/lead-scoring/training-base:latest",
)


@dsl.component(base_image=BASE_IMAGE)
def ingest(table: str, project: str, data: Output[Dataset]):
    """Load the BigQuery training table to parquet."""
    from leadscoring import config
    from leadscoring import data as dataio

    config.PROJECT_ID = project
    df = dataio.load(table_ref=table)
    df.to_parquet(data.path)
    data.metadata["rows"] = len(df)
    data.metadata["columns"] = list(df.columns)


@dsl.component(base_image=BASE_IMAGE)
def prepare_segment(data: Input[Dataset], segment: str, seg_out: Output[Dataset]):
    """Subset the dataset to a single segment."""
    import pandas as pd

    from leadscoring import data as dataio

    df = pd.read_parquet(data.path)
    seg = dataio.segment_frame(df, segment)
    seg.to_parquet(seg_out.path)
    seg_out.metadata["segment"] = segment
    seg_out.metadata["rows"] = len(seg)


@dsl.component(base_image=BASE_IMAGE)
def fit_preprocess(
    seg: Input[Dataset], override_json: str, preprocessor: Output[Artifact]
):
    """Fit the ColumnTransformer on 100% of the segment (the serving transformer)."""
    import json

    import joblib
    import pandas as pd

    from leadscoring import preprocess

    df = pd.read_parquet(seg.path)
    override = json.loads(override_json) if override_json else None
    pre, feats, num, cat = preprocess.fit_preprocessor(df, override=override)
    joblib.dump(
        {"preprocessor": pre, "features": feats, "num": num, "cat": cat},
        preprocessor.path,
    )
    preprocessor.metadata.update({"n_features": len(feats), "num": num, "cat": cat})


@dsl.component(base_image=BASE_IMAGE)
def train_model(
    seg: Input[Dataset],
    preprocessor: Input[Artifact],
    n_iter: int,
    n_seeds: int,
    model: Output[Model],
):
    """Tune (leak-free) + multi-seed stability + refit deployable model on 100%.

    Uses the preprocessor produced by ``fit_preprocess`` for the final fit.
    Robust metrics + params are stored on the model artifact metadata for the
    evaluate/package steps.
    """
    import joblib
    import pandas as pd

    from leadscoring import evaluate, train

    df = pd.read_parquet(seg.path)
    bundle = joblib.load(preprocessor.path)
    feats = bundle["features"]

    tuned = train.tune_segment(df, override=feats, n_iter=n_iter)
    stability = evaluate.holdout_stability(df, tuned["params"], override=feats, n_seeds=n_seeds)

    xgb_model = train.fit_with_preprocessor(
        df, bundle["preprocessor"], bundle["num"], bundle["cat"],
        tuned["params"], stability["median_best_iter"],
    )
    xgb_model.save_model(model.path)  # native xgb json (portable)

    meta = {
        "params": tuned["params"],
        "n_estimators": stability["median_best_iter"],
        "test_metrics": tuned["test_metrics"],
        "stability": stability,
        "base_rate": float(df["y"].mean()),
        "n_train": int(len(df)),
        "features": feats,
        "num": bundle["num"],
        "cat": bundle["cat"],
    }
    model.metadata.update({k: meta[k] for k in ("n_estimators", "base_rate", "n_train")})
    joblib.dump(meta, model.path + ".meta.joblib")


@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(
    seg: Input[Dataset],
    data: Input[Dataset],
    model: Input[Model],
    segment: str,
    daily_volume: int,
    metrics: Output[Metrics],
    cls_metrics: Output[ClassificationMetrics],
    report: Output[HTML],
):
    """Emit scalar Metrics + ROC curve + an HTML lift report to the Vertex UI."""
    import joblib
    import pandas as pd

    from leadscoring import evaluate

    df = pd.read_parquet(seg.path)
    # Segment's slice of the global daily volume, by its share of the training rows.
    n_total = len(pd.read_parquet(data.path))
    seg_daily = daily_volume * (len(df) / max(n_total, 1))
    meta = joblib.load(model.path + ".meta.joblib")
    params, feats, stab = meta["params"], meta["features"], meta["stability"]

    # Scalar robust metrics (what the team reads first)
    metrics.log_metric("pr_auc", stab["pr_auc"]["mean"])
    metrics.log_metric("pr_auc_std", stab["pr_auc"]["std"])
    metrics.log_metric("roc_auc", stab["roc"]["mean"])
    metrics.log_metric("lift_top_decile", stab["lift_top"]["mean"])
    metrics.log_metric("lift_top2_deciles", stab["lift_top2"]["mean"])
    metrics.log_metric("base_rate", meta["base_rate"])
    metrics.log_metric("n_train", meta["n_train"])

    # ROC curve + decile lift + test metrics (PR-AUC + confusion matrix) on a held-out split
    lift_tab, base, (y_true, scores) = evaluate.lift_by_decile(df, params, override=feats)
    test = evaluate.test_block(y_true, scores)
    metrics.log_metric("test_pr_auc", test["pr_auc"])
    metrics.log_metric("test_precision_top10", test["precision"])
    metrics.log_metric("test_recall_top10", test["recall"])
    metrics.log_metric("seg_daily_volume", seg_daily)

    # Client-facing tables (built on the SAME held-out scores, so the rates are honest).
    capacity = evaluate.capacity_table(y_true, scores, base, seg_daily)
    grade_tab = evaluate.grade_table(y_true, scores, base, seg_daily)
    import math

    fpr, tpr, thr = evaluate.roc_points(y_true, scores)
    # sklearn>=1.3 makes thr[0]=inf -> "Infinity" in the KFP payload, which Vertex's
    # metadata store rejects. Clamp to finite so log_roc_curve serializes cleanly.
    # (Done here too so the fix applies without rebuilding the training image, since
    # the component source is embedded in the compiled pipeline.)
    thr = [t if math.isfinite(t) else 1.0 for t in thr]
    cls_metrics.log_roc_curve(fpr, tpr, thr)

    with open(report.path, "w") as f:
        f.write(evaluate.html_report(segment, lift_tab, base, stab, test,
                                     capacity=capacity, grade_tab=grade_tab))


@dsl.component(base_image=BASE_IMAGE)
def package_artifact(
    seg: Input[Dataset],
    preprocessor: Input[Artifact],
    model: Input[Model],
    segment: str,
    candidate_uri: str,
):
    """Bundle {preprocessor, model, features, metrics} -> joblib -> GCS as the CANDIDATE.

    Writes to the ``candidate`` stage only; ``validate_and_promote`` decides whether
    this becomes the ``live`` model that serving loads.
    """
    import joblib
    import pandas as pd
    import xgboost as xgb

    from leadscoring import config, evaluate, preprocess

    bundle = joblib.load(preprocessor.path)
    meta = joblib.load(model.path + ".meta.joblib")
    clf = xgb.XGBClassifier()
    clf.load_model(model.path)

    # A/B/C/D cutoffs from the PRODUCTION model's own score distribution, so a live
    # score grades consistently with the deployed model (serving reads grade_thresholds).
    df = pd.read_parquet(seg.path)
    X = preprocess.transform(bundle["preprocessor"], df, bundle["num"], bundle["cat"])
    grade_thr = evaluate.grade_thresholds(clf.predict_proba(X)[:, 1])

    artifact = {
        "preprocessor": bundle["preprocessor"],
        "model": clf,
        "features": bundle["features"],
        "num": bundle["num"],
        "cat": bundle["cat"],
        "segmento": segment,
        "params": meta["params"],
        "n_estimators": meta["n_estimators"],
        "base_rate": meta["base_rate"],
        "n_train": meta["n_train"],
        "metrics": meta["stability"],
        "grade_thresholds": grade_thr,
        "schema_version": 2,
    }
    local = f"/tmp/lead_scoring_{segment}.joblib"
    joblib.dump(artifact, local)

    # upload to GCS (candidate stage)
    from google.cloud import storage

    assert candidate_uri.startswith("gs://"), candidate_uri
    bkt, _, blob = candidate_uri[len("gs://"):].partition("/")
    storage.Client(project=config.PROJECT_ID).bucket(bkt).blob(blob).upload_from_filename(local)
    print(f"uploaded candidate {candidate_uri}")


@dsl.component(base_image=BASE_IMAGE)
def validate_and_promote(
    model: Input[Model],
    segment: str,
    candidate_uri: str,
    live_uri: str,
    metric: str,
    min_abs: float,
    max_regression: float,
    decision: Output[Metrics],
    report: Output[HTML],
):
    """Promote the candidate to 'live' only if it doesn't regress vs current live.

    SOFT gate: on failure it keeps the existing live model and records the reason,
    but never raises — the pipeline stays green and a human reviews the report.

    Compares the honest multi-seed ``metric`` (e.g. ``lift_top``) mean:
      * sanity:        candidate >= min_abs            (beats random)
      * no-regression: candidate >= live - max_regression
    First run (no live model yet) bootstraps: promote if sanity passes.
    """
    import tempfile

    import joblib
    from google.cloud import storage

    from leadscoring import config

    client = storage.Client(project=config.PROJECT_ID)

    def _read_metric(uri):
        """Return the gate metric mean from a joblib at `uri`, or None if absent."""
        bkt, _, blob = uri[len("gs://"):].partition("/")
        b = client.bucket(bkt).blob(blob)
        if not b.exists():
            return None
        fd, local = tempfile.mkstemp(suffix=".joblib")
        import os as _os

        _os.close(fd)
        b.download_to_filename(local)
        art = joblib.load(local)
        return float(art["metrics"][metric]["mean"])

    # Candidate metric comes from the model meta written by train_model.
    meta = joblib.load(model.path + ".meta.joblib")
    cand = float(meta["stability"][metric]["mean"])
    live = _read_metric(live_uri)

    sane = cand >= min_abs
    if not sane:
        promote, reason = False, f"candidate {metric}={cand:.3f} < min {min_abs} (no better than random)"
    elif live is None:
        promote, reason = True, f"no live model yet — bootstrap promote (candidate {metric}={cand:.3f})"
    elif cand >= live - max_regression:
        promote, reason = True, f"candidate {metric}={cand:.3f} >= live {live:.3f} - {max_regression} (ok)"
    else:
        promote, reason = False, f"REGRESSION: candidate {metric}={cand:.3f} < live {live:.3f} - {max_regression}"

    if promote:
        # copy candidate -> live within GCS (no re-upload)
        cb, _, cblob = candidate_uri[len("gs://"):].partition("/")
        lb, _, lblob = live_uri[len("gs://"):].partition("/")
        src_bucket = client.bucket(cb)
        src_bucket.copy_blob(src_bucket.blob(cblob), client.bucket(lb), lblob)
        print(f"PROMOTED {segment}: {candidate_uri} -> {live_uri} ({reason})")
    else:
        print(f"NOT promoted {segment}: live model kept. {reason}")

    decision.log_metric("promoted", 1.0 if promote else 0.0)
    decision.log_metric("candidate_lift", cand)
    decision.log_metric("live_lift", live if live is not None else -1.0)

    color = "#d9ead3" if promote else "#f4cccc"
    verdict = "PROMOTED ✅" if promote else "NOT promoted — live kept ⛔"
    live_txt = f"{live:.3f}" if live is not None else "(none — first run)"
    with open(report.path, "w") as f:
        f.write(f"""
        <html><body style="font-family:system-ui,sans-serif">
        <h2>Promotion decision — segment: {segment}</h2>
        <div style="background:{color};padding:10px;border-radius:6px"><b>{verdict}</b></div>
        <ul>
          <li>gate metric: <b>{metric}</b> (multi-seed mean)</li>
          <li>candidate: <b>{cand:.3f}</b></li>
          <li>current live: <b>{live_txt}</b></li>
          <li>rule: candidate &ge; {min_abs} (sanity) and &ge; live - {max_regression}</li>
          <li>reason: {reason}</li>
        </ul>
        <p><small>SOFT gate: a failure keeps the live model and never fails the pipeline.</small></p>
        </body></html>
        """)


@dsl.pipeline(name="lead-scoring-train", description="Segmented lead-scoring train + package")
def lead_scoring_pipeline(
    table: str,
    project: str,
    models_prefix: str,
    n_iter: int = 60,
    n_seeds: int = 5,
    daily_volume: int = 250,
    gate_metric: str = "lift_top",
    gate_min_abs: float = 1.0,
    gate_max_regression: float = 0.15,
):
    from leadscoring import config as cfg

    SEGMENTS = cfg.SEGMENTS
    OVERRIDES = cfg.FEATURE_OVERRIDES  # frozen feature lists from config.py

    raw = ingest(table=table, project=project)
    for segment in SEGMENTS:
        seg = prepare_segment(data=raw.outputs["data"], segment=segment)
        seg.set_display_name(f"prepare-{segment}")

        import json

        pre = fit_preprocess(
            seg=seg.outputs["seg_out"],
            override_json=json.dumps(OVERRIDES.get(segment, [])),
        )
        pre.set_display_name(f"preprocess-{segment}")

        trained = train_model(
            seg=seg.outputs["seg_out"],
            preprocessor=pre.outputs["preprocessor"],
            n_iter=n_iter,
            n_seeds=n_seeds,
        )
        trained.set_display_name(f"train-{segment}")

        ev = evaluate_model(
            seg=seg.outputs["seg_out"],
            data=raw.outputs["data"],
            model=trained.outputs["model"],
            segment=segment,
            daily_volume=daily_volume,
        )
        ev.set_display_name(f"evaluate-{segment}")

        candidate_uri = f"{models_prefix}/candidate/lead_scoring_{segment}.joblib"
        live_uri = f"{models_prefix}/live/lead_scoring_{segment}.joblib"

        pkg = package_artifact(
            seg=seg.outputs["seg_out"],
            preprocessor=pre.outputs["preprocessor"],
            model=trained.outputs["model"],
            segment=segment,
            candidate_uri=candidate_uri,
        )
        pkg.set_display_name(f"package-{segment}")

        promote = validate_and_promote(
            model=trained.outputs["model"],
            segment=segment,
            candidate_uri=candidate_uri,
            live_uri=live_uri,
            metric=gate_metric,
            min_abs=gate_min_abs,
            max_regression=gate_max_regression,
        )
        promote.after(pkg)  # the candidate joblib must exist before we can promote it
        promote.set_display_name(f"validate-and-promote-{segment}")
