"""Honest, multi-seed evaluation — single-split lift is unreliable (we saw 11.7x
collapse to 2.4x once averaged over seeds), so always report mean +/- std.

Ported from the notebook ``estabilidad``. Also produces the artifacts the KFP
``evaluate`` component renders in the Vertex Pipelines UI (ROC points + an HTML
lift-by-decile report).
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from . import config, preprocess, train


def holdout_stability(df: pd.DataFrame, params: dict, override=None, n_seeds: int = 5) -> dict:
    """Refit on N seeds, report mean/std of PR-AUC, ROC, lift-decil-top and top-2."""
    feats = preprocess.resolve_features(df, override=override)
    num, cat = preprocess.split_types(df, feats)
    X = preprocess.prep_X(df, num, cat)
    y = df[config.TARGET].astype(int)
    rows, best_iters = [], []
    for s in range(n_seeds):
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=s
        )
        X_ev, X_te, y_ev, y_te = train_test_split(
            X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=s
        )
        pre = preprocess.build_preprocessor(num, cat).fit(X_tr)
        A_tr, A_ev, A_te = pre.transform(X_tr), pre.transform(X_ev), pre.transform(X_te)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        m = train._xgb(spw, s, n_estimators=2000, early=True, **params)
        m.fit(A_tr, y_tr, eval_set=[(A_ev, y_ev)], verbose=False)
        p = m.predict_proba(A_te)[:, 1]
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"y": y_te.values, "d": dec}).groupby("d")["y"].mean()
        base = y_te.mean()
        rows.append(
            [
                average_precision_score(y_te, p),
                roc_auc_score(y_te, p),
                g.iloc[-1] / base,
                g.iloc[-2:].mean() / base,
            ]
        )
        best_iters.append(int(m.best_iteration or 1))
    r = np.array(rows)
    labels = ["pr_auc", "roc", "lift_top", "lift_top2"]
    summary = {
        lab: {"mean": float(r[:, i].mean()), "std": float(r[:, i].std())}
        for i, lab in enumerate(labels)
    }
    summary["best_iters"] = best_iters
    summary["median_best_iter"] = int(np.median(best_iters))
    summary["n_seeds"] = n_seeds
    return summary


def lift_by_decile(df: pd.DataFrame, params: dict, override=None, seed: int = 42) -> pd.DataFrame:
    """Decile lift table on one held-out split (for display)."""
    feats = preprocess.resolve_features(df, override=override)
    num, cat = preprocess.split_types(df, feats)
    X = preprocess.prep_X(df, num, cat)
    y = df[config.TARGET].astype(int)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_ev, X_te, y_ev, y_te = train_test_split(X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=seed)
    pre = preprocess.build_preprocessor(num, cat).fit(X_tr)
    m = train._xgb((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), seed, 2000, True, **params)
    m.fit(pre.transform(X_tr), y_tr, eval_set=[(pre.transform(X_ev), y_ev)], verbose=False)
    p = m.predict_proba(pre.transform(X_te))[:, 1]
    base = y_te.mean()
    d = pd.DataFrame({"y": y_te.values, "decile": pd.qcut(p, 10, labels=False, duplicates="drop")})
    tab = d.groupby("decile")["y"].agg(["mean", "size"]).rename(columns={"mean": "conv", "size": "n"})
    tab["lift"] = tab["conv"] / base
    tab = tab.sort_index(ascending=False).reset_index()
    tab["decile"] = tab["decile"].astype(int) + 1
    return tab, float(base), (y_te.values, p)


def test_block(y_true, scores, frac: float = 0.10) -> dict:
    """Held-out test metrics: PR-AUC + confusion matrix at the operating point.

    The confusion matrix is computed at the **top-`frac` cutoff** (flag the top 10%
    as "to call"), NOT at 0.5 — with scale_pos_weight a 0.5 threshold is meaningless.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    thr = float(np.quantile(scores, 1 - frac))
    y_pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "frac": frac,
        "threshold": thr,
        "n_test": int(len(y_true)),
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def roc_points(y_true, scores, n: int = 200):
    """Down-sampled ROC points for KFP ClassificationMetrics.

    sklearn>=1.3 sets ``thresholds[0] = np.inf`` — that serializes to ``Infinity``
    in the KFP metadata payload, which the Vertex metadata store rejects ("Failed
    to parse the output metadata payload"). Replace any non-finite threshold with
    1.0 (a score >= 1.0 flags nothing, which is the intended meaning of that point).
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    thr = np.where(np.isfinite(thr), thr, 1.0)
    if len(fpr) > n:
        idx = np.linspace(0, len(fpr) - 1, n).astype(int)
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
    return fpr.tolist(), tpr.tolist(), thr.tolist()


def html_report(segment: str, lift_tab: pd.DataFrame, base: float, stability: dict,
                test: dict | None = None) -> str:
    """Self-contained HTML report (test metrics + confusion matrix + lift + stability)."""
    chart = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(lift_tab["decile"].astype(str), lift_tab["lift"], color="#3367d6")
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.set_xlabel("decile (10 = highest score)")
        ax.set_ylabel("lift vs base")
        ax.set_title(f"{segment} — lift by decile")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        chart = f'<img src="data:image/png;base64,{b64}" style="max-width:640px"/>'
    except Exception as e:  # matplotlib optional — table still renders
        chart = f"<p><i>(chart unavailable: {e})</i></p>"

    rows = "".join(
        f"<tr><td>{int(r.decile)}</td><td>{r.conv*100:.2f}%</td>"
        f"<td>{int(r.n)}</td><td>{r.lift:.2f}x</td></tr>"
        for r in lift_tab.itertuples()
    )

    def ms(k):
        return f"{stability[k]['mean']:.4f} &plusmn; {stability[k]['std']:.4f}"

    def msx(k):
        return f"{stability[k]['mean']:.2f}x &plusmn; {stability[k]['std']:.2f}"

    test_section = ""
    if test:
        cm = test["cm"]
        pct = int(round(test["frac"] * 100))
        test_section = f"""
    <h3>Test metrics (held-out)</h3>
    <ul>
      <li><b>PR-AUC: {test['pr_auc']:.4f}</b> (base {base:.4f})</li>
      <li>ROC-AUC: {test['roc_auc']:.4f}</li>
      <li>n test: {test['n_test']}</li>
    </ul>
    <h4>Confusion matrix &mdash; operating point: top {pct}% (score &ge; {test['threshold']:.4f})</h4>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;text-align:center">
      <tr><th></th><th>pred 0 (no llamar)</th><th>pred 1 (llamar)</th></tr>
      <tr><th>real 0 (no convierte)</th><td>{cm['tn']}</td><td>{cm['fp']}</td></tr>
      <tr><th>real 1 (convierte)</th><td>{cm['fn']}</td><td style="background:#d9ead3"><b>{cm['tp']}</b></td></tr>
    </table>
    <p>precision @top{pct}% = <b>{test['precision']*100:.1f}%</b> &nbsp;|&nbsp;
       recall @top{pct}% = <b>{test['recall']*100:.1f}%</b></p>
    """

    return f"""
    <html><body style="font-family:system-ui,sans-serif">
    <h2>Lead scoring — segment: {segment}</h2>
    <p>base conversion: <b>{base*100:.2f}%</b></p>
    {test_section}
    <h3>Robust metrics (holdout, {stability.get('n_seeds')} seeds)</h3>
    <ul>
      <li>PR-AUC: <b>{ms('pr_auc')}</b></li>
      <li>ROC-AUC: <b>{ms('roc')}</b></li>
      <li>lift top decile: <b>{msx('lift_top')}</b></li>
      <li>lift top-2 deciles: <b>{msx('lift_top2')}</b></li>
    </ul>
    {chart}
    <h3>Lift by decile (single holdout)</h3>
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse">
      <tr><th>decile</th><th>conversion</th><th>n</th><th>lift</th></tr>
      {rows}
    </table>
    </body></html>
    """
