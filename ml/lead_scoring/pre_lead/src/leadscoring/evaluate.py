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


def capacity_table(
    y_true,
    scores,
    base: float,
    daily_volume: float,
    cuts=(2, 5, 10, 15, 20, 30, 50, 60, 70, 80, 90),
) -> pd.DataFrame:
    """Cumulative-gains ("capacity") table — the client-facing view of the model.

    For each ``Top P%`` of leads (ranked by score, highest first) it answers: if we
    call that slice, how many leads/day is it, what conversion rate do we hit, what
    share of all conversions do we capture, and how many times better than random.

    Rank-based (take the first ``k = round(P% · n)`` by score) rather than a quantile
    threshold, so ties in the (discrete) score don't distort the cut. ``daily_volume``
    only scales the per-day columns; the rates and lift are volume-independent.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    order = np.argsort(-s, kind="stable")  # highest score first
    y_sorted = y[order]
    total_conv = max(int(y.sum()), 1)
    rows = []
    for p in cuts:
        k = max(int(round(p / 100 * n)), 1)
        top = y_sorted[:k]
        tasa = float(top.mean())
        leads_dia = p / 100 * daily_volume
        rows.append(
            {
                "top_pct": p,
                "leads_dia": leads_dia,
                "conv_dia": leads_dia * tasa,
                "tasa_exito": tasa,
                "pct_capturadas": float(top.sum()) / total_conv,
                "vs_azar": (tasa / base) if base else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def grade_thresholds(scores) -> dict:
    """Fit the A/B/C/D score cutoffs from a score distribution (``config.GRADE_BANDS``).

    Returns ``{"A": q90, "B": q70, "C": q40}`` — the score at each band's lower
    percentile. Stored in the artifact so serving can grade a live score the same way.
    """
    s = np.asarray(scores, dtype=float)
    return {g: float(np.quantile(s, q / 100)) for g, q in config.GRADE_BANDS}


def grade_table(y_true, scores, base: float, daily_volume: float) -> pd.DataFrame:
    """Per-grade legend (NON-cumulative): for each A/B/C/D band, its conversion rate
    and lift vs random, so the grade returned by ``/score`` is readable as a number.

    Bands come from ``config.GRADE_BANDS`` (A = top 10%, B = 10–30%, C = 30–60%,
    D = bottom 40%), sliced on the rank-sorted leads.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    y_sorted = y[np.argsort(-s, kind="stable")]
    # Cumulative upper edge of each band as a percentile-from-top (e.g. 75 -> 25%).
    uppers = [100 - q for _, q in config.GRADE_BANDS] + [100]  # [25, 50, 100]
    grades = [g for g, _ in config.GRADE_BANDS] + [config.GRADE_FALLBACK]
    rows = []
    prev = 0
    for g, up in zip(grades, uppers):
        lo, hi = int(round(prev / 100 * n)), int(round(up / 100 * n))
        seg = y_sorted[lo:hi]
        tasa = float(seg.mean()) if len(seg) else float("nan")
        rows.append(
            {
                "grade": g,
                "banda": f"{prev}–{up}%",
                "leads_dia": (hi - lo) / max(n, 1) * daily_volume,
                "tasa_exito": tasa,
                "vs_azar": (tasa / base) if base else float("nan"),
            }
        )
        prev = up
    return pd.DataFrame(rows)


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


def _grades_section(grade_tab: pd.DataFrame) -> str:
    """Client-facing HTML: the A/B/C grade legend (conversion + lift per band)."""
    g_rows = "".join(
        f"<tr><td style='text-align:center'><b>{r.grade}</b></td><td>{r.banda}</td>"
        f"<td>~{r.leads_dia:.0f}</td><td>{r.tasa_exito*100:.1f}%</td>"
        f"<td>{r.vs_azar:.1f}x</td></tr>"
        for r in grade_tab.itertuples()
    )
    return f"""
    <h3>Grados A / B / C (por lead)</h3>
    <p>Cada lead recibe un grado segun su posicion en el ranking del modelo:
       A = el 25% con mas probabilidad de convertir, C = el 50% con menos.</p>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;text-align:right">
      <tr style="background:#1c4587;color:#fff;text-align:center">
        <th>Grado</th><th>Banda</th><th>Leads/dia</th><th>Tasa de exito</th><th>vs. azar</th></tr>
      {g_rows}
    </table>
    <hr/>
    """


def html_report(segment: str, lift_tab: pd.DataFrame, base: float, stability: dict,
                test: dict | None = None, capacity: pd.DataFrame | None = None,
                grade_tab: pd.DataFrame | None = None) -> str:
    """Self-contained HTML report: headline PR-AUC, the A/B/C grade legend (when
    ``grade_tab`` is given), then the team-facing robust metrics + lift-by-decile.
    (``test``/``capacity`` are accepted for back-compat but no longer rendered.)
    """
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

    def msx(k):
        return f"{stability[k]['mean']:.2f}x &plusmn; {stability[k]['std']:.2f}"

    grades_section = _grades_section(grade_tab) if grade_tab is not None else ""

    pr_mean = stability["pr_auc"]["mean"]
    pr_std = stability["pr_auc"]["std"]
    prauc_headline = f"""
    <div style="margin:14px 0;padding:16px 20px;background:#eef3fb;border-radius:8px;display:inline-block">
      <div style="font-size:14px;color:#333">PR-AUC <span style="color:#888">(holdout, {stability.get('n_seeds')} seeds)</span></div>
      <span style="font-size:42px;font-weight:700;color:#1c4587">{pr_mean:.3f}</span>
      <span style="font-size:18px;color:#666">&plusmn; {pr_std:.3f}</span>
      <span style="font-size:14px;color:#888">&nbsp; vs base {base*100:.1f}%</span>
    </div>
    """

    return f"""
    <html><body style="font-family:system-ui,sans-serif">
    <h2>Lead scoring — segment: {segment}</h2>
    <p>base conversion: <b>{base*100:.2f}%</b></p>
    {prauc_headline}
    {grades_section}
    <h3>Robust metrics (holdout, {stability.get('n_seeds')} seeds)</h3>
    <ul>
      <li>ROC-AUC: <b>{stability['roc']['mean']:.4f} &plusmn; {stability['roc']['std']:.4f}</b></li>
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
