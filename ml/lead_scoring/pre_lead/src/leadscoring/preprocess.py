"""Shared preprocessing — the single source of truth for train AND serve.

This module is baked into both the training base image and the serving image so
that the exact same code produces the feature matrix in both places. The fitted
``ColumnTransformer`` is also persisted and re-applied at serve time (never
re-fitted), which together guarantee zero training/serving skew.

Design rules (carried over from the modelling phase, do not "improve" silently):
- **Nulls are information.** Categoricals: NaN becomes its own ``MISSING``
  category. Numerics: NaN is left as ``np.nan`` (XGBoost routes it natively).
  We NEVER mean/median-impute.
- **Dynamic schema.** The feature list is derived from whatever columns exist,
  minus identifiers/target/segment, so a BigQuery schema change can't crash us.
- High-cardinality categoricals are tamed by ``OneHotEncoder(min_frequency=20)``
  rather than target encoding (which over-fits rare levels).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config

MISSING = "MISSING"

# Features engineered from `page_location` (not materialized in the table).
_UTM_CAMPAIGN_RE = re.compile(r"[?&]utm_campaign=([^&]+)")
DERIVED_COLUMNS = ("page_path", "utm_campaign")


def _utm_campaign_of(u) -> object:
    if not isinstance(u, str) or not u:
        return np.nan
    m = _UTM_CAMPAIGN_RE.search(u)
    return m.group(1) if m else np.nan


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features identically in train and serve.

    - ``page_path``    = ``page_name`` (the GA content label is the source of truth
      for the page; NOT parsed from the URL).
    - ``utm_campaign`` = REGEXP_EXTRACT(page_location, '[?&]utm_campaign=([^&]+)')

    Always adds both columns (NaN when the source is missing) so a model that lists
    them never hits a missing-column error.
    """
    out = df.copy()
    name = out["page_name"] if "page_name" in out.columns else pd.Series(
        np.nan, index=out.index
    )
    loc = out["page_location"] if "page_location" in out.columns else pd.Series(
        np.nan, index=out.index
    )
    # Empty/non-string page_name is missing (NaN), never an empty-string category.
    out["page_path"] = pd.Series(
        [v if (isinstance(v, str) and v) else np.nan for v in name],
        index=out.index, dtype=object,
    )
    out["utm_campaign"] = [_utm_campaign_of(u) for u in loc]
    return out


def resolve_features(
    df: pd.DataFrame,
    id_cols: list[str] | None = None,
    target: str = config.TARGET,
    segment_col: str = config.SEGMENT_COL,
    override: list[str] | None = None,
) -> list[str]:
    """Return the feature columns to model on.

    If ``override`` is given, keep only those that actually exist in ``df``
    (so a stale override can't reintroduce the "column not parsed" crash).
    Otherwise use every column except identifiers, target and segment.
    """
    id_cols = list(id_cols if id_cols is not None else config.ID_COLS)
    excluded = set(id_cols) | {target, segment_col}
    if override:
        return [c for c in override if c in df.columns and c not in excluded]
    return [c for c in df.columns if c not in excluded]


def split_types(df: pd.DataFrame, feats: list[str]) -> tuple[list[str], list[str]]:
    """Split features into (numeric, categorical).

    A column is numeric only if its dtype is numeric AND it is not an id-like
    column (``*_id`` or listed in ``ID_COLS``). ``product_id`` is numeric in the
    table but is an identifier of "which product" -> treat as categorical.
    """
    num, cat = [], []
    for c in feats:
        is_id_like = c.endswith("_id") or c in config.ID_COLS
        if pd.api.types.is_numeric_dtype(df[c]) and not is_id_like:
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def prep_X(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    """Build the model-ready frame, preserving nulls and avoiding the pd.NA trap.

    Numerics -> float with NaN kept. Categoricals -> python ``str`` with real
    ``np.nan`` for missing (NOT ``pd.NA``, which makes ``SimpleImputer`` raise
    "boolean value of NA is ambiguous").
    """
    out = pd.DataFrame(index=df.index)
    for c in num:
        out[c] = pd.to_numeric(df.get(c), errors="coerce").astype(float)
    for c in cat:
        s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        out[c] = pd.Series(
            np.where(s.isna(), np.nan, s.astype(str)), index=df.index, dtype=object
        )
    return out


def build_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """ColumnTransformer: numerics passthrough (NaN kept), categoricals MISSING+OHE."""
    cat_pipe = Pipeline(
        [
            ("miss", SimpleImputer(strategy="constant", fill_value=MISSING)),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=20, sparse_output=False
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", "passthrough", num), ("cat", cat_pipe, cat)],
        remainder="drop",
    )


def fit_preprocessor(
    df: pd.DataFrame, override: list[str] | None = None
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    """Resolve features, build and fit the transformer on ``df``.

    Returns ``(fitted_preprocessor, features, num, cat)``.
    """
    feats = resolve_features(df, override=override)
    num, cat = split_types(df, feats)
    pre = build_preprocessor(num, cat)
    pre.fit(prep_X(df, num, cat))
    return pre, feats, num, cat


def transform(pre: ColumnTransformer, df: pd.DataFrame, num: list[str], cat: list[str]):
    """Apply a fitted preprocessor to raw rows (handles missing columns gracefully)."""
    return pre.transform(prep_X(df, num, cat))
