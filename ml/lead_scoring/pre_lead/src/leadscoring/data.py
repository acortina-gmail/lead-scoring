"""BigQuery I/O for training."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, preprocess


def load(table_ref: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load the training table from BigQuery into a DataFrame.

    Adds the ``segmento`` routing column if the table doesn't already carry it
    (unbounce forms -> 'landing', else 'main').
    """
    from google.cloud import bigquery

    table_ref = table_ref or config.BQ_TABLE_REF
    # location is REQUIRED for non-US datasets (EU here) — without it the client
    # submits the query in "US" and fails with "Dataset not found in location US".
    client = bigquery.Client(project=config.PROJECT_ID, location=config.BQ_LOCATION)
    sql = f"SELECT * FROM `{table_ref}`"
    if limit:
        sql += f" LIMIT {int(limit)}"
    df = client.query(sql).to_dataframe()

    if config.SEGMENT_COL not in df.columns:
        form = df.get("form_name", pd.Series("", index=df.index)).astype(str)
        df[config.SEGMENT_COL] = np.where(
            form.str.lower().str.startswith("unbounce"), "landing", "main"
        )
    # Engineer page_path / utm_campaign from page_location (same code path as serving).
    df = preprocess.derive_columns(df)
    return df


def segment_frame(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Subset to one segment."""
    return df[df[config.SEGMENT_COL] == segment].copy()
