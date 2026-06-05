"""Train/serve parity tests for the derived columns.

`page_path` and `utm_campaign` are DERIVED from `page_location` by the same
`derive_columns` in both training (`data.load`) and serving (`app.py`). If this
drifts, the model sees different features at serve time than it trained on — the
single most important invariant in this repo. Nulls must be preserved as NaN,
never imputed.
"""

import numpy as np
import pandas as pd

from leadscoring import preprocess


def test_derive_columns_extracts_path_and_campaign():
    df = pd.DataFrame({"page_location": ["https://obs.edu/master-mba?utm_campaign=spring"]})
    out = preprocess.derive_columns(df)
    assert out.loc[0, "page_path"] == "/master-mba"
    assert out.loc[0, "utm_campaign"] == "spring"


def test_derive_columns_campaign_among_other_params():
    df = pd.DataFrame(
        {"page_location": ["https://obs.edu/x?gclid=abc&utm_campaign=fall&foo=1"]}
    )
    out = preprocess.derive_columns(df)
    assert out.loc[0, "page_path"] == "/x"
    assert out.loc[0, "utm_campaign"] == "fall"


def test_derive_columns_missing_is_nan_not_imputed():
    # None / empty / non-string must yield NaN (XGBoost-native missing), not a default.
    df = pd.DataFrame({"page_location": [None, "", 123]})
    out = preprocess.derive_columns(df)
    assert out["page_path"].isna().all()
    assert out["utm_campaign"].isna().all()


def test_derive_columns_no_campaign_param_is_nan():
    df = pd.DataFrame({"page_location": ["https://obs.edu/home"]})
    out = preprocess.derive_columns(df)
    assert out.loc[0, "page_path"] == "/home"
    assert out["utm_campaign"].isna().all()


def test_derive_columns_always_adds_both_when_page_location_absent():
    # A schema without page_location must not crash; both columns appear as NaN.
    df = pd.DataFrame({"product_id": [1, 2]})
    out = preprocess.derive_columns(df)
    assert list(preprocess.DERIVED_COLUMNS) == ["page_path", "utm_campaign"]
    for col in preprocess.DERIVED_COLUMNS:
        assert col in out.columns
        assert out[col].isna().all()
    # Original frame is untouched (derive_columns copies).
    assert "page_path" not in df.columns
    assert np.isnan(out["page_path"]).all()
