"""features.py -- cell 28, and the passthrough -> drop change."""

import numpy as np
import pandas as pd
import pytest

from disease_pred.features import build_preprocessor, output_names


@pytest.fixture(scope="module")
def fitted(prepared):
    spec = prepared["spec"]
    pre = build_preprocessor(spec["numeric"], spec["binary"], spec["categorical"],spec["indicators"])
    pre.fit(prepared["X_train"], prepared["y_train"])
    return pre


def test_remainder_is_drop(fitted):
    """passthrough would let an unexpected payload column into the design matrix."""
    assert fitted.remainder == "drop"


def test_unexpected_column_is_ignored(fitted, prepared):
    X = prepared["X_train"]
    n_cols = fitted.transform(X).shape[1]

    polluted = X.copy()
    polluted["patient_id"] = np.arange(len(polluted))
    assert fitted.transform(polluted).shape[1] == n_cols


def test_transform_leaves_no_missing_values(fitted, prepared):
    Z = fitted.transform(prepared["X_train"])
    assert not np.isnan(Z).any()


def test_scaling_uses_training_statistics(fitted, prepared):
    """A single row must be scaled with the TRAINING mean/sd, not its own."""
    X = prepared["X_train"]
    one = X.iloc[[0]]
    full = fitted.transform(X)
    assert np.allclose(fitted.transform(one)[0], full[0])


def test_imputer_fills_from_training_median(fitted, prepared):
    spec = prepared["spec"]
    X = prepared["X_train"]

    blank = X.iloc[[0]].copy()
    blank[spec["numeric"]] = np.nan
    Z = fitted.transform(blank)

    names = output_names(fitted)
    medians = fitted.named_transformers_["num"].named_steps["imputer"].statistics_
    scaler = fitted.named_transformers_["num"].named_steps["scalar"]
    expected = (medians - scaler.mean_) / scaler.scale_

    for col, want in zip(spec["numeric"], expected):
        assert Z[0, names.index(col)] == pytest.approx(want)


def test_output_names_strip_the_block_prefix(fitted):
    names = output_names(fitted)
    assert all("__" not in n for n in names)
    assert len(names) == len(set(names))
    assert any(n.startswith("missingindicator_") for n in names)

def flagged_columns(pre) -> set[str]:
    return {n.removeprefix("missingindicator_") for n in output_names(pre)
            if n.startswith("missingindicator_")}


def test_missing_indicator_block_present(fitted):
    """Cells 25-27 justified keeping it; a silent removal should fail here."""
    assert flagged_columns(fitted), "the MissingIndicator block disappeared"


def test_only_the_declared_columns_get_an_indicator(fitted, prepared):
    """Cells 25-27 did not keep the whole numeric+binary block.

    chol's indicator beat 0.5 AUC on its own, which made it a site proxy
    (VA/Switzerland) rather than signal, and fbs came off the same battery.
    Flagging numeric+binary wholesale puts both back in; this is the assertion
    that notices.
    """
    assert flagged_columns(fitted) == set(prepared["spec"]["indicators"])


@pytest.mark.parametrize("rejected", ["chol", "fbs"])
def test_rejected_indicators_stay_out(fitted, prepared, rejected):
    assert rejected in prepared["spec"]["features"], "guard: column still modelled"
    assert rejected not in flagged_columns(fitted)


def test_no_indicator_block_when_the_list_is_empty(prepared):
    """An empty list must drop the block, not hand ColumnTransformer []."""
    spec = prepared["spec"]
    pre = build_preprocessor(spec["numeric"], spec["binary"],
                             spec["categorical"], [])
    pre.fit(prepared["X_train"], prepared["y_train"])
    assert "missing" not in {name for name, _, _ in pre.transformers}
    assert not flagged_columns(pre)


def test_indicator_fires_for_a_column_complete_in_training(prepared):
    """Why features='all' on an explicit list, and why error_on_new is moot.

    age is never missing in the four source files, so features='missing-only'
    would fit no column for it and a payload without an age would raise (or,
    with error_on_new=False, silently lose the flag). 'all' fits the column
    anyway -- all zeros in training -- so the flag exists at predict time.
    """
    spec = prepared["spec"]
    pre = build_preprocessor(spec["numeric"], spec["binary"],
                             spec["categorical"], ["age"])
    pre.fit(prepared["X_train"], prepared["y_train"])

    assert prepared["X_train"]["age"].notna().all(), "guard: age was complete"
    col = output_names(pre).index("missingindicator_age")

    present = prepared["X_train"].iloc[[0]].copy()
    assert pre.transform(present)[0, col] == 0

    absent = present.copy()
    absent["age"] = np.nan
    assert pre.transform(absent)[0, col] == 1


def test_unseen_category_does_not_explode(fitted, prepared):
    spec = prepared["spec"]
    row = prepared["X_train"].iloc[[0]].copy()
    row[spec["categorical"][0]] = 999.0
    fitted.transform(row)          # handle_unknown='infrequent_if_exist'
