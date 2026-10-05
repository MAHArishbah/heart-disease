"""data.py -- cells 1, 8, 13, 23, 24."""

import numpy as np
import pandas as pd
import pytest

from disease_pred import data as data_mod
from disease_pred.config import PARAMS


def test_load_raw_tags_every_site(raw):
    assert set(raw["site"]) == set(PARAMS["data"]["files"])
    assert list(raw.columns) == data_mod.COLS + ["site"]
    assert len(raw) == 920


def test_question_marks_became_nan(raw):
    # The UCI files use '?' for missing; na_values must catch it everywhere.
    assert not raw.select_dtypes("object").drop(columns="site").columns.size


def test_clean_zeroes_out_impossible_values(raw, cleaned):
    for col in PARAMS["data"]["zero_is_missing"]:
        assert (raw[col] == 0).sum() > 0, f"{col} had no zeros to fix"
        assert (cleaned[col] == 0).sum() == 0
        assert cleaned[col].isna().sum() >= (raw[col] == 0).sum()


def test_clean_builds_binary_target_and_drops_num(cleaned):
    assert "num" not in cleaned.columns
    assert set(cleaned["target"].unique()) == {0, 1}


def test_clean_target_matches_num_gt_zero(raw):
    pre = data_mod.clean(raw, drop_duplicates=False)
    assert (pre["target"] == (raw["num"] > 0).astype(int)).all()


def test_clean_drops_duplicates(raw, cleaned):
    pre = data_mod.clean(raw, drop_duplicates=False)
    assert pre.duplicated().sum() > 0, "nothing to dedup -- cell 13 was a no-op"
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) < len(pre)


def test_clean_does_not_mutate_input(raw):
    before = raw.copy()
    data_mod.clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_select_features_sees_only_the_frame_it_is_given(cleaned):
    """The cell-23 leakage fix.

    If select_features consulted anything but its argument, the column list
    from a deliberately mangled frame would come back unchanged.
    """
    mangled = cleaned.copy()
    mangled["chol"] = np.nan          # 100% missing -> must be dropped

    assert "chol" in data_mod.select_features(cleaned)["features"]
    assert "chol" not in data_mod.select_features(mangled)["features"]
    assert "chol" in data_mod.select_features(mangled)["dropped_high_missing"]


def test_select_features_honours_the_threshold(cleaned):
    spec = data_mod.select_features(cleaned)
    thresh = PARAMS["features"]["missing_threshold"]
    for col in spec["features"]:
        if col == "site":
            continue
        assert cleaned[col].isna().mean() <= thresh
    for col in spec["dropped_high_missing"]:
        assert cleaned[col].isna().mean() > thresh


def test_select_features_partitions_without_overlap(cleaned):
    spec = data_mod.select_features(cleaned)
    blocks = spec["numeric"] + spec["binary"] + spec["categorical"]
    assert blocks == spec["features"]
    assert len(set(blocks)) == len(blocks)
    assert "target" not in blocks and "num" not in blocks


def test_indicators_come_from_yaml_not_a_literal(cleaned):
    """The list is a declared role, like `roles` and `zero_is_missing`."""
    spec = data_mod.select_features(cleaned)
    assert spec["indicators"] == PARAMS["features"]["missing_indicators"]


def test_indicators_are_a_subset_of_the_flaggable_blocks(cleaned):
    """MissingIndicator only ever sees numeric+binary; a nominal name there
    would be handed to a block that never receives that column."""
    spec = data_mod.select_features(cleaned)
    assert set(spec["indicators"]) <= set(spec["numeric"] + spec["binary"])


def test_a_high_missing_column_cannot_stay_an_indicator(cleaned):
    """The whole reason the list is resolved here and not hardcoded in
    features.py: a dropped column handed to ColumnTransformer is a KeyError
    at fit time."""
    mangled = cleaned.copy()
    mangled["thalach"] = np.nan

    assert "thalach" in data_mod.select_features(cleaned)["indicators"]
    spec = data_mod.select_features(mangled)
    assert "thalach" in spec["dropped_high_missing"]
    assert "thalach" not in spec["indicators"]
    assert set(spec["indicators"]) <= set(spec["features"])


def test_an_indicator_with_no_role_is_rejected(cleaned, monkeypatch):
    """A typo must fail loudly. Filtering alone would silently produce a
    different design matrix and a quietly different model."""
    monkeypatch.setitem(PARAMS["features"], "missing_indicators",
                        ["trestbps", "trestbp"])
    with pytest.raises(ValueError, match="trestbp"):
        data_mod.select_features(cleaned)


def test_split_is_stratified_and_sized(cleaned):
    X_train, X_test, y_train, y_test, g_train, g_test = data_mod.split(cleaned)

    assert len(X_train) + len(X_test) == len(cleaned)
    assert len(X_test) / len(cleaned) == pytest.approx(
        PARAMS["data"]["test_size"], rel=0.01)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    assert not set(X_train.index) & set(X_test.index)


def test_split_returns_aligned_groups(cleaned):
    X_train, X_test, _, _, g_train, g_test = data_mod.split(cleaned)
    assert list(g_train.index) == list(X_train.index)
    assert list(g_test.index) == list(X_test.index)


def test_split_is_deterministic(cleaned):
    a = data_mod.split(cleaned)[0].index
    b = data_mod.split(cleaned)[0].index
    assert list(a) == list(b)


def test_prepare_subsets_to_selected_features(prepared):
    spec = prepared["spec"]
    assert list(prepared["X_train"].columns) == spec["features"]
    assert list(prepared["X_test"].columns) == spec["features"]
    assert "target" not in prepared["X_train"].columns


def test_prepare_selects_on_train_only(prepared, cleaned):
    """prepare() must feed select_features the training rows, not the whole frame."""
    X_train = prepared["X_train"]
    expected = data_mod.select_features(cleaned.loc[X_train.index])
    assert prepared["spec"]["features"] == expected["features"]
