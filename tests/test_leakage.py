import pytest

from po_delay import config
from po_delay.features import FEATURE_COLUMNS, assert_no_leakage, build_features


def test_no_leakage_columns_in_feature_list():
    leaked = set(FEATURE_COLUMNS) & set(config.LEAKAGE_COLUMNS)
    assert not leaked


def test_assert_no_leakage_raises_on_injected_leak():
    with pytest.raises(ValueError):
        assert_no_leakage(FEATURE_COLUMNS + ["is_late"])
    with pytest.raises(ValueError):
        assert_no_leakage(FEATURE_COLUMNS + ["actual_delivery_date"])
    with pytest.raises(ValueError):
        assert_no_leakage(FEATURE_COLUMNS + ["delay_days"])


def test_build_features_excludes_leakage_and_id_columns(small_dataset):
    X, y = build_features(small_dataset)
    for col in config.LEAKAGE_COLUMNS:
        assert col not in X.columns
    for col in config.ID_COLUMNS:
        assert col not in X.columns
    assert y.name == config.TARGET_COLUMN
    assert set(y.unique()) <= {0, 1}


def test_build_features_row_count_matches_input(small_dataset):
    X, y = build_features(small_dataset)
    assert len(X) == len(small_dataset)
    assert len(y) == len(small_dataset)
