import pandas as pd
import pytest

from po_delay.validation import assert_no_time_leakage, rolling_origin_splits, time_based_split


def test_time_based_split_partitions_all_rows(small_dataset):
    split = time_based_split(small_dataset)
    total = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    assert total == len(small_dataset)
    all_idx = set(split.train_idx) | set(split.val_idx) | set(split.test_idx)
    assert len(all_idx) == len(small_dataset)  # no overlaps


def test_time_based_split_is_chronological(small_dataset):
    split = time_based_split(small_dataset)
    assert_no_time_leakage(small_dataset, split)  # should not raise

    dates = pd.to_datetime(small_dataset["ack_date"])
    assert dates.iloc[split.train_idx].max() <= dates.iloc[split.val_idx].min()
    assert dates.iloc[split.val_idx].max() <= dates.iloc[split.test_idx].min()


def test_assert_no_time_leakage_catches_shuffled_split(small_dataset):
    split = time_based_split(small_dataset)
    # Deliberately swap a training and test index to simulate a leaky split.
    bad_train = split.train_idx.copy()
    bad_test = split.test_idx.copy()
    bad_train[0], bad_test[0] = bad_test[0], bad_train[0]

    from po_delay.validation import TimeSplit

    bad_split = TimeSplit(train_idx=bad_train, val_idx=split.val_idx, test_idx=bad_test)
    with pytest.raises(ValueError):
        assert_no_time_leakage(small_dataset, bad_split)


def test_invalid_fractions_raise(small_dataset):
    with pytest.raises(ValueError):
        time_based_split(small_dataset, train_frac=0.7, val_frac=0.4)


def test_rolling_origin_splits_are_expanding_and_ordered(small_dataset):
    splits = rolling_origin_splits(small_dataset, n_splits=3, min_train_frac=0.3)
    dates = pd.to_datetime(small_dataset["ack_date"])
    prev_train_len = 0
    for train_idx, val_idx in splits:
        assert len(train_idx) >= prev_train_len
        prev_train_len = len(train_idx)
        assert dates.iloc[train_idx].max() <= dates.iloc[val_idx].min()
