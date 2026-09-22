import pandas as pd

from po_delay.data_generation import GeneratorParams, generate


def test_generate_is_deterministic():
    df1 = generate(GeneratorParams(n_orders=500, n_suppliers=20, seed=1))
    df2 = generate(GeneratorParams(n_orders=500, n_suppliers=20, seed=1))
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds_differ():
    df1 = generate(GeneratorParams(n_orders=500, n_suppliers=20, seed=1))
    df2 = generate(GeneratorParams(n_orders=500, n_suppliers=20, seed=2))
    assert not df1["is_late"].equals(df2["is_late"])


def test_generate_shape_and_columns(small_dataset):
    assert len(small_dataset) == 1500
    expected_cols = {
        "po_line_id",
        "supplier_id",
        "category",
        "order_date",
        "ack_date",
        "promised_date",
        "actual_delivery_date",
        "delay_days",
        "is_late",
    }
    assert expected_cols.issubset(set(small_dataset.columns))


def test_late_rate_is_reasonable(small_dataset):
    late_rate = small_dataset["is_late"].mean()
    # Not degenerate (all-late or all-on-time) - a real imbalanced-but-learnable rate.
    assert 0.03 < late_rate < 0.5


def test_promised_date_after_ack_date(small_dataset):
    ack = pd.to_datetime(small_dataset["ack_date"])
    promised = pd.to_datetime(small_dataset["promised_date"])
    assert (promised > ack).all()


def test_delay_days_consistent_with_dates(small_dataset):
    promised = pd.to_datetime(small_dataset["promised_date"])
    actual = pd.to_datetime(small_dataset["actual_delivery_date"])
    computed_delay = (actual - promised).dt.days
    pd.testing.assert_series_equal(computed_delay, small_dataset["delay_days"], check_names=False)


def test_is_late_matches_delay_days(small_dataset):
    assert ((small_dataset["delay_days"] > 0) == (small_dataset["is_late"] == 1)).all()


def test_new_suppliers_have_nan_trailing_stats(small_dataset):
    # At least some rows (the first order for a given supplier) must have no
    # trailing history yet.
    assert small_dataset["supplier_n_orders_trailing365"].eq(0).any()
    zero_history = small_dataset["supplier_n_orders_trailing365"].eq(0)
    assert small_dataset.loc[zero_history, "supplier_on_time_rate_trailing365"].isna().all()
