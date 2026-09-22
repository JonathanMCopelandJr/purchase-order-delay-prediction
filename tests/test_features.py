from po_delay.features import CATEGORICAL_FEATURES, add_derived_date_features, build_features


def test_categoricals_are_category_dtype(small_dataset):
    X, _ = build_features(small_dataset)
    for col in CATEGORICAL_FEATURES:
        assert str(X[col].dtype) == "category"


def test_derived_date_features_present(small_dataset):
    enriched = add_derived_date_features(small_dataset)
    for col in ["ack_month", "ack_day_of_week", "is_quarter_end", "is_holiday_period"]:
        assert col in enriched.columns
    assert enriched["ack_month"].between(1, 12).all()
    assert enriched["ack_day_of_week"].between(0, 6).all()
    assert set(enriched["is_quarter_end"].unique()) <= {0, 1}
    assert set(enriched["is_holiday_period"].unique()) <= {0, 1}


def test_is_new_supplier_flag_matches_trailing_count(small_dataset):
    enriched = add_derived_date_features(small_dataset)
    zero_history = enriched["supplier_n_orders_trailing365"].fillna(0).eq(0)
    assert (enriched["is_new_supplier"] == zero_history.astype(int)).all()


def test_no_nans_in_final_feature_matrix(small_dataset):
    X, _ = build_features(small_dataset)
    assert not X.isna().any().any()
