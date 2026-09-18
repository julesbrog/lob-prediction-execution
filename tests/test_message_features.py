import numpy as np
import pandas as pd
import pytest

from lob.message_features import (
    compute_activity_features,
    compute_depth_features,
    compute_flow_features,
    compute_noise_feature,
    compute_trade_features,
)


def make_events():
    return pd.DataFrame(
        {
            "time": [1.0, 1.0, 1.5, 2.0, 2.5, 4.0],
            # submit buy, execute sell order (buyer initiated), delete buy,
            # execute buy order (seller initiated), partial cancel sell, submit sell
            "event_type": [1, 4, 3, 5, 2, 1],
            "size": [100, 30, 50, 20, 10, 40],
            "direction": [1, -1, 1, 1, -1, -1],
            "bid_size_1": [100, 100, 50, 50, 50, 50],
            "ask_size_1": [80, 50, 50, 50, 40, 80],
            "bid_size_2": [10] * 6,
            "ask_size_2": [20] * 6,
        }
    )


def test_trade_features_sign_and_window():
    features = compute_trade_features(make_events(), windows=(2,))

    # Events 2 and 4 are trades: +30 (buyer initiated), -20 (seller initiated).
    expected_signed = [np.nan, 30.0, 30.0, -20.0, -20.0, 0.0]
    expected_total = [np.nan, 30.0, 30.0, 20.0, 20.0, 0.0]

    np.testing.assert_allclose(
        features["signed_trade_volume_2"], expected_signed, equal_nan=True
    )
    np.testing.assert_allclose(features["trade_volume_2"], expected_total, equal_nan=True)


def test_flow_features_submissions_minus_cancellations():
    features = compute_flow_features(make_events(), windows=(6,))

    # +100 (buy submit) - (+50 buy delete) - (-10 sell cancel) + (-40 sell submit)
    assert features["net_order_flow_6"].iloc[-1] == pytest.approx(100 - 50 + 10 - 40)
    assert features["cancel_volume_6"].iloc[-1] == pytest.approx(60)
    assert features["net_order_flow_6"].iloc[:-1].isna().all()


def test_activity_features_are_causal_and_finite_after_warmup():
    features = compute_activity_features(make_events(), windows=(2,))

    assert np.isnan(features["log_time_gap"].iloc[0])
    assert np.isfinite(features["log_time_gap"].iloc[1])  # zero gap, floored
    assert features["event_rate_2"].iloc[2] == pytest.approx(2 / (0.5 + 1e-6))
    assert features["event_rate_2"].iloc[:2].isna().all()


def test_depth_features_use_log1p_of_sums():
    features = compute_depth_features(make_events(), n_levels=2)

    assert features["log_bid_size_1"].iloc[0] == pytest.approx(np.log1p(100))
    assert features["log_ask_depth_2"].iloc[0] == pytest.approx(np.log1p(80 + 20))


def test_noise_feature_is_reproducible_and_independent():
    index = pd.RangeIndex(1000)
    first = compute_noise_feature(index, random_state=1)
    second = compute_noise_feature(index, random_state=1)
    other = compute_noise_feature(index, random_state=2)

    pd.testing.assert_series_equal(first, second)
    assert not np.allclose(first, other)
    assert abs(first.mean()) < 0.2


@pytest.mark.parametrize(
    "column, value",
    [("event_type", 9), ("direction", 0), ("size", -1)],
)
def test_message_validation(column, value):
    events = make_events()
    events.loc[0, column] = value

    with pytest.raises(ValueError):
        compute_trade_features(events, windows=(2,))


def test_non_monotonic_time_is_rejected():
    events = make_events()
    events.loc[3, "time"] = 0.0

    with pytest.raises(ValueError):
        compute_activity_features(events, windows=(2,))
