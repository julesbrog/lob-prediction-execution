import numpy as np
import pandas as pd
import pytest

from lob.features import (
    build_basic_features,
    compute_depth_imbalance,
    compute_event_ofi,
    compute_rolling_features,
    compute_rolling_ofi,
)


@pytest.mark.parametrize(
    "bid,ask,bid_size,ask_size,expected",
    [
        # Bid improves: add the new bid quantity.
        (101, 103, 15, 20, 15),
        # Bid unchanged: current quantity minus previous quantity.
        (100, 103, 15, 20, 5),
        # Bid worsens: subtract the previous bid quantity.
        (99, 103, 15, 20, -10),
        # Ask improves: subtract the new ask quantity.
        (100, 102, 10, 25, -25),
        # Ask unchanged: previous quantity minus current quantity.
        (100, 103, 10, 25, -5),
        # Ask worsens: add the previous ask quantity.
        (100, 104, 10, 25, 20),
    ],
)
def test_event_ofi_contributions(bid, ask, bid_size, ask_size, expected):
    events = pd.DataFrame(
        {
            "bid_price_1": [100, bid],
            "ask_price_1": [103, ask],
            "bid_size_1": [10, bid_size],
            "ask_size_1": [20, ask_size],
        }
    )

    result = compute_event_ofi(events)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(expected)


def test_rolling_ofi_uses_current_and_previous_events():
    event_ofi = pd.Series([np.nan, 2.0, -1.0, 4.0, -2.0])

    result = compute_rolling_ofi(event_ofi, windows=(2,))

    np.testing.assert_allclose(
        result["ofi_2"].to_numpy(),
        [np.nan, np.nan, 1.0, 3.0, 2.0],
        equal_nan=True,
    )


def test_rolling_returns_and_volatility():
    mid = pd.Series([100.0, 101.0, 99.0, 102.0])
    result = compute_rolling_features(mid, windows=(2,))

    expected_return = np.log(99.0 / 100.0)
    expected_volatility = np.sqrt(
        np.log(101.0 / 100.0) ** 2 + np.log(99.0 / 101.0) ** 2
    )

    assert result["log_return_2"].iloc[:2].isna().all()
    assert result["realized_vol_2"].iloc[:2].isna().all()
    assert result.loc[2, "log_return_2"] == pytest.approx(expected_return)
    assert result.loc[2, "realized_vol_2"] == pytest.approx(expected_volatility)


def test_features_do_not_depend_on_future_events():
    n = 12
    step = np.arange(n)

    events = pd.DataFrame(
        {
            "bid_price_1": 1_000_000 + 100 * step,
            "ask_price_1": 1_000_300 + 100 * step,
            "bid_size_1": 10 + step,
            "ask_size_1": 25 - step,
            "bid_price_2": 999_900 + 100 * step,
            "ask_price_2": 1_000_400 + 100 * step,
            "bid_size_2": 20 + step,
            "ask_size_2": 35 - step,
        }
    )

    def build_features(frame):
        features = build_basic_features(frame)
        features["imbalance_2"] = compute_depth_imbalance(frame, n_levels=2)
        features = features.join(
            compute_rolling_features(features["mid_price"], windows=(2, 3))
        )
        return features.join(
            compute_rolling_ofi(compute_event_ofi(frame), windows=(2, 3))
        )

    original = build_features(events)

    # Change only events after position 6.
    cutoff = 6
    changed = events.copy()
    price_cols = [col for col in events if "price" in col]
    size_cols = [col for col in events if "size" in col]

    changed.loc[cutoff + 1 :, price_cols] += 50_000
    changed.loc[cutoff + 1 :, size_cols] *= 3

    modified = build_features(changed)

    pd.testing.assert_frame_equal(
        original.iloc[: cutoff + 1],
        modified.iloc[: cutoff + 1],
    )
