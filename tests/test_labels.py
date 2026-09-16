import numpy as np
import pandas as pd

from lob.labels import make_targets


def test_changes_exactly_at_threshold_are_flat():
    # Mid-prices: $100.01 -> $100.02 -> $100.01.
    bid = pd.Series([1_000_000, 1_000_100, 1_000_000])
    ask = bid + 200

    result = make_targets(bid, ask, horizon=1, epsilon_units=200)

    assert result["target"].iloc[:2].tolist() == [0, 0]
    assert pd.isna(result["target"].iloc[-1])

    np.testing.assert_allclose(
        result["future_change"].iloc[:2],
        [0.01, -0.01],
    )


def test_changes_beyond_threshold_are_directional():
    bid = pd.Series([1_000_000, 1_000_200, 1_000_000])
    ask = bid + 200

    result = make_targets(bid, ask, horizon=1, epsilon_units=200)

    assert result["target"].iloc[:2].tolist() == [1, -1]


def test_zero_threshold_and_missing_future():
    bid = pd.Series([1_000_000, 1_000_000, 1_000_100, 999_900])
    ask = bid + 200

    result = make_targets(bid, ask, horizon=1)

    assert result["target"].iloc[:3].tolist() == [0, 1, -1]
    assert result.iloc[-1].isna().all()
    assert result.index.equals(bid.index)
