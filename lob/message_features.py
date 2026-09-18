"""Features from the LOBSTER message file, plus absolute depth.

Event types: 1 submission, 2 partial cancel, 3 deletion, 4 visible
execution, 5 hidden execution, 7 halt. direction is the side of the limit
order (1 buy, -1 sell), so an execution with direction -1 was initiated by a
buyer and counts as positive signed volume.
"""

import numpy as np
import pandas as pd

MESSAGE_COLUMNS = ["time", "event_type", "size", "direction"]


def _check_messages(events: pd.DataFrame) -> None:
    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    missing = [col for col in MESSAGE_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    if not events["time"].is_monotonic_increasing:
        raise ValueError("timestamps must be monotonically increasing")

    if not events["event_type"].isin([1, 2, 3, 4, 5, 7]).all():
        raise ValueError("event_type must be a LOBSTER event code")

    if not events["direction"].isin([-1, 1]).all():
        raise ValueError("direction must be -1 or 1")

    if (events["size"] < 0).any():
        raise ValueError("size must be non-negative")


def _check_windows(windows: tuple[int, ...]) -> None:
    for window in windows:
        if isinstance(window, bool) or not isinstance(window, int):
            raise ValueError("windows must contain integers")

        if window <= 0:
            raise ValueError("windows must be strictly positive")

    if len(set(windows)) != len(windows):
        raise ValueError("windows must not contain duplicates")


def compute_trade_features(
    events: pd.DataFrame,
    windows: tuple[int, ...] = (10, 50, 100),
) -> pd.DataFrame:
    """Signed and total executed volume over trailing windows (visible + hidden)."""

    _check_messages(events)
    _check_windows(windows)

    is_trade = events["event_type"].isin([4, 5])
    size = events["size"].astype(float).where(is_trade, 0.0)
    signed = -events["direction"].astype(float) * size

    features = pd.DataFrame(index=events.index)

    for window in windows:
        rolling = dict(window=window, min_periods=window)
        features[f"signed_trade_volume_{window}"] = signed.rolling(**rolling).sum()
        features[f"trade_volume_{window}"] = size.rolling(**rolling).sum()

    return features


def compute_flow_features(
    events: pd.DataFrame,
    windows: tuple[int, ...] = (10, 50, 100),
) -> pd.DataFrame:
    """Signed submissions minus signed cancellations, and cancelled volume.

    Executions are handled in compute_trade_features.
    """

    _check_messages(events)
    _check_windows(windows)

    size = events["size"].astype(float)
    direction = events["direction"].astype(float)

    submitted = (direction * size).where(events["event_type"].eq(1), 0.0)
    cancelled = (direction * size).where(events["event_type"].isin([2, 3]), 0.0)
    net_flow = submitted - cancelled

    cancel_volume = size.where(events["event_type"].isin([2, 3]), 0.0)

    features = pd.DataFrame(index=events.index)

    for window in windows:
        rolling = dict(window=window, min_periods=window)
        features[f"net_order_flow_{window}"] = net_flow.rolling(**rolling).sum()
        features[f"cancel_volume_{window}"] = cancel_volume.rolling(**rolling).sum()

    return features


def compute_activity_features(
    events: pd.DataFrame,
    windows: tuple[int, ...] = (50, 100),
) -> pd.DataFrame:
    """Log time since the previous event and trailing event rates."""

    _check_messages(events)
    _check_windows(windows)

    time = events["time"].astype(float)

    # Many events share a timestamp; a small floor keeps the log finite.
    gap = time.diff()
    features = pd.DataFrame(index=events.index)
    features["log_time_gap"] = np.log(gap + 1e-6)

    for window in windows:
        elapsed = time - time.shift(window)
        features[f"event_rate_{window}"] = window / (elapsed + 1e-6)

    return features


def compute_depth_features(
    events: pd.DataFrame,
    n_levels: int = 5,
) -> pd.DataFrame:
    """Log absolute quantities at the touch and over the first n_levels."""

    if isinstance(n_levels, bool) or not isinstance(n_levels, int):
        raise ValueError("n_levels must be an integer")

    if n_levels <= 0:
        raise ValueError("n_levels must be strictly positive")

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    bid_cols = [f"bid_size_{level}" for level in range(1, n_levels + 1)]
    ask_cols = [f"ask_size_{level}" for level in range(1, n_levels + 1)]

    missing = [col for col in bid_cols + ask_cols if col not in events.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    bid = events[bid_cols].to_numpy(dtype=float)
    ask = events[ask_cols].to_numpy(dtype=float)

    if np.any(bid < 0) or np.any(ask < 0):
        raise ValueError("quantities must be non-negative")

    return pd.DataFrame(
        {
            "log_bid_size_1": np.log1p(bid[:, 0]),
            "log_ask_size_1": np.log1p(ask[:, 0]),
            f"log_bid_depth_{n_levels}": np.log1p(bid.sum(axis=1)),
            f"log_ask_depth_{n_levels}": np.log1p(ask.sum(axis=1)),
        },
        index=events.index,
    )


def compute_noise_feature(
    index: pd.Index,
    random_state: int = 0,
) -> pd.Series:
    """N(0,1) noise column, used as a control in the ablations."""

    rng = np.random.default_rng(random_state)

    return pd.Series(rng.standard_normal(len(index)), index=index, name="noise")
