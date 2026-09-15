import numpy as np
import pandas as pd


def compute_mid_price(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Compute mid-prices from one-dimensional bid and ask arrays."""

    if bid.ndim != 1 or ask.ndim != 1:
        raise ValueError("bid and ask must be one-dimensional")

    if bid.shape != ask.shape:
        raise ValueError("bid and ask must have the same length")

    if not np.all(np.isfinite(bid)) or not np.all(np.isfinite(ask)):
        raise ValueError("prices must be finite")

    if np.any(bid <= 0) or np.any(ask <= 0):
        raise ValueError("prices must be strictly positive")

    if np.any(ask < bid):
        raise ValueError("ask must be greater than or equal to bid")

    # Convert before arithmetic to avoid integer overflow.
    bid = bid.astype(float, copy=False)
    ask = ask.astype(float, copy=False)

    return bid + (ask - bid) / 2


def compute_spread(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Compute bid-ask spreads."""

    if bid.ndim != 1 or ask.ndim != 1:
        raise ValueError("bid and ask must be one-dimensional")

    if bid.shape != ask.shape:
        raise ValueError("bid and ask must have the same length")

    if not np.all(np.isfinite(bid)) or not np.all(np.isfinite(ask)):
        raise ValueError("prices must be finite")

    if np.any(bid <= 0) or np.any(ask <= 0):
        raise ValueError("prices must be strictly positive")

    if np.any(ask < bid):
        raise ValueError("ask must be greater than or equal to bid")

    bid = bid.astype(float, copy=False)
    ask = ask.astype(float, copy=False)

    return ask - bid


def compute_imbalance(
    bid_size: np.ndarray,
    ask_size: np.ndarray,
) -> np.ndarray:
    """Compute quantity imbalance; return NaN when both quantities are zero."""

    if bid_size.ndim != 1 or ask_size.ndim != 1:
        raise ValueError("quantities must be one-dimensional")

    if bid_size.shape != ask_size.shape:
        raise ValueError("quantities must have the same length")

    if not np.all(np.isfinite(bid_size)) or not np.all(np.isfinite(ask_size)):
        raise ValueError("quantities must be finite")

    if np.any(bid_size < 0) or np.any(ask_size < 0):
        raise ValueError("quantities must be non-negative")

    bid_size = bid_size.astype(float, copy=False)
    ask_size = ask_size.astype(float, copy=False)

    total_size = bid_size + ask_size
    imbalance = np.full(bid_size.shape, np.nan, dtype=float)

    np.divide(
        bid_size - ask_size,
        total_size,
        out=imbalance,
        where=total_size > 0,
    )

    return imbalance


def compute_weighted_mid(
    bid: np.ndarray,
    ask: np.ndarray,
    bid_size: np.ndarray,
    ask_size: np.ndarray,
) -> np.ndarray:
    """Compute the quantity-weighted price; return NaN for zero total depth."""

    # These functions also validate prices and quantities.
    spread = compute_spread(bid, ask)
    imbalance = compute_imbalance(bid_size, ask_size)

    if bid.shape != bid_size.shape:
        raise ValueError("prices and quantities must have the same length")

    bid = bid.astype(float, copy=False)

    # (1 + imbalance) / 2 equals bid_size / (bid_size + ask_size).
    bid_weight = (1.0 + imbalance) / 2.0

    return bid + spread * bid_weight


def build_basic_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build level-one features from raw LOBSTER prices and quantities."""

    bid = events["bid_price_1"].to_numpy() / 10_000
    ask = events["ask_price_1"].to_numpy() / 10_000
    bid_size = events["bid_size_1"].to_numpy()
    ask_size = events["ask_size_1"].to_numpy()

    mid = compute_mid_price(bid, ask)
    weighted_mid = compute_weighted_mid(bid, ask, bid_size, ask_size)

    return pd.DataFrame(
        {
            "mid_price": mid,
            "spread": compute_spread(bid, ask),
            "imbalance_1": compute_imbalance(bid_size, ask_size),
            "weighted_mid": weighted_mid,
            "weighted_mid_offset": weighted_mid - mid,
        },
        index=events.index,
    )


def compute_depth_imbalance(
    events: pd.DataFrame,
    n_levels: int,
) -> pd.Series:
    """Compute imbalance across the first n_levels of the order book."""

    if isinstance(n_levels, bool) or not isinstance(n_levels, int):
        raise ValueError("n_levels must be an integer")

    if n_levels <= 0:
        raise ValueError("n_levels must be strictly positive")

    bid_cols = [f"bid_size_{level}" for level in range(1, n_levels + 1)]
    ask_cols = [f"ask_size_{level}" for level in range(1, n_levels + 1)]

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    missing_cols = [col for col in bid_cols + ask_cols if col not in events.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    bid_sizes = events[bid_cols].to_numpy(dtype=float)
    ask_sizes = events[ask_cols].to_numpy(dtype=float)

    if not np.all(np.isfinite(bid_sizes)) or not np.all(np.isfinite(ask_sizes)):
        raise ValueError("quantities must be finite")

    if np.any(bid_sizes < 0) or np.any(ask_sizes < 0):
        raise ValueError("quantities must be non-negative")

    total_bid_size = bid_sizes.sum(axis=1)
    total_ask_size = ask_sizes.sum(axis=1)

    imbalance = compute_imbalance(total_bid_size, total_ask_size)

    return pd.Series(
        imbalance,
        index=events.index,
        name=f"imbalance_{n_levels}",
    )


def compute_rolling_features(
    mid_price: pd.Series,
    windows: tuple[int, ...] = (10, 50, 100),
) -> pd.DataFrame:
    """Compute causal historical features for a single trading session."""

    if not isinstance(mid_price, pd.Series):
        raise ValueError("mid_price must be a pandas Series")

    prices = mid_price.to_numpy(dtype=float)

    if not np.all(np.isfinite(prices)):
        raise ValueError("mid-prices must be finite")

    if np.any(prices <= 0):
        raise ValueError("mid-prices must be strictly positive")

    for window in windows:
        if isinstance(window, bool) or not isinstance(window, int):
            raise ValueError("windows must contain integers")

        if window <= 0:
            raise ValueError("windows must be strictly positive")

    if len(set(windows)) != len(windows):
        raise ValueError("windows must not contain duplicates")

    log_prices = pd.Series(np.log(prices), index=mid_price.index)
    one_event_returns = log_prices.diff()

    features = pd.DataFrame(index=mid_price.index)

    for window in windows:
        features[f"log_return_{window}"] = log_prices.diff(window)

        squared_returns_sum = (
            one_event_returns.pow(2).rolling(window=window, min_periods=window).sum()
        )

        features[f"realized_vol_{window}"] = np.sqrt(squared_returns_sum)

    return features
