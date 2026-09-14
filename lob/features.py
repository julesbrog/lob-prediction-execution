import numpy as np


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
