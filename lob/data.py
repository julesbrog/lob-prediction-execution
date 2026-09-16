import numpy as np
import pandas as pd


def load_messages(path: str) -> pd.DataFrame:
    messages = pd.read_csv(path, header=None)
    if messages.shape[1] != 6:
        raise ValueError("must contain exactly 6 columns")
    messages.columns = ["time", "event_type", "order_id", "size", "price", "direction"]
    messages = messages.astype(
        {
            "time": "float",
            "event_type": "int",
            "order_id": "int",
            "size": "int",
            "price": "int",
            "direction": "int",
        }
    )
    if not messages["time"].is_monotonic_increasing:
        raise ValueError("not monotonic increasing")
    return messages


def load_orderbook(path: str, n_levels: int = 10) -> pd.DataFrame:
    if isinstance(n_levels, bool) or not isinstance(n_levels, int):
        raise ValueError("n_levels must be an integer")
    if n_levels <= 0:
        raise ValueError("n_levels must be strictly positive")
    orderbook = pd.read_csv(path, header=None)
    if orderbook.shape[1] != 4 * n_levels:
        raise ValueError("orderbook must contain exactly 4 * n_levels columns")

    columns = []

    for level in range(1, n_levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )

    orderbook.columns = columns
    orderbook = orderbook.astype("int64")

    return orderbook


def align_events(messages: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    if len(messages) != len(book):
        raise ValueError(
            "messages and order_book must contain the same number of events"
        )
    N = len(messages)
    if not messages.index.equals(pd.RangeIndex(N)) or not book.index.equals(
        pd.RangeIndex(N)
    ):
        raise ValueError("messages or book must have index from 0 to N - 1")
    if not set(messages.columns).isdisjoint(book.columns):
        raise ValueError("one column is commun between both")
    df = pd.concat([messages, book], axis=1)
    df.index.name = "event_id"
    return df


def validate_book(book: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    """Flag anomalies without modifying or removing order book rows."""

    if isinstance(n_levels, bool) or not isinstance(n_levels, int):
        raise ValueError("n_levels must be an integer")

    if n_levels <= 0:
        raise ValueError("n_levels must be strictly positive")

    if not book.columns.is_unique:
        raise ValueError("book must have unique column names")

    ask_cols = [f"ask_price_{level}" for level in range(1, n_levels + 1)]
    bid_cols = [f"bid_price_{level}" for level in range(1, n_levels + 1)]
    ask_size_cols = [f"ask_size_{level}" for level in range(1, n_levels + 1)]
    bid_size_cols = [f"bid_size_{level}" for level in range(1, n_levels + 1)]

    required_cols = ask_cols + bid_cols + ask_size_cols + bid_size_cols

    missing_cols = [col for col in required_cols if col not in book.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    for col in required_cols:
        if (
            not pd.api.types.is_numeric_dtype(book[col])
            or pd.api.types.is_bool_dtype(book[col])
            or pd.api.types.is_complex_dtype(book[col])
        ):
            raise ValueError(f"{col} must contain real numeric values")

    ask = book[ask_cols].to_numpy(dtype=float, na_value=np.nan)
    bid = book[bid_cols].to_numpy(dtype=float, na_value=np.nan)
    ask_size = book[ask_size_cols].to_numpy(dtype=float, na_value=np.nan)
    bid_size = book[bid_size_cols].to_numpy(dtype=float, na_value=np.nan)

    prices = np.concatenate([ask, bid], axis=1)
    sizes = np.concatenate([ask_size, bid_size], axis=1)
    values = np.concatenate([prices, sizes], axis=1)

    ask_sentinel = 9_999_999_999
    bid_sentinel = -9_999_999_999

    empty_ask = ask == ask_sentinel
    empty_bid = bid == bid_sentinel

    valid_ask = (
        np.isfinite(ask) & (ask > 0) & (ask != ask_sentinel) & (ask != bid_sentinel)
    )
    valid_bid = (
        np.isfinite(bid) & (bid > 0) & (bid != ask_sentinel) & (bid != bid_sentinel)
    )

    # Missing and infinite prices have their own flags.
    invalid_ask = np.isfinite(ask) & ~empty_ask & ~valid_ask
    invalid_bid = np.isfinite(bid) & ~empty_bid & ~valid_bid

    # Compare adjacent occupied, valid price levels.
    ask_pairs = valid_ask[:, :-1] & valid_ask[:, 1:]
    bid_pairs = valid_bid[:, :-1] & valid_bid[:, 1:]

    unordered_ask = (ask_pairs & (ask[:, 1:] <= ask[:, :-1])).any(axis=1)
    unordered_bid = (bid_pairs & (bid[:, 1:] >= bid[:, :-1])).any(axis=1)

    # Once an empty level appears, all deeper levels must be empty.
    ask_gap = (np.maximum.accumulate(empty_ask, axis=1) & ~empty_ask).any(axis=1)
    bid_gap = (np.maximum.accumulate(empty_bid, axis=1) & ~empty_bid).any(axis=1)

    # Sentinel prices must have zero volume.
    bad_empty_size = (empty_ask & (ask_size != 0)).any(axis=1) | (
        empty_bid & (bid_size != 0)
    ).any(axis=1)

    valid_best = valid_ask[:, 0] & valid_bid[:, 0]

    return pd.DataFrame(
        {
            "has_missing": np.isnan(values).any(axis=1),
            "has_infinite": np.isinf(values).any(axis=1),
            "has_negative_size": (sizes < 0).any(axis=1),
            "has_empty_level": (empty_ask.any(axis=1) | empty_bid.any(axis=1)),
            "has_invalid_price": (invalid_ask.any(axis=1) | invalid_bid.any(axis=1)),
            "is_crossed": valid_best & (bid[:, 0] > ask[:, 0]),
            "has_unordered_ask": unordered_ask,
            "has_unordered_bid": unordered_bid,
            "has_invalid_empty_layout": ask_gap | bid_gap,
            "has_invalid_empty_size": bad_empty_size,
        },
        index=book.index,
        dtype=bool,
    )


def compute_horizon_duration(
    events: pd.DataFrame,
    horizon: int = 50,
) -> pd.Series:
    """Compute the duration in seconds of an event horizon within one session."""

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("horizon must be an integer")

    if horizon <= 0:
        raise ValueError("horizon must be strictly positive")

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    if "time" not in events.columns:
        raise ValueError("events must contain a 'time' column")

    time = events["time"]

    if not np.all(np.isfinite(time.to_numpy(dtype=float, na_value=np.nan))):
        raise ValueError("timestamps must be finite")

    if not time.is_monotonic_increasing:
        raise ValueError("timestamps must be monotonically increasing")

    duration = time.shift(-horizon) - time

    return duration.rename("horizon_seconds")
