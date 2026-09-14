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
    """Flag order book anomalies without modifying or removing rows."""

    if isinstance(n_levels, bool) or not isinstance(n_levels, int):
        raise ValueError("n_levels must be an integer")

    if n_levels <= 0:
        raise ValueError("n_levels must be strictly positive")

    ask_sentinel = 9999999999
    bid_sentinel = -9999999999

    ask_price_cols = [f"ask_price_{level}" for level in range(1, n_levels + 1)]
    bid_price_cols = [f"bid_price_{level}" for level in range(1, n_levels + 1)]
    size_cols = [
        f"{side}_size_{level}"
        for level in range(1, n_levels + 1)
        for side in ("ask", "bid")
    ]

    required_cols = ask_price_cols + bid_price_cols + size_cols

    if not book.columns.is_unique:
        raise ValueError("book must have unique column names")

    missing_cols = [column for column in required_cols if column not in book.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    ask_prices = book[ask_price_cols]
    bid_prices = book[bid_price_cols]
    sizes = book[size_cols]

    empty_asks = ask_prices.eq(ask_sentinel)
    empty_bids = bid_prices.eq(bid_sentinel)

    invalid_asks = ask_prices.le(0) & ~empty_asks
    invalid_bids = bid_prices.le(0) & ~empty_bids

    best_ask = book["ask_price_1"]
    best_bid = book["bid_price_1"]

    valid_best_ask = best_ask.notna() & best_ask.gt(0) & best_ask.ne(ask_sentinel)
    valid_best_bid = best_bid.notna() & best_bid.gt(0) & best_bid.ne(bid_sentinel)

    checks = pd.DataFrame(index=book.index)

    checks["has_missing"] = book[required_cols].isna().any(axis=1)
    checks["has_negative_size"] = sizes.lt(0).any(axis=1)
    checks["has_empty_level"] = empty_asks.any(axis=1) | empty_bids.any(axis=1)
    checks["has_invalid_price"] = invalid_asks.any(axis=1) | invalid_bids.any(axis=1)
    checks["is_crossed"] = valid_best_ask & valid_best_bid & best_bid.gt(best_ask)

    return checks.fillna(False).astype(bool)
