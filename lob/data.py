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
