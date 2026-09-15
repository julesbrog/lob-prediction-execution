import numpy as np
import pandas as pd


def find_arrival_book_index(
    times: np.ndarray,
    decision_index: int,
    latency_seconds: float,
) -> int | None:
    """Return the book's row position at order arrival.

    Zero latency uses the decision row directly.
    Positive latency includes all events timestamped at arrival.
    Returns None if arrival is beyond the available history.
    """

    if not isinstance(times, np.ndarray) or times.ndim != 1:
        raise ValueError("times must be a one-dimensional NumPy array")

    if times.size == 0:
        raise ValueError("times must not be empty")

    if not np.issubdtype(times.dtype, np.number) or np.iscomplexobj(times):
        raise ValueError("timestamps must be real numbers")

    if not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be finite")

    if np.any(times[1:] < times[:-1]):
        raise ValueError("timestamps must be monotonically increasing")

    if isinstance(decision_index, (bool, np.bool_)) or not isinstance(
        decision_index, (int, np.integer)
    ):
        raise ValueError("decision_index must be an integer")

    if not 0 <= decision_index < len(times):
        raise ValueError("decision_index is outside the available history")

    if isinstance(latency_seconds, (bool, np.bool_)) or not isinstance(
        latency_seconds, (int, float, np.integer, np.floating)
    ):
        raise ValueError("latency_seconds must be a real number")

    if not np.isfinite(latency_seconds) or latency_seconds < 0:
        raise ValueError("latency_seconds must be finite and non-negative")

    if latency_seconds == 0:
        return int(decision_index)

    decision_time = float(times[decision_index])
    latency = float(latency_seconds)

    if latency > float(times[-1]) - decision_time:
        return None

    arrival_time = decision_time + latency

    return int(np.searchsorted(times, arrival_time, side="right") - 1)


def get_aggressive_execution_price(
    events: pd.DataFrame,
    book_index: int,
    side: int,
    quantity: int = 1,
) -> float | None:
    """Return an aggressive execution price in dollars per share.

    Assumes the order book has already been validated.
    Returns None if the best level has insufficient quantity.
    """

    if isinstance(book_index, (bool, np.bool_)) or not isinstance(
        book_index, (int, np.integer)
    ):
        raise ValueError("book_index must be an integer")

    if not 0 <= book_index < len(events):
        raise ValueError("book_index is outside the available history")

    if isinstance(side, (bool, np.bool_)) or not isinstance(side, (int, np.integer)):
        raise ValueError("side must be an integer")

    if side not in (-1, 1):
        raise ValueError("side must be 1 (buy) or -1 (sell)")

    if isinstance(quantity, (bool, np.bool_)) or not isinstance(
        quantity, (int, np.integer)
    ):
        raise ValueError("quantity must be an integer")

    if quantity <= 0:
        raise ValueError("quantity must be strictly positive")

    prefix = "ask" if side == 1 else "bid"
    price_column = f"{prefix}_price_1"
    size_column = f"{prefix}_size_1"

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    for column in (price_column, size_column):
        if column not in events.columns:
            raise ValueError(f"missing column: {column}")

    event_execution = events.iloc[book_index]

    if event_execution[size_column] < quantity:
        return None

    return float(event_execution[price_column] / 10_000)


def compute_execution_times(
    decision_time: float,
    expiry_time: float,
    latency_seconds: float,
    exit_send_deadline: float | None = None,
) -> dict[str, float]:
    """Compute arrival times, optionally enforcing a latest exit submission."""

    for name, value in (
        ("decision_time", decision_time),
        ("expiry_time", expiry_time),
        ("latency_seconds", latency_seconds),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a real number")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    decision_time = float(decision_time)
    expiry_time = float(expiry_time)
    latency_seconds = float(latency_seconds)

    if decision_time < 0 or expiry_time < 0:
        raise ValueError("timestamps must be non-negative")

    if latency_seconds < 0:
        raise ValueError("latency_seconds must be non-negative")

    if expiry_time < decision_time:
        raise ValueError("expiry_time must not precede decision_time")

    if exit_send_deadline is not None:
        if isinstance(exit_send_deadline, (bool, np.bool_)) or not isinstance(
            exit_send_deadline, (int, float, np.integer, np.floating)
        ):
            raise ValueError("exit_send_deadline must be a real number")

        if not np.isfinite(exit_send_deadline):
            raise ValueError("exit_send_deadline must be finite")

        exit_send_deadline = float(exit_send_deadline)

        if exit_send_deadline < decision_time:
            raise ValueError("exit_send_deadline must not precede decision_time")

    entry_arrival_time = decision_time + latency_seconds

    if not np.isfinite(entry_arrival_time):
        raise ValueError("entry arrival time must be finite")

    if exit_send_deadline is None:
        exit_send_time = max(expiry_time, entry_arrival_time)
    else:
        if entry_arrival_time > exit_send_deadline:
            raise ValueError(
                "entry arrives after the latest allowed exit submission time"
            )

        exit_send_time = max(
            entry_arrival_time,
            min(expiry_time, exit_send_deadline),
        )

    exit_arrival_time = exit_send_time + latency_seconds

    if not np.isfinite(exit_arrival_time):
        raise ValueError("exit arrival time must be finite")

    return {
        "entry_arrival_time": entry_arrival_time,
        "exit_send_time": exit_send_time,
        "exit_arrival_time": exit_arrival_time,
    }
