import numpy as np
import pandas as pd

from lob.execution import get_aggressive_execution_price, compute_execution_times


def simulate_round_trip(
    events: pd.DataFrame,
    decision_index: int,
    side: int,
    horizon: int = 50,
    quantity: int = 1,
    fee_per_share: float = 0.0,
) -> dict | None:
    """Simulate one zero-latency round trip on a validated trading session.

    Fees are in dollars per share per transaction.
    Returns None if entry liquidity is insufficient.
    Raises RuntimeError if entry succeeds but exit liquidity is insufficient.
    """

    if isinstance(decision_index, (bool, np.bool_)) or not isinstance(
        decision_index, (int, np.integer)
    ):
        raise ValueError("decision_index must be an integer")

    if not 0 <= decision_index < len(events):
        raise ValueError("decision_index is outside the available history")

    if isinstance(horizon, (bool, np.bool_)) or not isinstance(
        horizon, (int, np.integer)
    ):
        raise ValueError("horizon must be an integer")

    if horizon <= 0:
        raise ValueError("horizon must be strictly positive")

    if isinstance(fee_per_share, (bool, np.bool_)) or not isinstance(
        fee_per_share, (int, float, np.integer, np.floating)
    ):
        raise ValueError("fee_per_share must be a real number")

    if not np.isfinite(fee_per_share) or fee_per_share < 0:
        raise ValueError("fee_per_share must be finite and non-negative")

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    if "time" not in events.columns:
        raise ValueError("events must contain a 'time' column")

    decision_index = int(decision_index)
    exit_index = decision_index + int(horizon)

    if exit_index >= len(events):
        raise ValueError("exit_index is outside the available history")

    entry_time = float(events["time"].iloc[decision_index])
    exit_time = float(events["time"].iloc[exit_index])

    if not np.isfinite(entry_time) or not np.isfinite(exit_time):
        raise ValueError("entry and exit timestamps must be finite")

    if exit_time < entry_time:
        raise ValueError("exit timestamp must not precede entry timestamp")

    # This call also validates side and quantity.
    entry_price = get_aggressive_execution_price(
        events=events,
        book_index=decision_index,
        side=side,
        quantity=quantity,
    )

    if entry_price is None:
        return None

    exit_price = get_aggressive_execution_price(
        events=events,
        book_index=exit_index,
        side=-side,
        quantity=quantity,
    )

    if exit_price is None:
        raise RuntimeError(
            "insufficient quantity at exit: the open position cannot be closed"
        )

    side = int(side)
    quantity = int(quantity)

    fees = 2 * quantity * float(fee_per_share)
    gross_pnl = side * quantity * (exit_price - entry_price)
    net_pnl = gross_pnl - fees

    return {
        "entry_index": decision_index,
        "exit_index": exit_index,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": side,
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
    }


def run_backtest(
    events: pd.DataFrame,
    scores: pd.Series,
    horizon: int = 50,
    threshold: float = 0.3,
    quantity: int = 1,
    fee_per_share: float = 0.0,
) -> pd.DataFrame:
    """Run a zero-latency backtest with at most one open position.

    Scores are indexed by original event positions.
    Entry occurs when the score strictly exceeds either threshold.
    Exit occurs horizon events after entry.
    No new entry is allowed on the exit event.
    """

    # Validate scalar parameters even when no trade is triggered.
    for name, value in (("horizon", horizon), ("quantity", quantity)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")

        if value <= 0:
            raise ValueError(f"{name} must be strictly positive")

    for name, value in (
        ("threshold", threshold),
        ("fee_per_share", fee_per_share),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a real number")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    if fee_per_share < 0:
        raise ValueError("fee_per_share must be non-negative")

    horizon = int(horizon)
    quantity = int(quantity)

    # Validate the event sequence.
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError("events must be a non-empty DataFrame")

    if not events.index.equals(pd.RangeIndex(len(events))):
        raise ValueError("events must retain their original positional index")

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    required_cols = [
        "time",
        "bid_price_1",
        "bid_size_1",
        "ask_price_1",
        "ask_size_1",
    ]
    missing_cols = [column for column in required_cols if column not in events.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    times = events["time"].to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(times).all():
        raise ValueError("timestamps must be finite")

    if np.any(times[1:] < times[:-1]):
        raise ValueError("timestamps must be monotonically increasing")

    # Quote validity is assumed to have been checked before this call.

    # Validate scores and their alignment with event positions.
    if not isinstance(scores, pd.Series):
        raise ValueError("scores must be a pandas Series")

    if not scores.index.is_unique:
        raise ValueError("score indices must be unique")

    if not scores.index.is_monotonic_increasing:
        raise ValueError("score indices must be increasing")

    score_values = scores.to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")

    if np.any((score_values < -1) | (score_values > 1)):
        raise ValueError("scores must be between -1 and 1")

    if not scores.empty:
        if not pd.api.types.is_integer_dtype(scores.index.dtype):
            raise ValueError("score indices must be integer event positions")

        if scores.index.hasnans:
            raise ValueError("score indices must not contain missing values")

        if scores.index[0] < 0 or scores.index[-1] >= len(events):
            raise ValueError("score indices are outside the available history")

        if int(scores.index[-1]) + horizon >= len(events):
            raise ValueError("all scored events must have a complete future horizon")

    trade_columns = [
        "entry_index",
        "exit_index",
        "entry_time",
        "exit_time",
        "side",
        "quantity",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "fees",
        "net_pnl",
        "entry_score",
    ]

    trades = []
    last_exit_index = -1

    # Iterate only over events eligible to trigger an entry.
    for decision_index, score in zip(scores.index, score_values):
        decision_index = int(decision_index)

        if decision_index <= last_exit_index:
            continue

        if score > threshold:
            side = 1
        elif score < -threshold:
            side = -1
        else:
            continue

        trade = simulate_round_trip(
            events=events,
            decision_index=decision_index,
            side=side,
            horizon=horizon,
            quantity=quantity,
            fee_per_share=fee_per_share,
        )

        # No entry occurred if liquidity was insufficient.
        if trade is None:
            continue

        trade["entry_score"] = float(score)
        trades.append(trade)
        last_exit_index = trade["exit_index"]

    return pd.DataFrame(trades, columns=trade_columns)


def simulate_round_trip_with_latency(
    events: pd.DataFrame,
    decision_index: int,
    side: int,
    horizon: int = 50,
    quantity: int = 1,
    fee_per_share: float = 0.0,
    latency_seconds: float = 0.0,
    exit_send_deadline: float | None = None,
) -> dict:
    """Simulate one round trip with equal entry and exit latency.

    Assumes validated quotes for one session in original event order.
    Execution acknowledgements are assumed instantaneous at arrival.
    """

    # Validate integer parameters.
    for name, value in (
        ("decision_index", decision_index),
        ("side", side),
        ("horizon", horizon),
        ("quantity", quantity),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")

    if side not in (-1, 1):
        raise ValueError("side must be 1 (buy) or -1 (sell)")

    if horizon <= 0:
        raise ValueError("horizon must be strictly positive")

    if quantity <= 0:
        raise ValueError("quantity must be strictly positive")

    # Validate costs and latency.
    for name, value in (
        ("fee_per_share", fee_per_share),
        ("latency_seconds", latency_seconds),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a real number")

        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError("events must be a non-empty DataFrame")

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    required_cols = [
        "time",
        "bid_price_1",
        "bid_size_1",
        "ask_price_1",
        "ask_size_1",
    ]

    missing_cols = [column for column in required_cols if column not in events.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    decision_index = int(decision_index)
    horizon = int(horizon)
    side = int(side)
    quantity = int(quantity)
    latency_seconds = float(latency_seconds)
    fee_per_share = float(fee_per_share)

    if not 0 <= decision_index < len(events):
        raise ValueError("decision_index is outside the available history")

    expiry_index = decision_index + horizon

    if expiry_index >= len(events):
        raise ValueError("expiry_index is outside the available history")

    times = events["time"].to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("timestamps must be finite and non-negative")

    if np.any(times[1:] < times[:-1]):
        raise ValueError("timestamps must be monotonically increasing")

    # The expiry stays attached to the original decision event.
    decision_time = float(times[decision_index])
    expiry_time = float(times[expiry_index])

    timing = compute_execution_times(
        decision_time=decision_time,
        expiry_time=expiry_time,
        latency_seconds=latency_seconds,
        exit_send_deadline=exit_send_deadline,
    )
    forced_exit = exit_send_deadline is not None and expiry_time > float(
        exit_send_deadline
    )
    entry_arrival_time = timing["entry_arrival_time"]
    exit_arrival_time = timing["exit_arrival_time"]

    if entry_arrival_time > times[-1]:
        raise ValueError("entry arrival is beyond the available history")

    # Preserve the original event ordering at zero latency.
    if latency_seconds == 0:
        entry_book_index = decision_index
    else:
        entry_book_index = int(
            np.searchsorted(times, entry_arrival_time, side="right") - 1
        )

    entry_price = get_aggressive_execution_price(
        events=events,
        book_index=entry_book_index,
        side=side,
        quantity=quantity,
    )

    result = {
        "status": "entry_rejected",
        "decision_index": decision_index,
        "decision_time": decision_time,
        "expiry_index": expiry_index,
        "expiry_time": expiry_time,
        "latency_seconds": latency_seconds,
        "side": side,
        "quantity": quantity,
        "entry_book_index": entry_book_index,
        "exit_book_index": None,
        # These aliases preserve compatibility with PnL decomposition.
        "entry_index": entry_book_index,
        "exit_index": None,
        # No execution timestamps or prices until a fill occurs.
        "entry_time": None,
        "exit_send_time": None,
        "exit_time": None,
        "entry_price": None,
        "exit_price": None,
        "gross_pnl": None,
        "fees": 0.0,
        "net_pnl": None,
        "resolved_time": entry_arrival_time,
        "resolved_book_index": entry_book_index,
        "exit_send_deadline": (
            None if exit_send_deadline is None else float(exit_send_deadline)
        ),
        "forced_exit": False,
    }

    if entry_price is None:
        return result

    # An executed entry must never disappear because exit is impossible.
    if exit_arrival_time > times[-1]:
        raise RuntimeError(
            "entry executed, but exit arrival is beyond available history"
        )

    if latency_seconds == 0 and not forced_exit:
        # Preserve original event ordering for a normal zero-latency exit.
        exit_book_index = expiry_index
    else:
        # Delayed or deadline-triggered exit: use the book at actual arrival.
        exit_book_index = int(
            np.searchsorted(times, exit_arrival_time, side="right") - 1
        )

    exit_price = get_aggressive_execution_price(
        events=events,
        book_index=exit_book_index,
        side=-side,
        quantity=quantity,
    )

    if exit_price is None:
        raise RuntimeError("entry executed, but exit liquidity is insufficient")

    gross_pnl = side * quantity * (exit_price - entry_price)
    fees = 2 * quantity * fee_per_share

    result.update(
        {
            "status": "filled",
            "exit_book_index": exit_book_index,
            "exit_index": exit_book_index,
            "entry_time": entry_arrival_time,
            "exit_send_time": timing["exit_send_time"],
            "exit_time": exit_arrival_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": gross_pnl - fees,
            "resolved_time": exit_arrival_time,
            "resolved_book_index": exit_book_index,
            "forced_exit": bool(forced_exit),
        }
    )

    return result


def run_backtest_with_latency(
    events: pd.DataFrame,
    scores: pd.Series,
    horizon: int = 50,
    threshold: float = 0.3,
    quantity: int = 1,
    fee_per_share: float = 0.0,
    latency_seconds: float = 0.0,
    entry_cutoff_time: float | None = None,
    exit_send_deadline: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a backtest with one pending entry or open position at a time.

    Returns:
        trades: Completed round trips only.
        order_log: All submitted entries, including rejected entries.

    Scores must use original event positions as their index.
    Assumes validated quotes for one trading session.
    """

    # Validate parameters even if no signal triggers an order.
    for name, value in (
        ("horizon", horizon),
        ("quantity", quantity),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")

        if value <= 0:
            raise ValueError(f"{name} must be strictly positive")

    for name, value in (
        ("threshold", threshold),
        ("fee_per_share", fee_per_share),
        ("latency_seconds", latency_seconds),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a real number")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    if fee_per_share < 0:
        raise ValueError("fee_per_share must be non-negative")

    if latency_seconds < 0:
        raise ValueError("latency_seconds must be non-negative")

    horizon = int(horizon)
    quantity = int(quantity)
    threshold = float(threshold)
    fee_per_share = float(fee_per_share)
    latency_seconds = float(latency_seconds)

    # Validate the event sequence.
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError("events must be a non-empty DataFrame")

    if not events.index.equals(pd.RangeIndex(len(events))):
        raise ValueError("events must retain their original positional index")

    if not events.columns.is_unique:
        raise ValueError("events must have unique column names")

    required_cols = [
        "time",
        "bid_price_1",
        "bid_size_1",
        "ask_price_1",
        "ask_size_1",
    ]

    missing_cols = [column for column in required_cols if column not in events.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    times = events["time"].to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("timestamps must be finite and non-negative")

    if np.any(times[1:] < times[:-1]):
        raise ValueError("timestamps must be monotonically increasing")

    # Period boundaries must be supplied together.
    if (entry_cutoff_time is None) != (exit_send_deadline is None):
        raise ValueError(
            "entry_cutoff_time and exit_send_deadline must be supplied together"
        )

    if entry_cutoff_time is not None:
        for name, value in (
            ("entry_cutoff_time", entry_cutoff_time),
            ("exit_send_deadline", exit_send_deadline),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must be a real number")

            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        entry_cutoff_time = float(entry_cutoff_time)
        exit_send_deadline = float(exit_send_deadline)

        if entry_cutoff_time > exit_send_deadline:
            raise ValueError(
                "entry cutoff must not be later than the exit submission deadline"
            )

        if exit_send_deadline + latency_seconds > times[-1]:
            raise ValueError("the history must cover the latest possible exit arrival")
    # Validate scores and alignment.
    if not isinstance(scores, pd.Series):
        raise ValueError("scores must be a pandas Series")

    if not scores.index.is_unique:
        raise ValueError("score indices must be unique")

    if not scores.index.is_monotonic_increasing:
        raise ValueError("score indices must be increasing")

    score_values = scores.to_numpy(dtype=float, na_value=np.nan)

    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")

    if np.any((score_values < -1) | (score_values > 1)):
        raise ValueError("scores must be between -1 and 1")

    if not scores.empty:
        if not pd.api.types.is_integer_dtype(scores.index.dtype):
            raise ValueError("score indices must be integer event positions")

        if scores.index.hasnans:
            raise ValueError("score indices must not contain missing values")

        if scores.index[0] < 0 or scores.index[-1] >= len(events):
            raise ValueError("score indices are outside the available history")

        if int(scores.index[-1]) + horizon >= len(events):
            raise ValueError("all scored events must have a complete future horizon")

    columns = [
        "status",
        "decision_index",
        "decision_time",
        "expiry_index",
        "expiry_time",
        "latency_seconds",
        "side",
        "quantity",
        "entry_book_index",
        "exit_book_index",
        "entry_index",
        "exit_index",
        "entry_time",
        "exit_send_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "fees",
        "net_pnl",
        "resolved_time",
        "resolved_book_index",
        "entry_score",
        "exit_send_deadline",
        "forced_exit",
    ]

    records = []
    resolved_time = -np.inf
    resolved_book_index = -1

    for decision_index, score in zip(scores.index, score_values):
        decision_index = int(decision_index)
        decision_time = float(times[decision_index])
        if entry_cutoff_time is not None and decision_time >= entry_cutoff_time:
            break
        # Ignore signals while an order or position is unresolved.
        if decision_time < resolved_time:
            continue

        if decision_time == resolved_time and decision_index <= resolved_book_index:
            continue

        if score > threshold:
            side = 1
        elif score < -threshold:
            side = -1
        else:
            continue

        # Let execution errors propagate: never silently discard a position.
        result = simulate_round_trip_with_latency(
            events=events,
            decision_index=decision_index,
            side=side,
            horizon=horizon,
            quantity=quantity,
            fee_per_share=fee_per_share,
            latency_seconds=latency_seconds,
            exit_send_deadline=exit_send_deadline,
        )

        result["entry_score"] = float(score)
        records.append(result)

        # Rejected entries also occupy the simulator until their resolution.
        resolved_time = result["resolved_time"]
        resolved_book_index = result["resolved_book_index"]

    order_log = pd.DataFrame(records, columns=columns)

    # Nullable integer columns accommodate rejected entries without exits.
    integer_columns = [
        "decision_index",
        "expiry_index",
        "side",
        "quantity",
        "entry_book_index",
        "exit_book_index",
        "entry_index",
        "exit_index",
        "resolved_book_index",
    ]

    for column in integer_columns:
        order_log[column] = order_log[column].astype("Int64")

    float_columns = [
        "decision_time",
        "expiry_time",
        "latency_seconds",
        "entry_time",
        "exit_send_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "fees",
        "net_pnl",
        "resolved_time",
        "entry_score",
        "exit_send_deadline",
    ]

    for column in float_columns:
        order_log[column] = order_log[column].astype(float)

    order_log["forced_exit"] = order_log["forced_exit"].astype(bool)

    trades = (
        order_log.loc[order_log["status"].eq("filled")].copy().reset_index(drop=True)
    )

    return trades, order_log
