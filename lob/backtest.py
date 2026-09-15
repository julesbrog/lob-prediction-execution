import numpy as np
import pandas as pd

from lob.execution import get_aggressive_execution_price


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
