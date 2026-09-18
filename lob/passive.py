"""Passive (limit order) execution on a replayed LOBSTER session.

Assumptions, all deliberate and all optimistic or conservative in a known
direction:

- Zero latency for placement and cancellation.
- The order rests at the best quote (improve_ticks = 0, joining the back of
  the visible queue) or improve_ticks inside the spread (an empty queue).
- Only visible executions (event type 4) on our side and at our price
  consume the queue ahead of us. Cancellations are assumed to sit behind us,
  which is the conservative choice.
- A visible execution on our side at a price worse than ours would have hit
  our order first, so it fills us. This is the standard replay assumption
  and ignores the fact that our order might have changed other traders'
  behaviour.
- Fills are all-or-nothing for the requested quantity, which is one share in
  every experiment here.
- The order is cancelled `horizon` events after the decision if unfilled. A
  filled position is closed aggressively at the best opposite quote at that
  same event, so the exit pays half a spread and the entry earns the
  distance between the decision mid-price and the limit price.

Prices in `events` are LOBSTER integers (dollars times 10,000); outputs are
in dollars.
"""

import numpy as np
import pandas as pd

TICK_UNITS = 100  # $0.01 in LOBSTER price units.


def simulate_passive_round_trip(
    events: pd.DataFrame,
    decision_index: int,
    side: int,
    horizon: int = 50,
    improve_ticks: int = 0,
    quantity: int = 1,
    fee_per_share: float = 0.0,
) -> dict:
    """Place one limit order at decision_index and replay the next horizon events.

    Returns a record whether or not the order is filled. Unfilled orders keep
    the mid-price move over the horizon so the fill selection can be studied.
    """

    for name, value in (
        ("decision_index", decision_index),
        ("horizon", horizon),
        ("improve_ticks", improve_ticks),
        ("quantity", quantity),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")

    if horizon <= 0 or quantity <= 0:
        raise ValueError("horizon and quantity must be strictly positive")

    if improve_ticks < 0:
        raise ValueError("improve_ticks must be non-negative")

    if side not in (-1, 1):
        raise ValueError("side must be 1 (buy) or -1 (sell)")

    if not np.isfinite(fee_per_share) or fee_per_share < 0:
        raise ValueError("fee_per_share must be finite and non-negative")

    decision_index = int(decision_index)
    exit_index = decision_index + int(horizon)

    if decision_index < 0 or exit_index >= len(events):
        raise ValueError("decision or exit index is outside the available history")

    bid = events["bid_price_1"].to_numpy()
    ask = events["ask_price_1"].to_numpy()

    bid0 = int(bid[decision_index])
    ask0 = int(ask[decision_index])
    spread0 = ask0 - bid0

    if improve_ticks * TICK_UNITS >= spread0:
        raise ValueError("improve_ticks must keep the order inside the spread")

    if side == 1:
        limit_price = bid0 + improve_ticks * TICK_UNITS
        queue_ahead = int(events["bid_size_1"].iloc[decision_index]) if improve_ticks == 0 else 0
    else:
        limit_price = ask0 - improve_ticks * TICK_UNITS
        queue_ahead = int(events["ask_size_1"].iloc[decision_index]) if improve_ticks == 0 else 0

    # Replay the events strictly after the decision, up to the cancel event.
    window = events.iloc[decision_index + 1 : exit_index + 1]
    is_execution = window["event_type"].to_numpy() == 4
    same_side = window["direction"].to_numpy() == side
    prices = window["price"].to_numpy()
    sizes = window["size"].to_numpy()

    # A trade on our side at a worse price than ours would have hit us first.
    if side == 1:
        worse = prices < limit_price
    else:
        worse = prices > limit_price
    at_price = prices == limit_price

    relevant = is_execution & same_side
    fill_offset = None

    # Executed volume at our price must cover the queue ahead plus our order.
    remaining = queue_ahead + int(quantity)

    for offset in np.flatnonzero(relevant):
        if worse[offset]:
            fill_offset = int(offset)
            break

        if at_price[offset]:
            remaining -= int(sizes[offset])
            if remaining <= 0:
                fill_offset = int(offset)
                break

    mid_decision = (bid0 + ask0) / 20_000
    mid_exit = (int(bid[exit_index]) + int(ask[exit_index])) / 20_000
    spread_exit = (int(ask[exit_index]) - int(bid[exit_index])) / 10_000

    record = {
        "decision_index": decision_index,
        "decision_time": float(events["time"].iloc[decision_index]),
        "exit_index": exit_index,
        "exit_time": float(events["time"].iloc[exit_index]),
        "side": int(side),
        "quantity": int(quantity),
        "improve_ticks": int(improve_ticks),
        "limit_price": limit_price / 10_000,
        "queue_ahead": queue_ahead,
        "spread_decision": spread0 / 10_000,
        "mid_decision": mid_decision,
        "mid_exit": mid_exit,
        "spread_exit": spread_exit,
        "mid_move": side * (mid_exit - mid_decision),
        "filled": fill_offset is not None,
        "fill_index": np.nan,
        "fill_time": np.nan,
        "entry_price": np.nan,
        "exit_price": np.nan,
        "mid_pnl": np.nan,
        "entry_edge": np.nan,
        "exit_cost": np.nan,
        "fees": np.nan,
        "net_pnl": np.nan,
    }

    if fill_offset is None:
        return record

    fill_index = decision_index + 1 + fill_offset
    entry_price = limit_price / 10_000
    exit_price = (int(bid[exit_index]) if side == 1 else int(ask[exit_index])) / 10_000

    fees = 2 * quantity * float(fee_per_share)
    mid_pnl = side * quantity * (mid_exit - mid_decision)
    entry_edge = side * quantity * (mid_decision - entry_price)
    exit_cost = quantity * spread_exit / 2
    net_pnl = side * quantity * (exit_price - entry_price) - fees

    np.testing.assert_allclose(
        net_pnl, mid_pnl + entry_edge - exit_cost - fees, rtol=1e-9, atol=1e-9
    )

    record.update(
        {
            "fill_index": fill_index,
            "fill_time": float(events["time"].iloc[fill_index]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "mid_pnl": mid_pnl,
            "entry_edge": entry_edge,
            "exit_cost": exit_cost,
            "fees": fees,
            "net_pnl": net_pnl,
        }
    )

    return record


def run_passive_backtest(
    events: pd.DataFrame,
    scores: pd.Series,
    horizon: int = 50,
    threshold: float = 0.3,
    improve_ticks: int = 0,
    quantity: int = 1,
    fee_per_share: float = 0.0,
) -> pd.DataFrame:
    """Run the passive strategy with at most one resting order or open position.

    Entry rule and event bookkeeping match `run_backtest`: an order is placed
    when the score strictly exceeds the threshold, and the next decision is
    allowed only after the cancel-or-exit event.
    """

    if not events.index.equals(pd.RangeIndex(len(events))):
        raise ValueError("events must retain their original positional index")

    required = [
        "time", "event_type", "price", "size", "direction",
        "bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1",
    ]
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")

    if not isinstance(scores, pd.Series) or not scores.index.is_monotonic_increasing:
        raise ValueError("scores must be a Series indexed by increasing event positions")

    if not scores.empty and int(scores.index[-1]) + horizon >= len(events):
        raise ValueError("all scored events must have a complete future horizon")

    records = []
    last_exit_index = -1

    for decision_index, score in zip(scores.index, scores.to_numpy(dtype=float)):
        decision_index = int(decision_index)

        if decision_index <= last_exit_index:
            continue

        if score > threshold:
            side = 1
        elif score < -threshold:
            side = -1
        else:
            continue

        spread_units = int(events["ask_price_1"].iloc[decision_index]) - int(
            events["bid_price_1"].iloc[decision_index]
        )
        if improve_ticks * TICK_UNITS >= spread_units:
            # Cannot improve inside a spread this narrow; skip the signal.
            continue

        record = simulate_passive_round_trip(
            events=events,
            decision_index=decision_index,
            side=side,
            horizon=horizon,
            improve_ticks=improve_ticks,
            quantity=quantity,
            fee_per_share=fee_per_share,
        )
        record["entry_score"] = float(score)
        records.append(record)
        last_exit_index = record["exit_index"]

    return pd.DataFrame(records)
