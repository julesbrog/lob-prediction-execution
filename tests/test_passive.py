import numpy as np
import pandas as pd
import pytest

from lob.passive import run_passive_backtest, simulate_passive_round_trip

# Prices in LOBSTER units: bid 100.00, ask 100.10 (10 ticks wide).
BID, ASK = 1_000_000, 1_001_000


def make_events(rows):
    """rows: list of (event_type, price, size, direction, bid, ask, bid_size, ask_size)."""

    frame = pd.DataFrame(
        rows,
        columns=[
            "event_type", "price", "size", "direction",
            "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1",
        ],
    )
    frame.insert(0, "time", np.arange(len(frame), dtype=float))
    return frame


def quiet(n, bid=BID, ask=ASK, bid_size=300, ask_size=300):
    return [(1, bid, 10, 1, bid, ask, bid_size, ask_size)] * n


def test_join_queue_fills_only_after_queue_ahead_is_executed():
    rows = quiet(1, bid_size=50)
    # Seller-initiated executions at the bid: 30 then 25 shares.
    rows += [(4, BID, 30, 1, BID, ASK, 20, 300), (4, BID, 25, 1, BID, ASK, 0, 300)]
    rows += quiet(3)
    events = make_events(rows)

    record = simulate_passive_round_trip(events, 0, side=1, horizon=5, improve_ticks=0)

    assert record["queue_ahead"] == 50
    assert record["filled"]
    assert record["fill_index"] == 2  # 30 < 51, 30 + 25 >= 51
    assert record["entry_price"] == pytest.approx(100.00)
    assert record["exit_price"] == pytest.approx(100.00)  # sell at bid
    assert record["entry_edge"] == pytest.approx(0.05)
    assert record["exit_cost"] == pytest.approx(0.05)
    assert record["net_pnl"] == pytest.approx(0.0)


def test_join_queue_not_filled_when_executions_are_too_small():
    rows = quiet(1, bid_size=50) + [(4, BID, 30, 1, BID, ASK, 20, 300)] + quiet(4)
    events = make_events(rows)

    record = simulate_passive_round_trip(events, 0, side=1, horizon=5, improve_ticks=0)

    assert not record["filled"]
    assert np.isnan(record["net_pnl"])
    assert record["mid_move"] == pytest.approx(0.0)


def test_buyer_initiated_trades_do_not_fill_a_buy_order():
    rows = quiet(1, bid_size=10) + [(4, ASK, 500, -1, BID, ASK, 10, 100)] + quiet(4)
    events = make_events(rows)

    record = simulate_passive_round_trip(events, 0, side=1, horizon=5, improve_ticks=0)

    assert not record["filled"]


def test_improving_one_tick_is_hit_by_the_first_seller_initiated_trade():
    rows = quiet(1, bid_size=10_000) + [(4, BID, 1, 1, BID, ASK, 9_999, 300)] + quiet(4)
    events = make_events(rows)

    joined = simulate_passive_round_trip(events, 0, side=1, horizon=5, improve_ticks=0)
    improved = simulate_passive_round_trip(events, 0, side=1, horizon=5, improve_ticks=1)

    assert not joined["filled"]
    assert improved["filled"]
    assert improved["queue_ahead"] == 0
    assert improved["limit_price"] == pytest.approx(100.01)
    assert improved["entry_edge"] == pytest.approx(0.04)


def test_sell_side_mirror_and_pnl_decomposition():
    new_bid, new_ask = BID + 500, ASK + 500  # mid moves up 5 ticks
    rows = quiet(1, ask_size=5)
    rows += [(4, ASK, 6, -1, BID, ASK, 300, 0)]  # buyer-initiated, clears the queue and us
    rows += [(1, new_bid, 10, 1, new_bid, new_ask, 300, 300)] * 4
    events = make_events(rows)

    record = simulate_passive_round_trip(events, 0, side=-1, horizon=5, improve_ticks=0)

    assert record["filled"]
    assert record["entry_price"] == pytest.approx(100.10)
    assert record["exit_price"] == pytest.approx(100.15)  # buy back at the new ask
    assert record["mid_pnl"] == pytest.approx(-0.05)
    assert record["entry_edge"] == pytest.approx(0.05)
    assert record["exit_cost"] == pytest.approx(0.05)
    assert record["net_pnl"] == pytest.approx(-0.05)


def test_improve_inside_a_one_tick_spread_is_rejected():
    events = make_events(quiet(6, bid=BID, ask=BID + 100))

    with pytest.raises(ValueError):
        simulate_passive_round_trip(events, 0, side=1, horizon=5, improve_ticks=1)


def test_run_passive_backtest_one_order_at_a_time():
    rows = quiet(1, bid_size=1) + [(4, BID, 5, 1, BID, ASK, 300, 300)] + quiet(20)
    events = make_events(rows)
    scores = pd.Series([0.9, 0.9, 0.9, -0.9, 0.0, 0.9], index=[0, 1, 2, 3, 4, 8])

    orders = run_passive_backtest(events, scores, horizon=3, threshold=0.5)

    # Decisions at 0 (busy until 3), 8; 1, 2 and 3 are skipped while busy.
    assert orders["decision_index"].tolist() == [0, 8]
    assert orders["filled"].tolist() == [True, False]
    assert orders["entry_score"].tolist() == [0.9, 0.9]
