import numpy as np
import pandas as pd
import pytest

from lob.backtest import (
    run_backtest,
    run_backtest_with_latency,
    simulate_round_trip,
    simulate_round_trip_with_latency,
)
from lob.execution import compute_execution_times
from lob.evaluation import decompose_trade_pnl


@pytest.fixture
def book():
    return pd.DataFrame(
        {
            "time": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            "bid_price_1": [
                1000000,
                1000100,
                1000200,
                1000300,
                1000400,
                1000500,
                1000600,
                1000700,
            ],
            "ask_price_1": [
                1000200,
                1000300,
                1000400,
                1000500,
                1000600,
                1000700,
                1000800,
                1000900,
            ],
            "bid_size_1": [10] * 8,
            "ask_size_1": [10] * 8,
        }
    )


@pytest.mark.parametrize(
    "decision, expiry, latency, expected",
    [
        (0.0, 1.0, 0.0, [0.0, 1.0, 1.0]),
        (0.0, 1.0, 0.25, [0.25, 1.0, 1.25]),
        (0.0, 0.25, 0.5, [0.5, 0.5, 1.0]),
    ],
)
def test_execution_times(decision, expiry, latency, expected):
    timing = compute_execution_times(decision, expiry, latency)

    assert [
        timing["entry_arrival_time"],
        timing["exit_send_time"],
        timing["exit_arrival_time"],
    ] == pytest.approx(expected)


@pytest.mark.parametrize("side", [1, -1])
def test_zero_latency_matches_original(book, side):
    parameters = {
        "decision_index": 0,
        "side": side,
        "horizon": 2,
        "quantity": 2,
        "fee_per_share": 0.001,
    }

    original = simulate_round_trip(book, **parameters)
    delayed = simulate_round_trip_with_latency(
        book,
        **parameters,
        latency_seconds=0.0,
    )

    assert delayed["status"] == "filled"

    for key in original:
        assert delayed[key] == pytest.approx(original[key])


def test_latency_changes_both_execution_quotes(book):
    trade = simulate_round_trip_with_latency(
        book,
        decision_index=0,
        side=1,
        horizon=2,
        latency_seconds=0.25,
    )

    # Entry arrives at 0.25; expiry is 0.5; exit arrives at 0.75.
    assert trade["entry_book_index"] == 1
    assert trade["exit_book_index"] == 3
    assert trade["entry_time"] == pytest.approx(0.25)
    assert trade["exit_send_time"] == pytest.approx(0.5)
    assert trade["exit_time"] == pytest.approx(0.75)

    assert trade["entry_price"] == pytest.approx(100.03)
    assert trade["exit_price"] == pytest.approx(100.03)
    assert trade["net_pnl"] == pytest.approx(0.0)
    assert trade["resolved_time"] == pytest.approx(0.75)


def test_late_entry_triggers_exit_on_fill(book):
    trade = simulate_round_trip_with_latency(
        book,
        decision_index=0,
        side=1,
        horizon=1,
        latency_seconds=0.5,
    )

    # Expiry is 0.25, but entry arrives at 0.5.
    # Exit is sent at entry arrival and reaches the market at 1.0.
    assert trade["status"] == "filled"
    assert trade["entry_time"] == pytest.approx(0.5)
    assert trade["exit_send_time"] == pytest.approx(0.5)
    assert trade["exit_time"] == pytest.approx(1.0)

    assert trade["entry_book_index"] == 2
    assert trade["exit_book_index"] == 4


def test_execution_time_differs_from_quote_timestamp(book):
    trade = simulate_round_trip_with_latency(
        book,
        decision_index=0,
        side=1,
        horizon=2,
        latency_seconds=0.125,
    )

    # Arrival occurs between events.
    assert trade["entry_book_index"] == 0
    assert trade["entry_time"] == pytest.approx(0.125)

    assert trade["exit_book_index"] == 2
    assert trade["exit_time"] == pytest.approx(0.625)


def test_entry_and_exit_can_use_the_same_book(book):
    sparse_book = book.iloc[:4].copy()
    sparse_book["time"] = [0.0, 0.125, 1.0, 2.0]

    trade = simulate_round_trip_with_latency(
        sparse_book,
        decision_index=0,
        side=1,
        horizon=1,
        latency_seconds=0.25,
    )

    assert trade["entry_time"] == pytest.approx(0.25)
    assert trade["exit_time"] == pytest.approx(0.5)
    assert trade["entry_book_index"] == 1
    assert trade["exit_book_index"] == 1

    # No quote change: the round trip loses one spread.
    assert trade["net_pnl"] == pytest.approx(-0.02)


def test_rejected_entry_retains_resolution_time(book):
    book.loc[1, "ask_size_1"] = 0

    result = simulate_round_trip_with_latency(
        book,
        decision_index=0,
        side=1,
        horizon=2,
        latency_seconds=0.25,
    )

    assert result["status"] == "entry_rejected"
    assert result["resolved_time"] == pytest.approx(0.25)
    assert result["resolved_book_index"] == 1
    assert result["entry_price"] is None
    assert result["exit_price"] is None
    assert result["net_pnl"] is None


def test_failed_exit_raises(book):
    book.loc[3, "bid_size_1"] = 0

    with pytest.raises(RuntimeError):
        simulate_round_trip_with_latency(
            book,
            decision_index=0,
            side=1,
            horizon=2,
            latency_seconds=0.25,
        )


def test_exit_beyond_history_raises_after_entry(book):
    # Entry arrives at 1.75, within the history.
    # Exit arrives at 2.25, beyond the history.
    with pytest.raises(RuntimeError):
        simulate_round_trip_with_latency(
            book,
            decision_index=5,
            side=1,
            horizon=1,
            latency_seconds=0.5,
        )


def test_entry_beyond_history_raises(book):
    with pytest.raises(ValueError):
        simulate_round_trip_with_latency(
            book,
            decision_index=6,
            side=1,
            horizon=1,
            latency_seconds=0.75,
        )


@pytest.fixture
def loop_book():
    return pd.DataFrame(
        {
            "time": [
                0.0,
                0.125,
                0.25,
                0.375,
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
                1.125,
                1.25,
                1.5,
            ],
            "bid_price_1": [1000000] * 12,
            "ask_price_1": [1000200] * 12,
            "bid_size_1": [10] * 12,
            "ask_size_1": [10] * 12,
        }
    )


def test_latency_loop_matches_original_at_zero(loop_book):
    scores = pd.Series(
        [0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8],
        index=range(9),
    )

    original = run_backtest(
        loop_book,
        scores,
        horizon=2,
        fee_per_share=0.001,
    )

    trades, order_log = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=2,
        fee_per_share=0.001,
        latency_seconds=0.0,
    )

    pd.testing.assert_frame_equal(
        original,
        trades[original.columns],
        check_dtype=False,
    )

    assert trades["entry_index"].tolist() == [0, 3, 6]
    assert order_log["status"].eq("filled").all()


def test_loop_blocks_until_actual_resolution_time(loop_book):
    scores = pd.Series([0.8] * 5, index=[0, 1, 2, 3, 4])

    trades, order_log = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=2,
        latency_seconds=0.1875,
    )

    # First exit occurs at 0.4375, using the book from row 3 (0.375).
    # Row 3 is still too early for a new decision; row 4 (0.5) is allowed.
    assert trades["decision_index"].tolist() == [0, 4]
    assert trades.iloc[0]["exit_book_index"] == 3
    assert trades.iloc[0]["resolved_time"] == pytest.approx(0.4375)
    assert len(order_log) == 2


def test_rejected_entry_blocks_signals_until_arrival(loop_book):
    # The first entry arrives at row 2, where buying is impossible.
    loop_book.loc[2, "ask_size_1"] = 0
    scores = pd.Series([0.8] * 4, index=[0, 1, 2, 3])

    trades, order_log = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=1,
        latency_seconds=0.25,
    )

    # Row 1 is before rejection; row 2 is processed before the rejection.
    # The next eligible decision is row 3.
    assert order_log["decision_index"].tolist() == [0, 3]
    assert order_log["status"].tolist() == ["entry_rejected", "filled"]
    assert order_log.iloc[0]["resolved_time"] == pytest.approx(0.25)

    assert trades["decision_index"].tolist() == [3]


def test_zero_latency_preserves_order_with_equal_timestamps(loop_book):
    loop_book.loc[0:3, "time"] = [0.0, 0.0, 0.25, 0.25]
    scores = pd.Series([0.8] * 4, index=[0, 1, 2, 3])

    trades, _ = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=1,
        latency_seconds=0.0,
    )

    assert trades["decision_index"].tolist() == [0, 2]
    assert trades["exit_index"].tolist() == [1, 3]


def test_positive_latency_blocks_all_events_at_resolution_timestamp(loop_book):
    loop_book["time"] = [
        0.0,
        0.0,
        0.25,
        0.25,
        0.5,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    ]
    scores = pd.Series([0.8] * 7, index=range(7))

    trades, _ = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=1,
        latency_seconds=0.25,
    )

    # First exit arrives at 0.5, after both events with that timestamp.
    assert trades.iloc[0]["resolved_book_index"] == 5
    assert trades["decision_index"].tolist() == [0, 6]


def test_latency_loop_returns_empty_tables_without_signals(loop_book):
    scores = pd.Series([0.0, 0.3, -0.3], index=[0, 1, 2])

    trades, order_log = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=2,
        threshold=0.3,
        latency_seconds=0.125,
    )

    assert trades.empty
    assert order_log.empty
    assert "net_pnl" in trades.columns
    assert "resolved_time" in order_log.columns


def test_latency_loop_does_not_hide_failed_exit(loop_book):
    # Entry arrives at 0.125; exit arrives at 0.375, at row 3.
    loop_book.loc[3, "bid_size_1"] = 0
    scores = pd.Series([0.8], index=[0])

    with pytest.raises(RuntimeError):
        run_backtest_with_latency(
            loop_book,
            scores,
            horizon=2,
            latency_seconds=0.125,
        )


@pytest.mark.parametrize(
    "expiry, latency, deadline, expected_send, expected_arrival",
    [
        # Normal exit occurs before the deadline.
        (0.5, 0.125, 0.75, 0.5, 0.625),
        # Deadline advances the exit.
        (1.0, 0.125, 0.75, 0.75, 0.875),
        # Entry arrives after normal expiry: close immediately on fill.
        (0.125, 0.25, 0.75, 0.25, 0.5),
        # Entry arrives exactly at the deadline: still allowed.
        (1.0, 0.25, 0.25, 0.25, 0.5),
        # A deadline can also advance an exit at zero latency.
        (1.0, 0.0, 0.75, 0.75, 0.75),
    ],
)
def test_execution_deadline(expiry, latency, deadline, expected_send, expected_arrival):
    timing = compute_execution_times(
        decision_time=0.0,
        expiry_time=expiry,
        latency_seconds=latency,
        exit_send_deadline=deadline,
    )

    assert timing["exit_send_time"] == pytest.approx(expected_send)
    assert timing["exit_arrival_time"] == pytest.approx(expected_arrival)
    assert timing["exit_send_time"] <= deadline


def test_entry_after_execution_deadline_is_rejected():
    with pytest.raises(ValueError):
        compute_execution_times(
            decision_time=0.0,
            expiry_time=1.0,
            latency_seconds=0.5,
            exit_send_deadline=0.25,
        )


@pytest.mark.parametrize("deadline", [np.nan, np.inf, True, -1.0])
def test_invalid_execution_deadline(deadline):
    with pytest.raises(ValueError):
        compute_execution_times(
            decision_time=0.0,
            expiry_time=1.0,
            latency_seconds=0.125,
            exit_send_deadline=deadline,
        )


@pytest.mark.parametrize(
    "latency, expected_exit_time, expected_exit_index",
    [
        (0.0, 0.625, 5),
        (0.125, 0.75, 6),
    ],
)
def test_period_deadline_forces_early_exit(
    loop_book, latency, expected_exit_time, expected_exit_index
):
    scores = pd.Series([0.8], index=[0])

    trades, order_log = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=8,  # Normal expiry at time 1.0.
        latency_seconds=latency,
        entry_cutoff_time=0.5,
        exit_send_deadline=0.625,
    )

    assert len(trades) == 1

    trade = trades.iloc[0]

    assert bool(trade["forced_exit"])
    assert trade["expiry_time"] == pytest.approx(1.0)
    assert trade["exit_send_time"] == pytest.approx(0.625)
    assert trade["exit_time"] == pytest.approx(expected_exit_time)
    assert trade["exit_book_index"] == expected_exit_index

    assert order_log["forced_exit"].dtype == bool
    assert trade["exit_send_deadline"] == pytest.approx(0.625)


def test_period_cutoff_rejects_decisions_at_or_after_limit(loop_book):
    # These decisions occur at 0.5 and 0.625.
    scores = pd.Series([0.8, -0.8], index=[4, 5])

    trades, order_log = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=2,
        latency_seconds=0.125,
        entry_cutoff_time=0.5,
        exit_send_deadline=0.75,
    )

    assert trades.empty
    assert order_log.empty
    assert order_log["forced_exit"].dtype == bool


def test_period_deadline_allows_same_book_entry_and_exit(loop_book):
    scores = pd.Series([0.8], index=[1])

    trades, _ = run_backtest_with_latency(
        loop_book,
        scores,
        horizon=7,
        latency_seconds=0.0,
        entry_cutoff_time=0.15,
        exit_send_deadline=0.1875,
    )

    trade = trades.iloc[0]

    # Entry at 0.125; forced exit at 0.1875, before the next event.
    assert trade["entry_book_index"] == 1
    assert trade["exit_book_index"] == 1
    assert trade["entry_time"] == pytest.approx(0.125)
    assert trade["exit_time"] == pytest.approx(0.1875)
    assert bool(trade["forced_exit"])
    assert trade["net_pnl"] == pytest.approx(-0.02)


def test_period_deadline_requires_history_for_exit_arrival(loop_book):
    scores = pd.Series([0.0], index=[0])

    # History ends at 1.5, but the latest exit could arrive at 1.625.
    with pytest.raises(ValueError):
        run_backtest_with_latency(
            loop_book,
            scores,
            horizon=2,
            latency_seconds=0.125,
            entry_cutoff_time=1.0,
            exit_send_deadline=1.5,
        )


def test_decomposition_accepts_same_book_with_later_exit(loop_book):
    trades, _ = run_backtest_with_latency(
        loop_book,
        pd.Series([0.8], index=[1]),
        horizon=7,
        latency_seconds=0.0,
        entry_cutoff_time=0.15,
        exit_send_deadline=0.1875,
    )

    result = decompose_trade_pnl(loop_book, trades)
    trade = result.iloc[0]

    assert trade["entry_index"] == trade["exit_index"]
    assert trade["exit_time"] > trade["entry_time"]
    assert trade["mid_pnl"] == pytest.approx(0.0)
    assert trade["spread_cost"] == pytest.approx(0.02)
    assert trade["reconstructed_net_pnl"] == pytest.approx(-0.02)


def test_decomposition_rejects_exit_before_entry_time(loop_book):
    trades, _ = run_backtest_with_latency(
        loop_book,
        pd.Series([0.8], index=[1]),
        horizon=2,
        latency_seconds=0.0,
    )

    trades.loc[0, "exit_time"] = trades.loc[0, "entry_time"] - 0.01

    with pytest.raises(ValueError):
        decompose_trade_pnl(loop_book, trades)
