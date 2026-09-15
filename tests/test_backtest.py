import pandas as pd
import pytest

from lob.backtest import run_backtest, simulate_round_trip


@pytest.fixture
def book():
    return pd.DataFrame(
        {
            "time": [10.0, 11.0],
            "bid_price_1": [1000000, 1000500],
            "bid_size_1": [10, 10],
            "ask_price_1": [1000200, 1000700],
            "ask_size_1": [20, 20],
        },
        # Verify that the function uses positions, not index labels.
        index=[100, 200],
    )


def test_long_round_trip(book):
    trade = simulate_round_trip(
        book,
        decision_index=0,
        side=1,
        horizon=1,
        fee_per_share=0.001,
    )

    assert trade is not None
    assert trade["entry_price"] == pytest.approx(100.02)
    assert trade["exit_price"] == pytest.approx(100.05)
    assert trade["gross_pnl"] == pytest.approx(0.03)
    assert trade["fees"] == pytest.approx(0.002)
    assert trade["net_pnl"] == pytest.approx(0.028)
    assert trade["entry_time"] == 10.0
    assert trade["exit_time"] == 11.0


def test_short_round_trip(book):
    trade = simulate_round_trip(
        book,
        decision_index=0,
        side=-1,
        horizon=1,
        fee_per_share=0.001,
    )

    assert trade["entry_price"] == pytest.approx(100.00)
    assert trade["exit_price"] == pytest.approx(100.07)
    assert trade["gross_pnl"] == pytest.approx(-0.07)
    assert trade["net_pnl"] == pytest.approx(-0.072)


def test_quantity_scales_pnl_and_fees(book):
    trade = simulate_round_trip(
        book,
        decision_index=0,
        side=1,
        horizon=1,
        quantity=5,
        fee_per_share=0.001,
    )

    assert trade["gross_pnl"] == pytest.approx(0.15)
    assert trade["fees"] == pytest.approx(0.01)
    assert trade["net_pnl"] == pytest.approx(0.14)


@pytest.mark.parametrize("side", [1, -1])
def test_unchanged_quotes_lose_one_spread(book, side):
    price_columns = ["bid_price_1", "ask_price_1"]
    book.loc[200, price_columns] = book.loc[100, price_columns]

    trade = simulate_round_trip(
        book,
        decision_index=0,
        side=side,
        horizon=1,
    )

    assert trade["gross_pnl"] == pytest.approx(-0.02)


def test_insufficient_entry_liquidity_returns_none(book):
    trade = simulate_round_trip(
        book,
        decision_index=0,
        side=1,
        horizon=1,
        quantity=21,
    )

    assert trade is None


def test_insufficient_exit_liquidity_raises(book):
    # Entry can buy 15 shares, but exit can sell only 10.
    with pytest.raises(RuntimeError):
        simulate_round_trip(
            book,
            decision_index=0,
            side=1,
            horizon=1,
            quantity=15,
        )


def test_exit_beyond_history_raises(book):
    with pytest.raises(ValueError):
        simulate_round_trip(
            book,
            decision_index=0,
            side=1,
            horizon=2,
        )


def test_negative_fees_raise(book):
    with pytest.raises(ValueError):
        simulate_round_trip(
            book,
            decision_index=0,
            side=1,
            horizon=1,
            fee_per_share=-0.001,
        )


@pytest.fixture
def constant_book():
    return pd.DataFrame(
        {
            "time": [float(i) for i in range(8)],
            "bid_price_1": [1000000] * 8,
            "bid_size_1": [10] * 8,
            "ask_price_1": [1000200] * 8,
            "ask_size_1": [10] * 8,
        }
    )


def test_backtest_prevents_overlapping_positions(constant_book):
    scores = pd.Series(
        [0.8, -0.8, 0.8, -0.8, 0.8, 0.8],
        index=range(6),
    )

    trades = run_backtest(
        constant_book,
        scores,
        horizon=2,
    )

    assert trades["entry_index"].tolist() == [0, 3]
    assert trades["exit_index"].tolist() == [2, 5]
    assert trades["side"].tolist() == [1, -1]

    # Two round trips on unchanged quotes lose two spreads.
    assert trades["net_pnl"].sum() == pytest.approx(-0.04)


def test_backtest_threshold_is_strict(constant_book):
    scores = pd.Series(
        [0.3, -0.3, 0.0],
        index=[0, 1, 2],
    )

    trades = run_backtest(
        constant_book,
        scores,
        horizon=2,
        threshold=0.3,
    )

    assert trades.empty
    assert "net_pnl" in trades.columns


def test_failed_entry_does_not_block_next_signal(constant_book):
    constant_book.loc[0, "ask_size_1"] = 0

    scores = pd.Series([0.8, 0.8], index=[0, 1])

    trades = run_backtest(
        constant_book,
        scores,
        horizon=2,
    )

    assert trades["entry_index"].tolist() == [1]
    assert trades["exit_index"].tolist() == [3]


def test_sparse_scores_preserve_event_horizon(constant_book):
    scores = pd.Series([0.8, -0.8], index=[1, 4])

    trades = run_backtest(
        constant_book,
        scores,
        horizon=2,
    )

    # Horizon counts original events, not rows in the scores Series.
    assert trades["entry_index"].tolist() == [1, 4]
    assert trades["exit_index"].tolist() == [3, 6]


def test_backtest_propagates_failed_exit(constant_book):
    constant_book.loc[2, "bid_size_1"] = 0
    scores = pd.Series([0.8], index=[0])

    with pytest.raises(RuntimeError):
        run_backtest(
            constant_book,
            scores,
            horizon=2,
        )
