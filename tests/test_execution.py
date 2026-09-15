import numpy as np
import pandas as pd
import pytest

from lob.execution import (
    find_arrival_book_index,
    get_aggressive_execution_price,
)


@pytest.fixture
def book():
    return pd.DataFrame(
        {
            "time": [10.000, 10.004, 10.009, 10.009, 10.020],
            "bid_price_1": [1000000, 1000100, 1000200, 1000300, 1000400],
            "bid_size_1": [10, 10, 10, 10, 10],
            "ask_price_1": [1000200, 1000300, 1000400, 1000500, 1000600],
            "ask_size_1": [20, 20, 20, 20, 20],
        },
        # Deliberately different from row positions.
        index=[100, 200, 300, 400, 500],
    )


def test_arrival_between_events(book):
    times = book["time"].to_numpy()

    index = find_arrival_book_index(times, 0, 0.006)

    assert index == 1


def test_zero_latency_preserves_decision_row(book):
    times = book["time"].to_numpy()

    # Rows 2 and 3 share a timestamp: zero latency must return row 2.
    assert find_arrival_book_index(times, 2, 0.0) == 2


def test_positive_latency_includes_equal_timestamps():
    times = np.array([0.0, 0.5, 0.5, 1.0])

    assert find_arrival_book_index(times, 0, 0.5) == 2


def test_arrival_after_history(book):
    times = book["time"].to_numpy()

    assert find_arrival_book_index(times, 4, 0.001) is None


def test_arrival_at_last_timestamp():
    times = np.array([0.0, 0.5, 1.0])

    assert find_arrival_book_index(times, 1, 0.5) == 2


def test_buy_at_ask_and_sell_at_bid(book):
    assert get_aggressive_execution_price(book, 0, 1) == pytest.approx(100.02)
    assert get_aggressive_execution_price(book, 0, -1) == pytest.approx(100.00)


def test_quantity_limits(book):
    assert get_aggressive_execution_price(book, 0, 1, 20) == pytest.approx(100.02)
    assert get_aggressive_execution_price(book, 0, 1, 21) is None

    assert get_aggressive_execution_price(book, 0, -1, 10) == pytest.approx(100.00)
    assert get_aggressive_execution_price(book, 0, -1, 11) is None


def test_delayed_buy_uses_arrival_quote(book):
    times = book["time"].to_numpy()

    index = find_arrival_book_index(times, 0, 0.006)
    price = get_aggressive_execution_price(book, index, side=1)

    # Ask at arrival is 100.03, not the decision-time ask of 100.02.
    assert price == pytest.approx(100.03)


@pytest.mark.parametrize(
    "book_index, side, quantity",
    [
        (-1, 1, 1),
        (5, 1, 1),
        (True, 1, 1),
        (0, 0, 1),
        (0, True, 1),
        (0, 1, 0),
        (0, 1, -1),
        (0, 1, 1.5),
    ],
)
def test_invalid_execution_arguments(book, book_index, side, quantity):
    with pytest.raises(ValueError):
        get_aggressive_execution_price(book, book_index, side, quantity)


@pytest.mark.parametrize("latency", [-0.001, np.nan, np.inf, True])
def test_invalid_latency(book, latency):
    with pytest.raises(ValueError):
        find_arrival_book_index(book["time"].to_numpy(), 0, latency)


def test_unsorted_timestamps():
    with pytest.raises(ValueError):
        find_arrival_book_index(np.array([0.0, 2.0, 1.0]), 0, 0.1)
