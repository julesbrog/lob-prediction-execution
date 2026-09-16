import numpy as np
import pandas as pd
import pytest

from lob.data import validate_book


@pytest.fixture
def valid_book():
    return pd.DataFrame(
        {
            "ask_price_1": [1_000_200],
            "ask_size_1": [10],
            "bid_price_1": [1_000_000],
            "bid_size_1": [12],
            "ask_price_2": [1_000_300],
            "ask_size_2": [20],
            "bid_price_2": [999_900],
            "bid_size_2": [22],
            "ask_price_3": [1_000_400],
            "ask_size_3": [30],
            "bid_price_3": [999_800],
            "bid_size_3": [32],
        },
        index=pd.Index([42], name="event_id"),
    )


def test_valid_book_is_unchanged(valid_book):
    original = valid_book.copy(deep=True)

    checks = validate_book(valid_book, n_levels=3)

    assert not checks.any().any()
    assert checks.index.equals(valid_book.index)
    pd.testing.assert_frame_equal(valid_book, original)


@pytest.mark.parametrize("column", ["ask_price_1", "bid_size_2"])
def test_infinite_values_are_flagged(valid_book, column):
    valid_book[column] = np.inf

    checks = validate_book(valid_book, n_levels=3)

    assert checks.loc[42, "has_infinite"]
    assert not checks.loc[42, "has_missing"]


def test_missing_values_are_flagged(valid_book):
    valid_book["ask_size_2"] = np.nan

    checks = validate_book(valid_book, n_levels=3)

    assert checks.loc[42, "has_missing"]
    assert not checks.loc[42, "has_infinite"]


@pytest.mark.parametrize(
    "column,value,flag",
    [
        ("ask_price_2", 1_000_100, "has_unordered_ask"),
        ("ask_price_2", 1_000_200, "has_unordered_ask"),
        ("bid_price_2", 1_000_100, "has_unordered_bid"),
        ("bid_price_2", 1_000_000, "has_unordered_bid"),
    ],
)
def test_wrong_price_order_is_flagged(valid_book, column, value, flag):
    valid_book.loc[42, column] = value

    checks = validate_book(valid_book, n_levels=3)

    assert checks.loc[42, flag]


@pytest.mark.parametrize(
    "side,sentinel",
    [("ask", 9_999_999_999), ("bid", -9_999_999_999)],
)
def test_trailing_empty_level_is_consistent(valid_book, side, sentinel):
    valid_book.loc[42, f"{side}_price_3"] = sentinel
    valid_book.loc[42, f"{side}_size_3"] = 0

    flags = validate_book(valid_book, n_levels=3).loc[42]

    assert flags["has_empty_level"]
    assert not flags.drop("has_empty_level").any()


def test_occupied_level_after_empty_level_is_flagged(valid_book):
    valid_book.loc[42, "ask_price_2"] = 9_999_999_999
    valid_book.loc[42, "ask_size_2"] = 0

    checks = validate_book(valid_book, n_levels=3)

    assert checks.loc[42, "has_invalid_empty_layout"]


def test_empty_level_with_volume_is_flagged(valid_book):
    valid_book.loc[42, "bid_price_3"] = -9_999_999_999

    checks = validate_book(valid_book, n_levels=3)

    assert checks.loc[42, "has_invalid_empty_size"]


def test_single_level_book(valid_book):
    checks = validate_book(valid_book, n_levels=1)

    assert not checks.any().any()
