import numpy as np
import pandas as pd


def make_targets(
    bid_raw: pd.Series,
    ask_raw: pd.Series,
    horizon: int = 50,
    epsilon_units: int = 0,
) -> pd.DataFrame:
    """Build labels using exact integer comparisons.

    Input prices are in LOBSTER units: dollars multiplied by 10,000.
    One epsilon unit corresponds to $0.00005 of mid-price movement.

    Output prices and changes are expressed in dollars.
    """
    for name, value in (
        ("horizon", horizon),
        ("epsilon_units", epsilon_units),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")

    if horizon <= 0:
        raise ValueError("horizon must be strictly positive")

    if epsilon_units < 0:
        raise ValueError("epsilon_units must be non-negative")

    if not isinstance(bid_raw, pd.Series) or not isinstance(ask_raw, pd.Series):
        raise ValueError("bid_raw and ask_raw must be pandas Series")

    if not bid_raw.index.equals(ask_raw.index):
        raise ValueError("bid and ask must have identical indices")

    if not bid_raw.index.is_unique:
        raise ValueError("event indices must be unique")

    # Bound each price so their sum cannot overflow int64.
    max_price = np.iinfo(np.int64).max // 2

    for name, prices in (("bid_raw", bid_raw), ("ask_raw", ask_raw)):
        if not pd.api.types.is_integer_dtype(
            prices.dtype
        ) or pd.api.types.is_bool_dtype(prices.dtype):
            raise ValueError(f"{name} must contain integer prices")

        if prices.isna().any():
            raise ValueError(f"{name} must not contain missing values")

        if prices.le(0).any():
            raise ValueError(f"{name} must contain strictly positive prices")

        if prices.isin([9999999999, -9999999999]).any():
            raise ValueError(f"{name} contains empty-level sentinel prices")

        if prices.gt(max_price).any():
            raise ValueError(f"{name} contains prices too large for safe arithmetic")

    bid = bid_raw.astype("Int64")
    ask = ask_raw.astype("Int64")

    if ask.lt(bid).any():
        raise ValueError("ask prices must be greater than or equal to bid prices")

    # Nullable integers preserve exact arithmetic after shifting.
    mid_sum = bid + ask
    future_sum = mid_sum.shift(-int(horizon))
    change_units = future_sum - mid_sum

    target = pd.Series(
        pd.NA,
        index=bid.index,
        dtype="Int64",
        name="target",
    )

    valid = change_units.notna()
    target.loc[valid] = 0
    target.loc[(change_units > int(epsilon_units)).fillna(False)] = 1
    target.loc[(change_units < -int(epsilon_units)).fillna(False)] = -1

    return pd.DataFrame(
        {
            "future_mid": future_sum.to_numpy(dtype=float, na_value=np.nan) / 20_000,
            "future_change": change_units.to_numpy(dtype=float, na_value=np.nan)
            / 20_000,
            "target": target,
        },
        index=bid.index,
    )
