import pandas as pd
import numpy as np


def make_targets(
    mid_price: pd.Series,
    horizon: int = 50,
    epsilon: float = 0.0,
) -> pd.DataFrame:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("horizon must be an integer")

    if horizon <= 0:
        raise ValueError("horizon must be strictly positive")

    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")

    if not np.all(np.isfinite(mid_price)) or (mid_price <= 0).any():
        raise ValueError("mid-prices must be finite and strictly positive")

    future_mid = mid_price.shift(-horizon)
    future_change = future_mid - mid_price

    target = pd.Series(pd.NA, index=mid_price.index, dtype="Int64")

    valid = future_change.notna()

    target.loc[valid] = 0
    target.loc[valid & (future_change > epsilon)] = 1
    target.loc[valid & (future_change < -epsilon)] = -1

    return pd.DataFrame(
        {
            "future_mid": future_mid,
            "future_change": future_change,
            "target": target,
        },
        index=mid_price.index,
    )
