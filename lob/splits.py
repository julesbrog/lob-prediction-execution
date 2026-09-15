import math
import pandas as pd
import numpy as np


def make_temporal_splits(
    n_events: int,
    horizon: int = 50,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> dict[str, slice]:

    for name, value in (("n_events", n_events), ("horizon", horizon)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be strictly positive")

    for name, value in (
        ("train_fraction", train_fraction),
        ("val_fraction", val_fraction),
    ):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")

    if train_fraction + val_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    train_end = int(n_events * train_fraction)
    val_end = int(n_events * (train_fraction + val_fraction))

    splits = {
        "train": slice(0, train_end - horizon),
        "validation": slice(train_end, val_end - horizon),
        "test": slice(val_end, n_events - horizon),
    }

    for name, selection in splits.items():
        if selection.stop <= selection.start:
            raise ValueError(f"{name} is empty after purging")

    return splits


def prepare_datasets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    splits: dict[str, slice],
    feature_columns: list[str],
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Select temporal subsets and remove invalid rows with a shared mask."""

    if not features.index.equals(targets.index):
        raise ValueError("features and targets must have the same index")

    if not features.index.is_unique:
        raise ValueError("event indices must be unique")

    if not features.columns.is_unique or not targets.columns.is_unique:
        raise ValueError("column names must be unique")

    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be non-empty and unique")

    missing_cols = [
        column for column in feature_columns if column not in features.columns
    ]
    if missing_cols:
        raise ValueError(f"missing feature columns: {missing_cols}")

    if "target" not in targets.columns:
        raise ValueError("targets must contain a 'target' column")

    forbidden_cols = {"target", "future_mid", "future_change"}
    if forbidden_cols.intersection(feature_columns):
        raise ValueError("future information must not be used as features")

    datasets = {}

    for name, selection in splits.items():
        X = features.iloc[selection][feature_columns].copy()
        y = targets.iloc[selection]["target"].copy()

        finite_features = pd.Series(
            np.isfinite(X.to_numpy(dtype=float, na_value=np.nan)).all(axis=1),
            index=X.index,
        )
        valid = finite_features & y.notna()

        X = X.loc[valid].copy()
        y = y.loc[valid].copy()

        if X.empty:
            raise ValueError(f"{name} is empty after removing invalid rows")

        if not y.isin([-1, 0, 1]).all():
            raise ValueError(f"{name} contains invalid target classes")

        datasets[name] = (X, y.astype("int64"))

    return datasets
