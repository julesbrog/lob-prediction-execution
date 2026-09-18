import numpy as np
import pandas as pd


def moving_block_indices(
    n_observations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One moving block bootstrap resample of positions 0..n-1.

    Blocks of block_size consecutive positions drawn with replacement,
    concatenated, truncated to n.
    """

    if isinstance(n_observations, (bool, np.bool_)) or not isinstance(
        n_observations, (int, np.integer)
    ):
        raise ValueError("n_observations must be an integer")

    if isinstance(block_size, (bool, np.bool_)) or not isinstance(
        block_size, (int, np.integer)
    ):
        raise ValueError("block_size must be an integer")

    if n_observations <= 0:
        raise ValueError("n_observations must be strictly positive")

    if block_size <= 0:
        raise ValueError("block_size must be strictly positive")

    if block_size > n_observations:
        raise ValueError("block_size must not exceed n_observations")

    n_blocks = int(np.ceil(n_observations / block_size))
    starts = rng.integers(0, n_observations - block_size + 1, size=n_blocks)
    offsets = np.arange(block_size)

    indices = (starts[:, None] + offsets[None, :]).ravel()

    return indices[:n_observations]


def block_bootstrap(
    values: np.ndarray,
    statistic,
    block_size: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> dict[str, float]:
    """Percentile CI of statistic(values) with a moving block bootstrap.

    values must be in time order (rows, if 2-D). Returns the point estimate,
    the CI bounds, the bootstrap std error and the fraction of resamples > 0.
    """

    values = np.asarray(values, dtype=float)

    if values.ndim not in (1, 2):
        raise ValueError("values must be one- or two-dimensional")

    if values.size == 0:
        raise ValueError("values must not be empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")

    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise ValueError("n_resamples must be an integer")

    if n_resamples <= 0:
        raise ValueError("n_resamples must be strictly positive")

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    rng = np.random.default_rng(random_state)
    n_observations = len(values)

    estimates = np.empty(n_resamples, dtype=float)

    for i in range(int(n_resamples)):
        indices = moving_block_indices(n_observations, int(block_size), rng)
        estimates[i] = statistic(values[indices])

    if not np.all(np.isfinite(estimates)):
        raise RuntimeError("bootstrap produced non-finite statistics")

    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])

    return {
        "estimate": float(statistic(values)),
        "lower": float(lower),
        "upper": float(upper),
        "std_error": float(estimates.std(ddof=1)) if n_resamples > 1 else np.nan,
        "positive_fraction": float(np.mean(estimates > 0)),
    }


def bootstrap_group_means(
    values: pd.Series,
    groups: pd.Series,
    block_size: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Block bootstrap of the mean of values within each group.

    Blocks are drawn on the full sequence so group membership is resampled
    together with the values.
    """

    if not isinstance(values, pd.Series) or not isinstance(groups, pd.Series):
        raise ValueError("values and groups must be pandas Series")

    if not values.index.equals(groups.index):
        raise ValueError("values and groups must share the same index")

    if values.isna().any() or groups.isna().any():
        raise ValueError("values and groups must not contain missing values")

    array = values.to_numpy(dtype=float)

    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")

    labels, codes = np.unique(groups.to_numpy(), return_inverse=True)
    n_groups = len(labels)

    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, (int, np.integer)
    ):
        raise ValueError("n_resamples must be an integer")

    if n_resamples <= 0:
        raise ValueError("n_resamples must be strictly positive")

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    rng = np.random.default_rng(random_state)
    n_observations = len(array)

    def group_means(sample_values: np.ndarray, sample_codes: np.ndarray) -> np.ndarray:
        sums = np.bincount(sample_codes, weights=sample_values, minlength=n_groups)
        counts = np.bincount(sample_codes, minlength=n_groups)

        means = np.full(n_groups, np.nan, dtype=float)
        np.divide(sums, counts, out=means, where=counts > 0)

        return means

    estimates = np.empty((int(n_resamples), n_groups), dtype=float)

    for i in range(int(n_resamples)):
        indices = moving_block_indices(n_observations, int(block_size), rng)
        estimates[i] = group_means(array[indices], codes[indices])

    if np.isnan(estimates).any():
        raise RuntimeError("a group was empty in at least one resample")

    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2], axis=0)

    return pd.DataFrame(
        {
            "count": np.bincount(codes, minlength=n_groups),
            "estimate": group_means(array, codes),
            "lower": lower,
            "upper": upper,
            "std_error": estimates.std(axis=0, ddof=1),
            "positive_fraction": np.mean(estimates > 0, axis=0),
        },
        index=pd.Index(labels, name=groups.name or "group"),
    )
