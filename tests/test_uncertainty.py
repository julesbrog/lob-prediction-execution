import numpy as np
import pandas as pd
import pytest

from lob.uncertainty import (
    block_bootstrap,
    bootstrap_group_means,
    moving_block_indices,
)


def test_moving_block_indices_keeps_length_and_consecutive_blocks():
    rng = np.random.default_rng(0)
    indices = moving_block_indices(103, 10, rng)

    assert len(indices) == 103
    assert indices.min() >= 0 and indices.max() < 103

    # Inside each block, positions increase by exactly one.
    for start in range(0, 100, 10):
        block = indices[start : start + 10]
        assert np.all(np.diff(block) == 1)


def test_moving_block_indices_with_full_block_is_identity():
    rng = np.random.default_rng(0)
    indices = moving_block_indices(20, 20, rng)

    assert np.array_equal(indices, np.arange(20))


@pytest.mark.parametrize("n, block", [(0, 1), (10, 0), (10, 11), (10, True)])
def test_moving_block_indices_rejects_invalid_sizes(n, block):
    with pytest.raises(ValueError):
        moving_block_indices(n, block, np.random.default_rng(0))


def test_block_bootstrap_interval_contains_mean_of_iid_noise():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=0.5, scale=1.0, size=2_000)

    result = block_bootstrap(values, np.mean, block_size=50, n_resamples=500)

    assert result["estimate"] == pytest.approx(values.mean())
    assert result["lower"] < 0.5 < result["upper"]
    assert result["positive_fraction"] == 1.0


def test_block_bootstrap_widens_for_positively_autocorrelated_series():
    rng = np.random.default_rng(2)
    noise = rng.normal(size=5_000)

    # AR(1) with strong persistence.
    values = np.empty_like(noise)
    values[0] = noise[0]
    for t in range(1, len(noise)):
        values[t] = 0.9 * values[t - 1] + noise[t]

    naive = block_bootstrap(values, np.mean, block_size=1, n_resamples=300)
    blocked = block_bootstrap(values, np.mean, block_size=200, n_resamples=300)

    assert blocked["std_error"] > 2 * naive["std_error"]


def test_block_bootstrap_is_reproducible():
    values = np.arange(100, dtype=float)

    first = block_bootstrap(values, np.mean, block_size=10, random_state=3)
    second = block_bootstrap(values, np.mean, block_size=10, random_state=3)

    assert first == second


def test_block_bootstrap_rejects_bad_inputs():
    with pytest.raises(ValueError):
        block_bootstrap(np.array([1.0, np.nan]), np.mean, block_size=1)

    with pytest.raises(ValueError):
        block_bootstrap(np.array([]), np.mean, block_size=1)

    with pytest.raises(ValueError):
        block_bootstrap(np.ones(10), np.mean, block_size=1, confidence=1.0)


def test_bootstrap_group_means_matches_direct_means():
    index = pd.RangeIndex(600)
    groups = pd.Series(np.repeat([0, 1, 2], 200), index=index, name="bin")
    values = pd.Series(
        np.concatenate([np.full(200, -1.0), np.zeros(200), np.full(200, 2.0)]),
        index=index,
    )

    result = bootstrap_group_means(values, groups, block_size=20, n_resamples=200)

    assert list(result.index) == [0, 1, 2]
    assert result["count"].tolist() == [200, 200, 200]
    assert result["estimate"].tolist() == [-1.0, 0.0, 2.0]

    # Constant groups have zero-width intervals.
    assert np.allclose(result["lower"], result["estimate"])
    assert np.allclose(result["upper"], result["estimate"])


def test_bootstrap_group_means_rejects_mismatched_index():
    values = pd.Series([1.0, 2.0], index=[0, 1])
    groups = pd.Series([0, 1], index=[1, 2])

    with pytest.raises(ValueError):
        bootstrap_group_means(values, groups, block_size=1)
