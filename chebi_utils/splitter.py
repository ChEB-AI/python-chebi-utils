"""Generate stratified train/validation/test splits from ChEBI DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_col: str | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create stratified train/validation/test splits of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to split.
    train_ratio : float
        Fraction of data for training (default 0.8).
    val_ratio : float
        Fraction of data for validation (default 0.1).
    test_ratio : float
        Fraction of data for testing (default 0.1).
    stratify_col : str or None
        Column name to use for stratification. If None, splits are random.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys ``'train'``, ``'val'``, ``'test'``, each
        containing a DataFrame.

    Raises
    ------
    ValueError
        If the ratios do not sum to 1 or any ratio is outside ``[0, 1]``.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if any(r < 0 or r > 1 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("All ratios must be between 0 and 1")

    rng = np.random.default_rng(seed)

    if stratify_col is not None:
        return _stratified_split(df, train_ratio, val_ratio, test_ratio, stratify_col, rng)
    return _random_split(df, train_ratio, val_ratio, test_ratio, rng)


def _random_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,  # noqa: ARG001
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    indices = rng.permutation(len(df))
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def _stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,  # noqa: ARG001
    stratify_col: str,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in df.groupby(stratify_col, sort=False):
        group_indices = rng.permutation(np.array(group.index.tolist()))
        n = len(group_indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))

        train_indices.extend(group_indices[:n_train].tolist())
        val_indices.extend(group_indices[n_train : n_train + n_val].tolist())
        test_indices.extend(group_indices[n_train + n_val :].tolist())

    return {
        "train": df.loc[train_indices].reset_index(drop=True),
        "val": df.loc[val_indices].reset_index(drop=True),
        "test": df.loc[test_indices].reset_index(drop=True),
    }
