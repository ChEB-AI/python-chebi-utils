"""Generate stratified train/validation/test splits from ChEBI DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_multilabel_splits(
    df: pd.DataFrame,
    labels_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = 42,
) -> dict[str, pd.DataFrame]:
    """Create stratified train/validation/test splits for multilabel DataFrames.

    Automatically detects whether the dataset is multilabel (each entry has
    more than one label) or single-label, and applies the appropriate
    stratification strategy:

    - Multilabel: uses ``MultilabelStratifiedShuffleSplit`` from the
      ``iterative-stratification`` package.
    - Single-label: uses ``StratifiedShuffleSplit`` from ``scikit-learn``.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to split.  Must contain a column ``labels_col`` whose
        values are sequences of labels (e.g. lists of strings or ints).
    labels_col : str
        Name of the column that contains the label sequences.
    train_ratio : float
        Fraction of data for training (default 0.8).
    val_ratio : float
        Fraction of data for validation (default 0.1).
    test_ratio : float
        Fraction of data for testing (default 0.1).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys ``'train'``, ``'val'``, ``'test'``, each
        containing a DataFrame.

    Raises
    ------
    ValueError
        If the ratios do not sum to 1, any ratio is outside ``[0, 1]``, or
        ``labels_col`` is not found in *df*.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if any(r < 0 or r > 1 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("All ratios must be between 0 and 1")
    if labels_col not in df.columns:
        raise ValueError(f"Column '{labels_col}' not found in DataFrame")

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import MultiLabelBinarizer

    labels_list: list[list] = df[labels_col].tolist()
    is_multilabel = any(len(lbl) > 1 for lbl in labels_list)

    df_reset = df.reset_index(drop=True)

    if is_multilabel:
        mlb = MultiLabelBinarizer()
        labels_matrix = mlb.fit_transform(labels_list)
    else:
        labels_matrix = [lbl[0] for lbl in labels_list]

    # ── Step 1: carve out the test set ──────────────────────────────────────
    if is_multilabel:
        test_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        )
        train_val_idx, test_idx = next(test_splitter.split(labels_matrix, labels_matrix))
    else:
        test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(test_splitter.split(labels_matrix, labels_matrix))

    df_test = df_reset.iloc[test_idx]
    df_trainval = df_reset.iloc[train_val_idx]

    # ── Step 2: split train/val from the remaining data ─────────────────────
    labels_trainval = (
        labels_matrix[train_val_idx]
        if is_multilabel
        else [labels_matrix[i] for i in train_val_idx]
    )
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)

    if is_multilabel:
        val_splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio_adjusted, random_state=seed
        )
        train_idx_inner, val_idx_inner = next(val_splitter.split(labels_trainval, labels_trainval))
    else:
        val_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio_adjusted, random_state=seed
        )
        train_idx_inner, val_idx_inner = next(val_splitter.split(labels_trainval, labels_trainval))

    df_train = df_trainval.iloc[train_idx_inner]
    df_val = df_trainval.iloc[val_idx_inner]

    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }


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
