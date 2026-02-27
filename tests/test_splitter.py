"""Tests for chebi_utils.splitter."""

from __future__ import annotations

import pandas as pd
import pytest

from chebi_utils.splitter import create_multilabel_splits


@pytest.fixture
def multilabel_df():
    """DataFrame with three bool label columns starting at index 2 (200 rows).

    Column layout: chebi_id | mol | label_A | label_B | label_C
    Each row gets one of six label combinations in a round-robin pattern.
    """
    patterns = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ]
    rows = [patterns[i % len(patterns)] for i in range(200)]
    label_a, label_b, label_c = zip(*rows)
    return pd.DataFrame(
        {
            "chebi_id": [f"CHEBI:{i}" for i in range(200)],
            "mol": ["mol"] * 200,
            "label_A": list(label_a),
            "label_B": list(label_b),
            "label_C": list(label_c),
        }
    )


@pytest.fixture
def singlelabel_df():
    """DataFrame with a single bool label column at index 2 (200 rows)."""
    return pd.DataFrame(
        {
            "chebi_id": [f"CHEBI:{i}" for i in range(200)],
            "mol": ["mol"] * 200,
            "label_A": [i % 2 == 0 for i in range(200)],
        }
    )


class TestCreateMultilabelSplits:
    def test_returns_three_splits(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_sizes_sum_to_total(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df)
        assert sum(len(v) for v in splits.values()) == len(multilabel_df)

    def test_no_overlap(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df)
        train_ids = set(splits["train"]["chebi_id"])
        val_ids = set(splits["val"]["chebi_id"])
        test_ids = set(splits["test"]["chebi_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_rows_covered(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df)
        all_ids = (
            set(splits["train"]["chebi_id"])
            | set(splits["val"]["chebi_id"])
            | set(splits["test"]["chebi_id"])
        )
        assert all_ids == set(multilabel_df["chebi_id"])

    def test_label_columns_preserved(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df)
        for split in splits.values():
            assert list(split.columns) == list(multilabel_df.columns)

    def test_reproducible_with_same_seed(self, multilabel_df):
        splits1 = create_multilabel_splits(multilabel_df, seed=7)
        splits2 = create_multilabel_splits(multilabel_df, seed=7)
        pd.testing.assert_frame_equal(splits1["train"], splits2["train"])

    def test_different_seeds_give_different_splits(self, multilabel_df):
        splits1 = create_multilabel_splits(multilabel_df, seed=1)
        splits2 = create_multilabel_splits(multilabel_df, seed=2)
        assert not splits1["train"]["chebi_id"].equals(splits2["train"]["chebi_id"])

    def test_approximate_split_sizes(self, multilabel_df):
        splits = create_multilabel_splits(
            multilabel_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        n = len(multilabel_df)
        assert abs(len(splits["test"]) - int(n * 0.1)) <= 2
        assert abs(len(splits["val"]) - int(n * 0.1)) <= 2

    def test_custom_label_start_col(self, multilabel_df):
        # Drop the 'mol' column so labels start at index 1
        df_no_mol = multilabel_df.drop(columns=["mol"])
        splits = create_multilabel_splits(df_no_mol, label_start_col=1)
        assert sum(len(v) for v in splits.values()) == len(df_no_mol)

    def test_invalid_ratios_raise_error(self, multilabel_df):
        with pytest.raises(ValueError, match="must equal 1.0"):
            create_multilabel_splits(multilabel_df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_out_of_range_label_start_col_raises_error(self, multilabel_df):
        with pytest.raises(ValueError, match="out of range"):
            create_multilabel_splits(multilabel_df, label_start_col=100)

    def test_singlelabel_path(self, singlelabel_df):
        """Single label column should use StratifiedShuffleSplit without error."""
        splits = create_multilabel_splits(singlelabel_df)
        assert sum(len(v) for v in splits.values()) == len(singlelabel_df)
        train_ids = set(splits["train"]["chebi_id"])
        val_ids = set(splits["val"]["chebi_id"])
        test_ids = set(splits["test"]["chebi_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
