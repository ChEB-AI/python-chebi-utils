"""Tests for chebi_utils.splitter."""

from __future__ import annotations

import pandas as pd
import pytest

from chebi_utils.splitter import create_multilabel_splits, create_splits


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [f"CHEBI:{i}" for i in range(100)],
            "category": (["A"] * 50) + (["B"] * 30) + (["C"] * 20),
        }
    )


class TestCreateSplitsRandom:
    def test_returns_three_splits(self, sample_df):
        splits = create_splits(sample_df)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_split_sizes_sum_to_total(self, sample_df):
        splits = create_splits(sample_df)
        total = sum(len(v) for v in splits.values())
        assert total == len(sample_df)

    def test_default_ratios(self, sample_df):
        splits = create_splits(sample_df)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_reproducible_with_same_seed(self, sample_df):
        splits1 = create_splits(sample_df, seed=0)
        splits2 = create_splits(sample_df, seed=0)
        pd.testing.assert_frame_equal(splits1["train"], splits2["train"])

    def test_different_seeds_give_different_splits(self, sample_df):
        splits1 = create_splits(sample_df, seed=0)
        splits2 = create_splits(sample_df, seed=1)
        assert not splits1["train"]["id"].equals(splits2["train"]["id"])

    def test_no_overlap_between_splits(self, sample_df):
        splits = create_splits(sample_df)
        train_ids = set(splits["train"]["id"])
        val_ids = set(splits["val"]["id"])
        test_ids = set(splits["test"]["id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_rows_covered(self, sample_df):
        splits = create_splits(sample_df)
        all_ids = set(splits["train"]["id"]) | set(splits["val"]["id"]) | set(splits["test"]["id"])
        assert all_ids == set(sample_df["id"])

    def test_custom_ratios(self, sample_df):
        splits = create_splits(sample_df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 10

    def test_invalid_ratios_raise_error(self, sample_df):
        with pytest.raises(ValueError, match="must equal 1.0"):
            create_splits(sample_df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_negative_ratio_raises_error(self, sample_df):
        with pytest.raises(ValueError, match="between 0 and 1"):
            create_splits(sample_df, train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)


class TestCreateSplitsStratified:
    def test_stratified_split_returns_three_splits(self, sample_df):
        splits = create_splits(sample_df, stratify_col="category")
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_stratified_sizes_sum_to_total(self, sample_df):
        splits = create_splits(sample_df, stratify_col="category")
        total = sum(len(v) for v in splits.values())
        assert total == len(sample_df)

    def test_stratified_preserves_class_proportions(self, sample_df):
        splits = create_splits(sample_df, stratify_col="category")
        train = splits["train"]
        counts = train["category"].value_counts(normalize=True)
        # Category A: 50% of data -> ~50% of train
        assert abs(counts.get("A", 0) - 0.5) < 0.1

    def test_stratified_no_overlap(self, sample_df):
        splits = create_splits(sample_df, stratify_col="category")
        train_ids = set(splits["train"]["id"])
        val_ids = set(splits["val"]["id"])
        test_ids = set(splits["test"]["id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_stratified_reproducible(self, sample_df):
        splits1 = create_splits(sample_df, stratify_col="category", seed=42)
        splits2 = create_splits(sample_df, stratify_col="category", seed=42)
        pd.testing.assert_frame_equal(splits1["train"], splits2["train"])


@pytest.fixture
def multilabel_df():
    """DataFrame with multilabel 'labels' column (200 rows)."""
    all_labels = [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"], ["B", "C"]]
    labels = [all_labels[i % len(all_labels)] for i in range(200)]
    return pd.DataFrame(
        {
            "id": [f"CHEBI:{i}" for i in range(200)],
            "labels": labels,
        }
    )


@pytest.fixture
def singlelabel_df():
    """DataFrame with single-label 'labels' column."""
    return pd.DataFrame(
        {
            "id": [f"CHEBI:{i}" for i in range(200)],
            "labels": [["A"] if i % 2 == 0 else ["B"] for i in range(200)],
        }
    )


class TestCreateMultilabelSplits:
    def test_returns_three_splits(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df, labels_col="labels")
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_sizes_sum_to_total(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df, labels_col="labels")
        assert sum(len(v) for v in splits.values()) == len(multilabel_df)

    def test_no_overlap(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df, labels_col="labels")
        train_ids = set(splits["train"]["id"])
        val_ids = set(splits["val"]["id"])
        test_ids = set(splits["test"]["id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_rows_covered(self, multilabel_df):
        splits = create_multilabel_splits(multilabel_df, labels_col="labels")
        all_ids = set(splits["train"]["id"]) | set(splits["val"]["id"]) | set(splits["test"]["id"])
        assert all_ids == set(multilabel_df["id"])

    def test_reproducible_with_same_seed(self, multilabel_df):
        splits1 = create_multilabel_splits(multilabel_df, labels_col="labels", seed=7)
        splits2 = create_multilabel_splits(multilabel_df, labels_col="labels", seed=7)
        pd.testing.assert_frame_equal(splits1["train"], splits2["train"])

    def test_different_seeds_give_different_splits(self, multilabel_df):
        splits1 = create_multilabel_splits(multilabel_df, labels_col="labels", seed=1)
        splits2 = create_multilabel_splits(multilabel_df, labels_col="labels", seed=2)
        assert not splits1["train"]["id"].equals(splits2["train"]["id"])

    def test_approximate_split_sizes(self, multilabel_df):
        splits = create_multilabel_splits(
            multilabel_df, labels_col="labels", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        n = len(multilabel_df)
        assert abs(len(splits["test"]) - int(n * 0.1)) <= 2
        assert abs(len(splits["val"]) - int(n * 0.1)) <= 2

    def test_invalid_ratios_raise_error(self, multilabel_df):
        with pytest.raises(ValueError, match="must equal 1.0"):
            create_multilabel_splits(
                multilabel_df, labels_col="labels", train_ratio=0.5, val_ratio=0.3, test_ratio=0.3
            )

    def test_missing_labels_col_raises_error(self, multilabel_df):
        with pytest.raises(ValueError, match="not found in DataFrame"):
            create_multilabel_splits(multilabel_df, labels_col="nonexistent")

    def test_singlelabel_path(self, singlelabel_df):
        """Single-label lists should use StratifiedShuffleSplit without error."""
        splits = create_multilabel_splits(singlelabel_df, labels_col="labels")
        assert sum(len(v) for v in splits.values()) == len(singlelabel_df)
        train_ids = set(splits["train"]["id"])
        val_ids = set(splits["val"]["id"])
        test_ids = set(splits["test"]["id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
