"""Tests for chebi_utils.dataset_builder."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from rdkit import Chem

from chebi_utils.dataset_builder import (
    build_labeled_dataset,
)


def _make_mol(smiles: str) -> Chem.Mol:
    """Helper to create a sanitised Mol from SMILES."""
    return Chem.MolFromSmiles(smiles)


@pytest.fixture
def simple_graph() -> nx.DiGraph:
    """Build a small ChEBI-like directed graph (child -> parent via is_a).

    Hierarchy::

        A ─is_a─> B ─is_a─> D
        A ─is_a─> C ─is_a─> D
        E ─is_a─> C

    Ontology descendants:
        D: {A, B, C, E}
        C: {A, E}
        B: {A}
        A: (none)
        E: (none)
    """
    g = nx.DiGraph()
    for node in ["A", "B", "C", "D", "E"]:
        g.add_node(node, name=node, smiles=None, subset=None)

    g.add_edge("A", "B", relation="is_a")
    g.add_edge("A", "C", relation="is_a")
    g.add_edge("B", "D", relation="is_a")
    g.add_edge("C", "D", relation="is_a")
    g.add_edge("E", "C", relation="is_a")
    return g


@pytest.fixture
def simple_molecules() -> pd.DataFrame:
    """Three molecules with IDs A, B, E (matching graph nodes)."""
    return pd.DataFrame(
        {
            "chebi_id": ["A", "B", "E"],
            "mol": [_make_mol("C"), _make_mol("CC"), _make_mol("CCC")],
        }
    )


class TestBuildLabeledDataset:
    def test_returns_dataframe_with_base_columns(self, simple_graph, simple_molecules):
        df, labels = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=2)
        assert "chebi_id" in df.columns
        assert "mol" in df.columns
        # Label columns should also be present
        for lbl in labels:
            assert lbl in df.columns

    def test_one_row_per_molecule(self, simple_graph, simple_molecules):
        df, _ = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=2)
        # 3 molecules with valid Mol -> 3 rows
        assert len(df) == 3

    def test_label_columns_are_boolean(self, simple_graph, simple_molecules):
        df, labels = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=2)
        for lbl in labels:
            assert df[lbl].dtype == bool

    def test_one_hot_values_correct(self, simple_graph, simple_molecules):
        # Labels (min=2): {B, C, D}
        # A -> ancestors {A,B,C,D} -> B=True, C=True, D=True
        # B -> ancestors {B,D}     -> B=True, C=False, D=True
        # E -> ancestors {E,C,D}   -> B=False, C=True, D=True
        df, _ = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=2)
        a_row = df[df["chebi_id"] == "A"].iloc[0]
        assert a_row["B"] == True  # noqa: E712
        assert a_row["C"] == True  # noqa: E712
        assert a_row["D"] == True  # noqa: E712

        b_row = df[df["chebi_id"] == "B"].iloc[0]
        assert b_row["B"] == True  # noqa: E712
        assert b_row["C"] == False  # noqa: E712
        assert b_row["D"] == True  # noqa: E712

        e_row = df[df["chebi_id"] == "E"].iloc[0]
        assert e_row["B"] == False  # noqa: E712
        assert e_row["C"] == True  # noqa: E712
        assert e_row["D"] == True  # noqa: E712

    def test_mol_objects_preserved(self, simple_graph, simple_molecules):
        df, _ = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=1)
        for _, row in df.iterrows():
            assert isinstance(row["mol"], Chem.rdchem.Mol)

    def test_none_mols_are_excluded(self, simple_graph):
        mol_df = pd.DataFrame(
            {
                "chebi_id": ["A", "B"],
                "mol": [_make_mol("C"), None],
            }
        )
        df, _ = build_labeled_dataset(simple_graph, mol_df, min_molecules=1)
        assert set(df["chebi_id"]) == {"A"}

    def test_high_threshold_returns_empty(self, simple_graph, simple_molecules):
        df, labels = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=100)
        assert df.empty
        assert labels == []

    def test_molecule_not_in_graph(self, simple_graph):
        """Molecules with chebi_ids not present in the graph are still handled."""
        mol_df = pd.DataFrame(
            {
                "chebi_id": ["Z"],
                "mol": [_make_mol("C")],
            }
        )
        df, labels = build_labeled_dataset(simple_graph, mol_df, min_molecules=1)
        assert "Z" in labels
        assert df.iloc[0]["Z"] == True  # noqa: E712

    def test_non_isa_edges_ignored(self):
        """Only is_a edges should be used for hierarchy traversal."""
        g = nx.DiGraph()
        for n in ["X", "Y", "Z"]:
            g.add_node(n, name=n, smiles=None, subset=None)
        g.add_edge("X", "Y", relation="is_a")
        g.add_edge("X", "Z", relation="has_part")

        mol_df = pd.DataFrame(
            {
                "chebi_id": ["X"],
                "mol": [_make_mol("C")],
            }
        )
        df, labels = build_labeled_dataset(g, mol_df, min_molecules=1)
        # X is_a Y, so labels should include X and Y (but NOT Z via has_part)
        assert set(labels) == {"X", "Y"}
        assert df.iloc[0]["X"] == True  # noqa: E712
        assert df.iloc[0]["Y"] == True  # noqa: E712

    def test_empty_molecules_dataframe(self, simple_graph):
        mol_df = pd.DataFrame(columns=["chebi_id", "mol"])
        df, labels = build_labeled_dataset(simple_graph, mol_df, min_molecules=1)
        assert df.empty
        assert labels == []

    def test_returned_labels_list_sorted(self, simple_graph, simple_molecules):
        _, labels = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=2)
        assert labels == ["B", "C", "D"]

    def test_returned_labels_match_columns(self, simple_graph, simple_molecules):
        df, labels = build_labeled_dataset(simple_graph, simple_molecules, min_molecules=1)
        label_cols = [c for c in df.columns if c not in ("chebi_id", "mol")]
        assert label_cols == labels
