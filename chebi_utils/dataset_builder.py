"""Build a labeled dataset by matching molecules to ChEBI ontology classes."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import pandas as pd

from chebi_utils.obo_extractor import get_hierarchy_subgraph


def _count_molecules_per_class(closure: nx.DiGraph, mol_ids: set[str]) -> dict[str, int]:
    """Count how many molecules fall under each ontology class.

    Uses the precomputed transitive closure so that ancestor look-ups are O(1).

    Parameters
    ----------
    closure : nx.DiGraph
        Transitive closure of the ``is_a`` hierarchy.
    mol_ids : set[str]
        ChEBI IDs of molecules with valid ``Mol`` objects.

    Returns
    -------
    dict[str, int]
        Mapping from ChEBI class ID to count of molecules in its subtree.
    """
    counts: Counter[str] = Counter()
    for mid in mol_ids:
        if mid in closure:
            for ancestor in closure.successors(mid):
                counts[ancestor] += 1
        # The molecule itself always counts for its own class
        counts[mid] += 1
    return dict(counts)


def build_labeled_dataset(
    chebi_graph: nx.DiGraph,
    molecules: pd.DataFrame,
    min_molecules: int = 50,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a labeled dataset matching molecules to ontology classes.

    Each molecule is assigned to every selected label class that it belongs to
    (directly or through a chain of ``is_a`` relationships).  Only classes with
    at least *min_molecules* descendant molecules (including indirect
    descendants) are retained as labels.

    Labels are encoded **one-hot**: the returned DataFrame contains one boolean
    column per selected label.

    Parameters
    ----------
    chebi_graph : nx.DiGraph
        Full ChEBI ontology graph from :func:`build_chebi_graph`.
    molecules : pd.DataFrame
        DataFrame from :func:`extract_molecules` containing at least
        ``chebi_id`` and ``mol`` columns.
    min_molecules : int
        Minimum number of descendant molecules a class must have to be
        selected as a label (default 50).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A tuple of:
        - DataFrame with columns ``chebi_id``, ``mol``, and one boolean
          column per selected label.  Each row represents one molecule.
        - Sorted list of selected label ChEBI IDs.
    """
    # Keep only molecules with a valid Mol object
    mol_df = molecules[molecules["mol"].notna()].copy()
    mol_ids = set(mol_df["chebi_id"])

    # Build transitive closure of hierarchy once
    hierarchy = get_hierarchy_subgraph(chebi_graph)
    closure = nx.transitive_closure_dag(hierarchy)

    # Determine label set
    counts = _count_molecules_per_class(closure, mol_ids)
    labels = {cls for cls, count in counts.items() if count >= min_molecules}
    sorted_labels = sorted(labels)

    if not labels:
        return pd.DataFrame(columns=["chebi_id", "mol"]), sorted_labels

    # For each molecule compute its ancestor set (including itself) via closure
    label_matrix: list[dict[str, bool]] = []
    for cid in mol_df["chebi_id"]:
        if cid in closure:
            ancestors = set(closure.successors(cid)) | {cid}
        else:
            ancestors = {cid}
        mol_labels = ancestors & labels
        label_matrix.append({lbl: lbl in mol_labels for lbl in sorted_labels})

    label_df = pd.DataFrame(label_matrix, index=mol_df.index)
    result = pd.concat(
        [mol_df[["chebi_id", "mol"]].reset_index(drop=True), label_df.reset_index(drop=True)],
        axis=1,
    )

    return result, sorted_labels
