"""Select specific sample subsets from the ChEBI dataset using the hierarchy."""

import queue

import networkx as nx

from chebi_utils.obo_extractor import get_hierarchy_subgraph


def get_closest_negatives(
    samples: list[str], chebi_graph: nx.DiGraph, target_id: str, min_samples=25, max_samples=None
) -> set[str]:
    """Find samples close to a target class in the hierarchy but not below it.

    Performs a breadth-first walk outward from ``target_id`` over the undirected
    ``is_a`` hierarchy, collecting samples that are subclasses of the visited
    neighbours (but not of ``target_id`` itself). The walk aims to reach
    ``min_samples``, then keeps collecting further-out (non-sibling) samples only
    until ``max_samples`` is reached.

    Parameters
    ----------
    samples : list[str]
        Candidate ChEBI IDs (as strings) to select from.
    chebi_graph : nx.DiGraph
        Full ChEBI ontology graph from :func:`build_chebi_graph`.
    target_id : str
        ChEBI ID whose neighbourhood is searched for negatives.
    min_samples : int
        Target number of samples to collect before stopping expansion
        (default 25).
    max_samples : int or None
        Hard upper bound on the number of samples collected. ``None`` (default)
        means no upper bound.

    Returns
    -------
    set[str]
        The selected ChEBI IDs.
    """
    hierarchy_graph = nx.transitive_closure_dag(get_hierarchy_subgraph(chebi_graph))
    undirected_graph = get_hierarchy_subgraph(chebi_graph).to_undirected()

    q = queue.Queue()
    q.put(target_id)
    visited = set()  # visit closest labels
    selected = (
        set()
    )  # select samples that are subclasses of closest labels until we have enough samples
    siblings = True
    while not q.empty():
        current = q.get()
        for neighbor in undirected_graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                q.put(neighbor)
                for neighbor_sub in hierarchy_graph.predecessors(neighbor):
                    if str(neighbor_sub) in samples:
                        selected.add(str(neighbor_sub))
                    if (max_samples and len(selected) >= max_samples) or (
                        len(selected) >= min_samples and not siblings
                    ):
                        return selected
        if len(selected) >= min_samples:
            break
        siblings = False

    return selected


def get_direct_neighbors(
    samples: list[str],
    chebi_graph: nx.DiGraph,
    target_id: str,
) -> tuple[list[str], list[str]]:
    """Filter samples and sort them into positives and direct-neighbor negatives.

    A sample is *positive* when it is a descendant of ``target_id``, and a
    *negative* when it is not a descendant of ``target_id`` but is a "direct
    neighbor" — a descendant of all direct parents of ``target_id``.

    Parameters
    ----------
    samples : list[str]
        Candidate ChEBI IDs (as strings) to filter.
    chebi_graph : nx.DiGraph
        Full ChEBI ontology graph from :func:`build_chebi_graph`.
    target_id : str
        ChEBI ID defining the positive class.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(pos_ids, neg_ids)`` where ``pos_ids`` are positive molecule IDs and
        ``neg_ids`` are negative molecule IDs (empty when the target has no
        siblings).
    """
    hierarchy_graph = nx.transitive_closure_dag(get_hierarchy_subgraph(chebi_graph))
    pos_ids = [str(d) for d in hierarchy_graph.predecessors(target_id) if str(d) in samples]

    sample_space_by_parent = dict()
    for parent in chebi_graph.successors(target_id):
        sample_space_by_parent[parent] = set()
        for desc in hierarchy_graph.predecessors(parent):
            s = str(desc)
            if s in samples:
                sample_space_by_parent[parent].add(s)
    if len(sample_space_by_parent) == 0:
        return pos_ids, []
    sample_space = set.intersection(*sample_space_by_parent.values())
    neg_ids = list(sample_space - set(pos_ids))
    return pos_ids, neg_ids
