# functionality for selecting specific sample subsets from the ChEBI dataset

import networkx as nx
from chebi_utils.obo_extractor import get_hierarchy_subgraph


def get_closest_negatives(
    samples: list[str], chebi_graph: nx.DiGraph, target_id: str, min_samples=25, max_samples=None
) -> set[str]:
    # from the list of samples, find those that are not subclasses of the target_id, but close to it in the hierarchy.
    # goal: reach min_samples, but continue collecting samples (until max_samples) if they are siblings.
    hierarchy_graph = nx.transitive_closure_dag(get_hierarchy_subgraph(chebi_graph))
    undirected_graph = get_hierarchy_subgraph(chebi_graph).to_undirected()
    import queue

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
    """
    Filter samples and sort into two groups:
    positive: sample is a descendant of the target_id
    negative: sample is not a descendant of the target_id, but a "direct neighbor" -> a descendant of all direct parents of the target_id.

    Returns:
        pos_ids: list of positive validation molecule IDs
        neg_ids: list of negative validation molecule IDs
                 (empty when target has no siblings)
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
