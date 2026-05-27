"""Extract ChEBI ontology data using fastobo and build a networkx graph."""

from __future__ import annotations

from pathlib import Path

import fastobo
import networkx as nx


def _chebi_id_to_str(chebi_id: str) -> str:
    """Convert 'CHEBI:123' to '123' (string)."""
    return chebi_id.split(":")[1]


def _term_data(doc: "fastobo.term.TermFrame") -> dict | None:
    """Extract data from a single fastobo TermFrame.

    Returns
    -------
    dict or None
        Parsed term data, or ``None`` if the term is marked as obsolete.
    """
    parents: list[str] = []
    relations: dict = dict()
    name: str | None = None
    definition: str | None = None
    smiles: str | None = None
    subset: str | None = None

    for clause in doc:
        if isinstance(clause, fastobo.term.IsObsoleteClause):
            if clause.obsolete:
                return None
        elif isinstance(clause, fastobo.term.PropertyValueClause):
            pv = clause.property_value
            if str(pv.relation) in (
                "chemrof:smiles_string",
                "http://purl.obolibrary.org/obo/chebi/smiles",
            ):
                smiles = pv.value
        elif isinstance(clause, fastobo.term.SynonymClause):
            if "SMILES" in clause.raw_value() and smiles is None:
                smiles = clause.raw_value().split('"')[1]
        elif isinstance(clause, fastobo.term.RelationshipClause):
            relation = str(clause.typedef)
            if relation not in relations:
                relations[relation] = []
            relations[relation].append(_chebi_id_to_str(str(clause.term)))
        elif isinstance(clause, fastobo.term.IsAClause):
            parents.append(_chebi_id_to_str(str(clause.term)))
        elif isinstance(clause, fastobo.term.NameClause):
            name = str(clause.name)
        elif isinstance(clause, fastobo.term.DefClause):
            definition = str(clause.definition)
        elif isinstance(clause, fastobo.term.SubsetClause):
            subset = str(clause.subset)

    return {
        "id": _chebi_id_to_str(str(doc.id)),
        "parents": parents,
        "relations": relations,
        "name": name,
        "definition": definition,
        "smiles": smiles,
        "subset": subset,
    }


def build_chebi_graph(filepath: str | Path, top_class: str | None = "23367") -> nx.DiGraph:
    """Parse a ChEBI OBO file and build a directed graph of ontology terms.

    ``xref:`` lines are stripped before parsing as they can cause fastobo
    errors on some ChEBI releases.  Only non-obsolete CHEBI-prefixed terms
    are included.

    **Nodes** are string CHEBI IDs (e.g. ``"1"`` for ``CHEBI:1``) with
    attributes ``name``, ``smiles``, and ``subset``.

    **Edges** carry a ``relation`` attribute and represent:

    - ``is_a`` — directed from child to parent
    - ``has_part`` — directed from whole to part

    Parameters
    ----------
    filepath : str or Path
        Path to the ChEBI OBO file.
    top_class : str or None
        CHEBI ID of the top-class (default "23367" for "molecular entity").
        This will only return direct or indirect subclasses of the
        top-class (excluding the top-class). If ``top_class`` is not
        present in the parsed graph, the full graph is returned.
        If None, the full graph is returned without subgraph extraction.

    Returns
    -------
    nx.DiGraph
        Directed graph of ChEBI ontology terms and their relationships.
    """
    with open(filepath, encoding="utf-8") as f:
        content = "\n".join(line for line in f if not line.startswith("xref:"))

    graph: nx.DiGraph = nx.DiGraph()

    for frame in fastobo.loads(content):
        if not (
            frame and isinstance(frame.id, fastobo.id.PrefixedIdent) and frame.id.prefix == "CHEBI"
        ):
            continue

        term = _term_data(frame)
        if term is None:
            continue

        node_id = term["id"]
        graph.add_node(
            node_id,
            name=term["name"],
            definition=term["definition"],
            smiles=term["smiles"],
            subset=term["subset"],
        )

        for parent_id in term["parents"]:
            graph.add_edge(node_id, parent_id, relation="is_a")

        for relation, parts in term["relations"].items():
            for part_id in parts:
                graph.add_edge(node_id, part_id, relation=relation)

    if top_class is None:
        return graph

    hierarchy = get_hierarchy_subgraph(graph)
    if top_class not in hierarchy:
        return graph

    chebi_subgraph = nx.DiGraph(graph.subgraph(nx.ancestors(hierarchy, top_class)))
    return chebi_subgraph


def get_hierarchy_subgraph(chebi_graph: nx.DiGraph) -> nx.DiGraph:
    """Subgraph of ChEBI including only edges corresponding to hierarchical relations (is_a).
    Also removes nodes that are not connected by any is_a edges to other nodes."""
    return nx.DiGraph(
        chebi_graph.edge_subgraph(
            (u, v) for u, v, d in chebi_graph.edges(data=True) if d.get("relation") == "is_a"
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a ChEBI graph from an OBO file.")
    parser.add_argument("obo_file", type=Path, help="Path to the ChEBI OBO file.")
    args = parser.parse_args()

    graph = build_chebi_graph(args.obo_file)
    print(f"Final graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
