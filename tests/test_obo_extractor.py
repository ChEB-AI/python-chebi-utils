"""Tests for chebi_utils.obo_extractor."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from chebi_utils.obo_extractor import build_chebi_graph

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_OBO = FIXTURES / "sample.obo"


class TestBuildChebiGraph:
    def test_returns_directed_graph(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert isinstance(g, nx.DiGraph)

    def test_correct_number_of_nodes(self):
        # CHEBI:4 is obsolete and must be excluded -> 4 nodes remain
        g = build_chebi_graph(SAMPLE_OBO)
        assert len(g.nodes) == 4

    def test_node_ids_are_integers(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert all(isinstance(n, int) for n in g.nodes)

    def test_expected_nodes_present(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert set(g.nodes) == {1, 2, 3, 5}

    def test_obsolete_term_excluded(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert 4 not in g.nodes

    def test_node_name_attribute(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes[1]["name"] == "compound A"
        assert g.nodes[2]["name"] == "compound B"

    def test_smiles_extracted_from_property_value(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes[1]["smiles"] == "C"

    def test_smiles_none_when_absent(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes[2]["smiles"] is None

    def test_subset_extracted(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes[3]["subset"] == "3_STAR"

    def test_subset_none_when_absent(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes[1]["subset"] is None

    def test_isa_edge_present(self):
        g = build_chebi_graph(SAMPLE_OBO)
        # CHEBI:1 is_a CHEBI:2
        assert g.has_edge(1, 2)
        assert g.edges[1, 2]["relation"] == "is_a"

    def test_has_part_edge_present(self):
        g = build_chebi_graph(SAMPLE_OBO)
        # CHEBI:1 has_part CHEBI:3
        assert g.has_edge(1, 3)
        assert g.edges[1, 3]["relation"] == "has_part"

    def test_total_edge_count(self):
        g = build_chebi_graph(SAMPLE_OBO)
        # 1->2 (is_a), 1->3 (has_part), 2->5 (is_a)
        assert len(g.edges) == 3

    def test_typedef_stanza_excluded(self):
        g = build_chebi_graph(SAMPLE_OBO)
        # "has_part" Typedef id is not numeric CHEBI ID, should not appear as node
        assert "has_part" not in g.nodes

    def test_xref_lines_do_not_break_parsing(self, tmp_path):
        obo_with_xrefs = tmp_path / "xref.obo"
        obo_with_xrefs.write_text(
            "format-version: 1.2\n[Term]\nid: CHEBI:10\nname: test\nxref: Reaxys:123456\n"
        )
        g = build_chebi_graph(obo_with_xrefs)
        assert 10 in g.nodes
