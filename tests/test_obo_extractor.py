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
        # CHEBI:27189 is obsolete -> excluded;
        # 4 explicit + 5 implicit (superclasses and relation targets) = 9
        g = build_chebi_graph(SAMPLE_OBO)
        assert len(g.nodes) == 9

    def test_node_ids_are_strings(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert all(isinstance(n, str) for n in g.nodes)

    def test_expected_nodes_present(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert set(g.nodes) == {"10", "133004", "22750", "24921",
                                "28017", '75771', '28057', '28102', '37163'}

    def test_obsolete_term_excluded(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert "27189" not in g.nodes

    def test_node_name_attribute(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes["10"]["name"] == "(+)-Atherospermoline"
        assert g.nodes["133004"]["name"] == "bisbenzylisoquinoline alkaloid"

    def test_smiles_extracted_from_property_value(self):
        g = build_chebi_graph(SAMPLE_OBO)
        expected = (
            "COc1cc2c3cc1Oc1c(O)c(OC)cc4c1[C@H](Cc1ccc(O)c(c1)Oc1ccc(cc1)C[C@@H]3N(C)CC2)N(C)CC4"
        )
        assert g.nodes["10"]["smiles"] == expected

    def test_smiles_none_when_absent(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes["133004"]["smiles"] is None

    def test_subset_extracted(self):
        g = build_chebi_graph(SAMPLE_OBO)
        assert g.nodes["133004"]["subset"] == "3:STAR"

    def test_isa_edge_present(self):
        g = build_chebi_graph(SAMPLE_OBO)
        # CHEBI:10 is_a CHEBI:133004
        assert g.has_edge("10", "133004")
        assert g.edges["10", "133004"]["relation"] == "is_a"

    def test_isa_chain(self):
        g = build_chebi_graph(SAMPLE_OBO)
        # CHEBI:133004 is_a CHEBI:22750
        assert g.has_edge("133004", "22750")
        assert g.edges["133004", "22750"]["relation"] == "is_a"
        # CHEBI:22750 is_a CHEBI:24921
        assert g.has_edge("22750", "24921")
        assert g.edges["22750", "24921"]["relation"] == "is_a"

    def test_total_edge_count(self):
        g = build_chebi_graph(SAMPLE_OBO)
        # 10->133004 (is_a), 133004->22750 (is_a), 22750->24921 (is_a), ...
        assert len(g.edges) == 7

    def test_xref_lines_do_not_break_parsing(self, tmp_path):
        obo_with_xrefs = tmp_path / "xref.obo"
        obo_with_xrefs.write_text(
            "format-version: 1.2\n[Term]\nid: CHEBI:10\nname: test\nxref: Reaxys:123456\n"
        )
        g = build_chebi_graph(obo_with_xrefs)
        assert "10" in g.nodes
