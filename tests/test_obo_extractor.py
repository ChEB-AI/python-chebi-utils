"""Tests for chebi_utils.obo_extractor."""

from __future__ import annotations

from pathlib import Path

from chebi_utils.obo_extractor import extract_classes, extract_relations

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_OBO = FIXTURES / "sample.obo"


class TestExtractClasses:
    def test_returns_dataframe_with_expected_columns(self):
        df = extract_classes(SAMPLE_OBO)
        assert list(df.columns) == ["id", "name", "definition", "is_obsolete"]

    def test_correct_number_of_terms(self):
        df = extract_classes(SAMPLE_OBO)
        assert len(df) == 4

    def test_term_ids_are_present(self):
        df = extract_classes(SAMPLE_OBO)
        assert set(df["id"]) == {"CHEBI:1", "CHEBI:2", "CHEBI:3", "CHEBI:4"}

    def test_term_names_are_correct(self):
        df = extract_classes(SAMPLE_OBO)
        row = df[df["id"] == "CHEBI:1"].iloc[0]
        assert row["name"] == "compound A"

    def test_obsolete_flag(self):
        df = extract_classes(SAMPLE_OBO)
        assert df[df["id"] == "CHEBI:4"].iloc[0]["is_obsolete"]
        assert not df[df["id"] == "CHEBI:1"].iloc[0]["is_obsolete"]

    def test_definition_is_extracted(self):
        df = extract_classes(SAMPLE_OBO)
        row = df[df["id"] == "CHEBI:1"].iloc[0]
        assert "test compound" in row["definition"]

    def test_typedef_stanzas_are_excluded(self):
        df = extract_classes(SAMPLE_OBO)
        assert "has_role" not in df["id"].values


class TestExtractRelations:
    def test_returns_dataframe_with_expected_columns(self):
        df = extract_relations(SAMPLE_OBO)
        assert list(df.columns) == ["source_id", "target_id", "relation_type"]

    def test_isa_relations_extracted(self):
        df = extract_relations(SAMPLE_OBO)
        isa = df[df["relation_type"] == "is_a"]
        assert len(isa) == 2

    def test_typed_relation_extracted(self):
        df = extract_relations(SAMPLE_OBO)
        has_role = df[df["relation_type"] == "has_role"]
        assert len(has_role) == 1
        assert has_role.iloc[0]["source_id"] == "CHEBI:1"
        assert has_role.iloc[0]["target_id"] == "CHEBI:3"

    def test_isa_source_and_target(self):
        df = extract_relations(SAMPLE_OBO)
        isa = df[df["relation_type"] == "is_a"]
        sources = set(isa["source_id"])
        assert "CHEBI:1" in sources
        assert "CHEBI:2" in sources

    def test_total_relations_count(self):
        df = extract_relations(SAMPLE_OBO)
        assert len(df) == 3
