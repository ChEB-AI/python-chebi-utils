"""Tests for chebi_utils.sdf_extractor."""

from __future__ import annotations

import gzip
from pathlib import Path

from chebi_utils.sdf_extractor import extract_molecules

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_SDF = FIXTURES / "sample.sdf"


class TestExtractMolecules:
    def test_returns_two_molecules(self):
        df = extract_molecules(SAMPLE_SDF)
        assert len(df) == 2

    def test_chebi_id_column_present(self):
        df = extract_molecules(SAMPLE_SDF)
        assert "chebi_id" in df.columns

    def test_chebi_ids_correct(self):
        df = extract_molecules(SAMPLE_SDF)
        assert set(df["chebi_id"]) == {"CHEBI:1", "CHEBI:2"}

    def test_name_column_present(self):
        df = extract_molecules(SAMPLE_SDF)
        assert "name" in df.columns

    def test_smiles_column_present(self):
        df = extract_molecules(SAMPLE_SDF)
        assert "smiles" in df.columns

    def test_formula_column_present(self):
        df = extract_molecules(SAMPLE_SDF)
        assert "formula" in df.columns

    def test_inchi_column_present(self):
        df = extract_molecules(SAMPLE_SDF)
        assert "inchi" in df.columns

    def test_inchikey_column_present(self):
        df = extract_molecules(SAMPLE_SDF)
        assert "inchikey" in df.columns

    def test_molecule_properties(self):
        df = extract_molecules(SAMPLE_SDF)
        row = df[df["chebi_id"] == "CHEBI:1"].iloc[0]
        assert row["name"] == "compound A"
        assert row["smiles"] == "C"
        assert row["formula"] == "CH4"

    def test_gzipped_sdf(self, tmp_path):
        gz_path = tmp_path / "sample.sdf.gz"
        with open(SAMPLE_SDF, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.write(f_in.read())
        df = extract_molecules(gz_path)
        assert len(df) == 2
        assert set(df["chebi_id"]) == {"CHEBI:1", "CHEBI:2"}

    def test_empty_sdf_returns_empty_dataframe(self, tmp_path):
        empty_sdf = tmp_path / "empty.sdf"
        empty_sdf.write_text("")
        df = extract_molecules(empty_sdf)
        assert df.empty
