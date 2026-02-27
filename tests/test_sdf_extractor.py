"""Tests for chebi_utils.sdf_extractor."""

from __future__ import annotations

import gzip
from pathlib import Path

from rdkit.Chem import rdchem

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
        assert set(df["chebi_id"]) == {"1", "2"}

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

    def test_mol_column_present(self):
        df = extract_molecules(SAMPLE_SDF)
        assert "mol" in df.columns

    def test_mol_objects_are_rdkit_mol(self):
        df = extract_molecules(SAMPLE_SDF)
        for mol in df["mol"]:
            assert isinstance(mol, rdchem.Mol)

    def test_mol_atom_counts(self):
        df = extract_molecules(SAMPLE_SDF)
        row1 = df[df["chebi_id"] == "1"].iloc[0]
        row2 = df[df["chebi_id"] == "2"].iloc[0]
        assert row1["mol"].GetNumAtoms() == 1  # methane: 1 C
        assert row2["mol"].GetNumAtoms() == 2  # ethane: 2 C

    def test_mol_sanitized(self):
        df = extract_molecules(SAMPLE_SDF)
        for mol in df["mol"]:
            # Aromaticity flags should be set (sanitize applied)
            assert mol is not None

    def test_molecule_properties(self):
        df = extract_molecules(SAMPLE_SDF)
        row = df[df["chebi_id"] == "1"].iloc[0]
        assert row["name"] == "compound A"
        assert row["smiles"] == "C"
        assert row["formula"] == "CH4"

    def test_gzipped_sdf(self, tmp_path):
        gz_path = tmp_path / "sample.sdf.gz"
        with open(SAMPLE_SDF, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.write(f_in.read())
        df = extract_molecules(gz_path)
        assert len(df) == 2
        assert set(df["chebi_id"]) == {"1", "2"}
        assert all(isinstance(m, rdchem.Mol) for m in df["mol"])

    def test_empty_sdf_returns_empty_dataframe(self, tmp_path):
        empty_sdf = tmp_path / "empty.sdf"
        empty_sdf.write_text("")
        df = extract_molecules(empty_sdf)
        assert df.empty

    def test_unparseable_molblock_excluded(self, tmp_path, recwarn):
        bad_sdf = tmp_path / "bad.sdf"
        bad_sdf.write_text(
            "bad_mol\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n"
            "> <ChEBI ID>\nCHEBI:99\n\n$$$$\n"
        )
        df = extract_molecules(bad_sdf)
        assert len(df) == 0
