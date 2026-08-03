"""Tests for chebi_utils.read_molecule."""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import rdchem

from chebi_utils.read_molecule import parse_molblock, smiles_or_inchi_to_mol

ETHANOL_SMILES = "CCO"
ETHANOL_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
BENZENE_INCHI = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"

ETHANOL_MOLBLOCK = """ethanol
     RDKit          2D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2990    0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5981   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
"""


class TestSmilesToMol:
    @pytest.mark.parametrize(
        "smiles,num_atoms",
        [
            ("C", 1),  # methane
            (ETHANOL_SMILES, 3),
            ("c1ccccc1", 6),  # benzene
            ("[Na+].[Cl-]", 2),  # multi-component
            ("O=C(C)Oc1ccccc1C(=O)O", 13),  # aspirin
        ],
    )
    def test_valid_smiles_atom_counts(self, smiles, num_atoms):
        mol = smiles_or_inchi_to_mol(smiles)
        assert isinstance(mol, rdchem.Mol)
        assert mol.GetNumAtoms() == num_atoms

    def test_canonical_smiles_roundtrip(self):
        mol = smiles_or_inchi_to_mol("OCC")
        assert Chem.MolToSmiles(mol) == "CCO"

    def test_aromaticity_perceived(self):
        # Kekulized input must come back aromatic, i.e. sanitization ran
        mol = smiles_or_inchi_to_mol("C1=CC=CC=C1")
        assert all(atom.GetIsAromatic() for atom in mol.GetAtoms())

    def test_ring_info_available(self):
        mol = smiles_or_inchi_to_mol("C1CCCCC1")
        assert mol.GetRingInfo().NumRings() == 1

    def test_explicit_hydrogens_are_kept(self):
        # removeHs=False, so the explicit H atoms stay in the graph
        mol = smiles_or_inchi_to_mol("[H]C([H])([H])[H]")
        assert mol.GetNumAtoms() == 5

    def test_nitro_group_normalized_by_chembl_pipeline(self):
        # The ChEMBL structure pipeline rewrites N(=O)=O as the charge-separated form
        mol = smiles_or_inchi_to_mol("CN(=O)=O")
        assert Chem.MolToSmiles(mol) == "C[N+](=O)[O-]"

    def test_empty_string_gives_empty_mol(self):
        mol = smiles_or_inchi_to_mol("")
        assert mol is not None
        assert mol.GetNumAtoms() == 0

    @pytest.mark.parametrize(
        "smiles",
        [
            "not_a_molecule",
            "C(C",  # unclosed branch
            "C1CC",  # unclosed ring
            "[CH3",  # unclosed bracket atom
            "%%%",
        ],
    )
    def test_invalid_smiles_returns_none(self, smiles):
        with pytest.warns(UserWarning, match="RDKit failed at parsing"):
            assert smiles_or_inchi_to_mol(smiles) is None


class TestInchiToMol:
    @pytest.mark.parametrize(
        "inchi,num_atoms",
        [
            ("InChI=1S/CH4/h1H4", 1),  # methane
            ("InChI=1S/H2O/h1H2", 1),  # water
            (ETHANOL_INCHI, 3),
            (BENZENE_INCHI, 6),
        ],
    )
    def test_valid_inchi_atom_counts(self, inchi, num_atoms):
        mol = smiles_or_inchi_to_mol(inchi)
        assert isinstance(mol, rdchem.Mol)
        assert mol.GetNumAtoms() == num_atoms

    def test_inchi_and_smiles_agree(self):
        from_inchi = smiles_or_inchi_to_mol(ETHANOL_INCHI)
        from_smiles = smiles_or_inchi_to_mol(ETHANOL_SMILES)
        assert Chem.MolToSmiles(from_inchi) == Chem.MolToSmiles(from_smiles)

    def test_inchi_aromaticity_perceived(self):
        mol = smiles_or_inchi_to_mol(BENZENE_INCHI)
        assert Chem.MolToSmiles(mol) == "c1ccccc1"

    @pytest.mark.parametrize(
        "inchi",
        [
            "InChI=1S/garbage",
            "InChI=",
            "InChI=1S/CH4N/c1-2(1)1/h1H3",  # syntactically well-formed but not a real layer
        ],
    )
    def test_invalid_inchi_returns_none(self, inchi):
        with pytest.warns(UserWarning, match="RDKit failed at parsing"):
            assert smiles_or_inchi_to_mol(inchi) is None

    def test_inchi_without_prefix_is_treated_as_smiles(self):
        # Dispatch is based on the "InChI=" prefix, so a bare InChI body fails
        with pytest.warns(UserWarning, match="RDKit failed at parsing"):
            assert smiles_or_inchi_to_mol("1S/C2H6O/c1-2-3/h3H,2H2,1H3") is None


class TestSanitizationFallback:
    def test_unsanitizable_molecule_is_still_returned(self):
        # Neutral 5-valent nitrogen: sanitization fails, the raw mol comes back
        with pytest.warns(UserWarning, match="Failed to sanitize molecule"):
            mol = smiles_or_inchi_to_mol("C[N](C)(C)C")
        assert mol is not None
        assert mol.GetNumAtoms() == 5

    def test_ring_info_available_after_failed_sanitization(self):
        # FastFindRings runs in the fallback path, so ring queries do not raise
        with pytest.warns(UserWarning, match="Failed to sanitize molecule"):
            mol = smiles_or_inchi_to_mol("C1CCCCC1[N](C)(C)C")
        assert mol.GetRingInfo().NumRings() == 1

    def test_valid_molecule_emits_no_warning(self, recwarn):
        smiles_or_inchi_to_mol(ETHANOL_SMILES)
        assert len(recwarn) == 0


class TestParseMolblock:
    def test_valid_molblock(self):
        mol = parse_molblock(ETHANOL_MOLBLOCK, "CHEBI:16236")
        assert isinstance(mol, rdchem.Mol)
        assert mol.GetNumAtoms() == 3
        assert Chem.MolToSmiles(mol) == "CCO"

    def test_valid_molblock_without_chebi_id(self):
        mol = parse_molblock(ETHANOL_MOLBLOCK)
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    @pytest.mark.parametrize("molblock", ["", "nonsense", "one\ntwo\nthree\n"])
    def test_invalid_molblock_returns_none(self, molblock):
        with pytest.warns(UserWarning, match="Failed to parse molblock"):
            assert parse_molblock(molblock, "CHEBI:99") is None

    def test_warning_mentions_chebi_id(self):
        with pytest.warns(UserWarning, match="CHEBI:12345"):
            parse_molblock("nonsense", "CHEBI:12345")

    def test_atomless_molblock_returns_none(self):
        molblock = "empty\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n"
        with pytest.warns(UserWarning, match="Failed to parse molblock"):
            assert parse_molblock(molblock, "CHEBI:192499") is None
