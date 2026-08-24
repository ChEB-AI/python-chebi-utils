"""Tests for chebi_utils.extract_properties steroid numbering."""

from __future__ import annotations

import pytest

from chebi_utils.extract_properties import get_steroid_positions, mol_to_fol_atoms
from chebi_utils.read_molecule import smiles_or_inchi_to_mol

# Cholesterol: gonane core + C18/C19 + cholestane side chain C20–C27
CHOLESTEROL_SMILES = (
    "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
)
# Estrone: aromatic ring A, C18 present, C19 and side chain absent
ESTRONE_SMILES = "C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)CCc4c3ccc(O)c4"
# Pregnenolone: short acetyl side chain C20–C21
PREGNENOLONE_SMILES = (
    "CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
)

CORE_POSITIONS = {f"steroid_{n}" for n in range(1, 18)}
CHOLESTEROL_SIDE_CHAIN = {f"steroid_{n}" for n in range(20, 28)}


def _steroid_keys(smiles: str) -> set[str]:
    mol = smiles_or_inchi_to_mol(smiles)
    assert mol is not None
    return set(get_steroid_positions(mol))


class TestSteroidPositions:
    def test_non_steroid_has_no_positions(self):
        assert get_steroid_positions(smiles_or_inchi_to_mol("c1ccccc1")) == {}

    def test_cholesterol_has_core_angular_methyls_and_side_chain(self):
        keys = _steroid_keys(CHOLESTEROL_SMILES)
        assert CORE_POSITIONS <= keys
        assert "steroid_18" in keys
        assert "steroid_19" in keys
        assert CHOLESTEROL_SIDE_CHAIN <= keys

    def test_estrone_has_c18_but_not_c19_or_side_chain(self):
        keys = _steroid_keys(ESTRONE_SMILES)
        assert CORE_POSITIONS <= keys
        assert "steroid_18" in keys
        assert "steroid_19" not in keys
        assert not any(int(key.split("_")[1]) >= 20 for key in keys)

    def test_pregnenolone_has_c20_and_c21_only(self):
        keys = _steroid_keys(PREGNENOLONE_SMILES)
        assert CORE_POSITIONS <= keys
        assert "steroid_20" in keys
        assert "steroid_21" in keys
        assert "steroid_22" not in keys

    @pytest.mark.parametrize(
        "smiles,attachment,methyl",
        [
            (CHOLESTEROL_SMILES, "steroid_13", "steroid_18"),
            (CHOLESTEROL_SMILES, "steroid_10", "steroid_19"),
            (ESTRONE_SMILES, "steroid_13", "steroid_18"),
        ],
    )
    def test_angular_methyl_is_bonded_to_attachment_atom(self, smiles, attachment, methyl):
        mol = smiles_or_inchi_to_mol(smiles)
        positions = get_steroid_positions(mol)
        attachment_idx = positions[attachment][0]
        methyl_idx = positions[methyl][0]
        attachment_atom = mol.GetAtomWithIdx(attachment_idx)
        neighbor_indices = {neighbor.GetIdx() for neighbor in attachment_atom.GetNeighbors()}
        assert methyl_idx in neighbor_indices
        assert mol.GetAtomWithIdx(methyl_idx).GetAtomicNum() == 6

    @pytest.mark.parametrize(
        "smiles,parent,child",
        [
            (CHOLESTEROL_SMILES, "steroid_17", "steroid_20"),
            (CHOLESTEROL_SMILES, "steroid_20", "steroid_21"),
            (CHOLESTEROL_SMILES, "steroid_20", "steroid_22"),
            (CHOLESTEROL_SMILES, "steroid_25", "steroid_26"),
            (CHOLESTEROL_SMILES, "steroid_25", "steroid_27"),
            (PREGNENOLONE_SMILES, "steroid_17", "steroid_20"),
            (PREGNENOLONE_SMILES, "steroid_20", "steroid_21"),
        ],
    )
    def test_side_chain_parent_child_bond(self, smiles, parent, child):
        mol = smiles_or_inchi_to_mol(smiles)
        positions = get_steroid_positions(mol)
        parent_idx = positions[parent][0]
        child_idx = positions[child][0]
        parent_atom = mol.GetAtomWithIdx(parent_idx)
        neighbor_indices = {neighbor.GetIdx() for neighbor in parent_atom.GetNeighbors()}
        assert child_idx in neighbor_indices

    def test_mol_to_fol_atoms_includes_steroid_predicates(self):
        mol = smiles_or_inchi_to_mol(CHOLESTEROL_SMILES)
        atom_facts, _mol_facts = mol_to_fol_atoms(mol, with_steroids=True)
        assert "steroid_1" in atom_facts
        assert "steroid_18" in atom_facts
        assert "steroid_19" in atom_facts
        assert "steroid_20" in atom_facts
        assert "steroid_27" in atom_facts

    def test_mol_to_fol_atoms_can_skip_steroids(self):
        mol = smiles_or_inchi_to_mol(CHOLESTEROL_SMILES)
        atom_facts, _mol_facts = mol_to_fol_atoms(mol, with_steroids=False)
        assert not any(key.startswith("steroid_") for key in atom_facts)
