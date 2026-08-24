"""Print CIP R/S at steroid core positions for a list of molecules.

Run: python -m chebi_utils.steroid_cip_ab
"""

from rdkit import Chem

from chebi_utils.extract_properties import get_steroid_positions
from chebi_utils.read_molecule import smiles_or_inchi_to_mol

STEREO_CORE_POSITIONS = (5, 8, 9, 10, 13, 14, 17)


def get_steroid_cip_fingerprint(mol: Chem.Mol) -> dict[int, str | None]:
    """CIP labels at C5/C8/C9/C10/C13/C14/C17 (None if missing/achiral)."""
    positions = get_steroid_positions(mol)
    try:
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        pass

    fingerprint: dict[int, str | None] = {}
    for iupac_position in STEREO_CORE_POSITIONS:
        atom_indices = positions.get(f"steroid_{iupac_position}")
        if not atom_indices:
            fingerprint[iupac_position] = None
            continue
        atom = mol.GetAtomWithIdx(atom_indices[0])
        fingerprint[iupac_position] = (
            atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        )
    return fingerprint


# Paste substances here: (name, SMILES_or_InChI, "alpha"|"beta")
EXAMPLES: list[tuple[str, str, str]] = [
    ("3-oxo-5β-steroid", "*C1CCC2C3CC[C@]4([H])CC(=O)CCC4(C)C3CCC12C", "beta"),
    ("5beta-androstane-3beta,17alpha-diol", "[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@]4(C)[C@H](O)CC[C@@]34[H])[C@@]1(C)CC[C@H](O)C2", "beta"),
    ("5beta-tetrahydro-11-deoxycorticosterone", "[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@@]4(C)[C@@]3([H])CC[C@]4([H])C(=O)CO)[C@@]1(C)CC[C@@H](O)C2", "beta"),
    (" 5β-chola-3,8(14),11-trien-24-oic acid", "[H][C@@]12C=CCC[C@]1(C)[C@@]1([H])C=C[C@@]3(C)C(=C1CC2)CC[C@]3([H])[C@H](C)CCC(=O)O", "beta"),
    ("5β-chola-3,8(14),11-trien-24-oic acid", "[H][C@@]12C=CCC[C@]1(C)[C@@]1([H])C=C[C@@]3(C)C(=C1CC2)CC[C@]3([H])[C@H](C)CCC(=O)O", "beta"),
    ("andrastin A", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21C=O", "beta"),
    ("andrastin B", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21CO", "beta"),
    ("andrastin C", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21C", "beta"),
    ("andrastin D", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)C(=O)CC[C@@]21C", "beta"),
    ("andrastin E", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@H](O)CC[C@@]21C", "beta"),
    ("andrastin F", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](O)CC[C@@]21C", "beta"),
    ("Tetrahydro-11-dehydrocorticosterone", "[H][C@]12CC[C@@]3([H])[C@]4([H])CC[C@]([H])(C(=O)CO)[C@@]4(C)CC(=O)[C@]3([H])[C@@]1(C)CC[C@@H](O)C2", "beta"),
]


if __name__ == "__main__":
    for name, smiles_or_inchi, label in EXAMPLES:
        mol = smiles_or_inchi_to_mol(smiles_or_inchi)
        if mol is None:
            print(f"{name}: parse_failed")
            continue
        fingerprint = get_steroid_cip_fingerprint(mol)
        cip_text = " ".join(
            f"C{position}={fingerprint.get(position) or '-'}"
            for position in STEREO_CORE_POSITIONS
        )
        print(f"{name} [{label}]: {cip_text}")
