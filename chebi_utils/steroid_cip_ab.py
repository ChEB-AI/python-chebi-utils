"""Explore C5 steroid parity (+1/-1) vs known alpha/beta labels.

Uses RDKit tetrahedral CW/CCW remapped to a fixed steroid neighbor order
(C4, C6, C10), not CIP.

Run: python -m chebi_utils.steroid_cip_ab
"""

from rdkit import Chem

from chebi_utils.extract_properties import get_steroid_positions
from chebi_utils.read_molecule import smiles_or_inchi_to_mol


def _iupac_to_atom(mol: Chem.Mol) -> dict[int, int]:
    positions = get_steroid_positions(mol)
    return {
        int(key.split("_")[1]): atom_indices[0]
        for key, atom_indices in positions.items()
        if key.startswith("steroid_")
    }


def c5_steroid_parity(mol: Chem.Mol, iupac_to_atom: dict[int, int]) -> int | None:
    """Local C5 parity relative to steroid numbering C4, C6, C10.

    Returns ``+1`` / ``-1``, or ``None`` if C5 is missing / not tetrahedral /
    neighbours are unexpected. Does not name alpha/beta yet.
    """
    try:
        c4_idx = iupac_to_atom[4]
        c5_idx = iupac_to_atom[5]
        c6_idx = iupac_to_atom[6]
        c10_idx = iupac_to_atom[10]
    except KeyError:
        return None

    c5_atom = mol.GetAtomWithIdx(c5_idx)
    chiral_tag = c5_atom.GetChiralTag()
    if chiral_tag == Chem.ChiralType.CHI_UNSPECIFIED:
        return None

    neighbor_indices = [neighbor.GetIdx() for neighbor in c5_atom.GetNeighbors()]
    heavy_neighbor_indices = [
        atom_idx
        for atom_idx in neighbor_indices
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 1
    ]
    hydrogen_neighbor_indices = [
        atom_idx
        for atom_idx in neighbor_indices
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 1
    ]
    wanted_heavy_order = [c4_idx, c6_idx, c10_idx]
    if set(heavy_neighbor_indices) != set(wanted_heavy_order):
        return None
    if len(heavy_neighbor_indices) != 3:
        return None

    # RDKit CW/CCW follows GetNeighbors() order. With implicit H that is the
    # three heavy atoms; with explicit H the H is included in the order.
    if not hydrogen_neighbor_indices:
        observed_order = heavy_neighbor_indices
        wanted_order = wanted_heavy_order
    elif len(hydrogen_neighbor_indices) == 1:
        observed_order = neighbor_indices
        wanted_order = wanted_heavy_order + hydrogen_neighbor_indices
    else:
        return None

    permutation = [wanted_order.index(atom_idx) for atom_idx in observed_order]
    inversions = sum(
        1
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
        if permutation[left] > permutation[right]
    )
    permutation_sign = -1 if inversions % 2 else 1

    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        rdkit_sign = 1
    elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        rdkit_sign = -1
    else:
        return None

    return rdkit_sign * permutation_sign


# Paste substances here: (name, SMILES_or_InChI, "alpha"|"beta")
EXAMPLES: list[tuple[str, str, str]] = [
    ("3-oxo-5β-steroid", "*C1CCC2C3CC[C@]4([H])CC(=O)CCC4(C)C3CCC12C", "beta"),
    ("5beta-androstane-3beta,17alpha-diol", "[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@]4(C)[C@H](O)CC[C@@]34[H])[C@@]1(C)CC[C@H](O)C2", "beta"),
    ("5beta-tetrahydro-11-deoxycorticosterone", "[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@@]4(C)[C@@]3([H])CC[C@]4([H])C(=O)CO)[C@@]1(C)CC[C@@H](O)C2", "beta"),
    ("5β-chola-3,8(14),11-trien-24-oic acid", "[H][C@@]12C=CCC[C@]1(C)[C@@]1([H])C=C[C@@]3(C)C(=C1CC2)CC[C@]3([H])[C@H](C)CCC(=O)O", "beta"),
    ("5β-chola-3,8(14),11-trien-24-oic acid", "[H][C@@]12C=CCC[C@]1(C)[C@@]1([H])C=C[C@@]3(C)C(=C1CC2)CC[C@]3([H])[C@H](C)CCC(=O)O", "beta"),
    ("andrastin A", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21C=O", "beta"),
    ("andrastin B", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21CO", "beta"),
    ("andrastin C", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21C", "beta"),
    ("andrastin D", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)C(=O)CC[C@@]21C", "beta"),
    ("andrastin E", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@H](O)CC[C@@]21C", "beta"),
    ("andrastin F", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](O)CC[C@@]21C", "beta"),
    ("Tetrahydro-11-dehydrocorticosterone", "[H][C@]12CC[C@@]3([H])[C@]4([H])CC[C@]([H])(C(=O)CO)[C@@]4(C)CC(=O)[C@]3([H])[C@@]1(C)CC[C@@H](O)C2", "beta"),
    ("25S)-5alpha-cholestan..", "[H][C@@]12[C@@H](O)[C@@H](O)[C@]3(O)[C@]([H])(CC[C@]4(C)[C@@]([H])([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@@H](O)[C@]43[H])[C@@]1(C)CC[C@H](O)[C@@H]2O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta ...", "[H][C@@]12[C@@H](O)C[C@]3(O)[C@]([H])(CC[C@]4(C)[C@@]([H])([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@H](O)[C@]43[H])[C@@]1(C)CC[C@H](O)[C@@H]2O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta,6alpha,7beta,8", "[H][C@]1([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@H](O)[C@]2([H])[C@]1(C)CC[C@]1([H])[C@@]3(C)CC[C@H](O)C[C@]3([H])[C@@H](O)[C@H](O)[C@]12O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta,6al...", "[H][C@]1([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@@H](O)[C@]2([H])[C@]1(C)CC[C@]1([H])[C@@]3(C)CC[C@H](O)C[C@]3([H])[C@@H](O)C[C@]12O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta,6bet..", "[H][C@]1([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@H](O)[C@@]2([H])[C@]3([H])C[C@@H](O)[C@@]4([H])C[C@@H](O)CC[C@]4(C)[C@@]3([H])CC[C@]12C", "alpha"),
    ("5alpha-Dermophol", "[H][C@]12C[C@H](O)CC[C@]1(C)[C@@]1([H])C[C@H](O)[C@@]3(C)[C@@]([H])(CC[C@]3([H])[C@H](C)CCCC(O)(CO)CO)[C@]1([H])[C@H](O)C2", "alpha"),
    ("5alpha-Pregn-2-en-20-one", "[H][C@]12CC=CC[C@]1(C)[C@@]1([H])CC[C@]3(C)[C@@H](C(C)=O)CC[C@@]3([H])[C@]1([H])CC2", "alpha"),
]


if __name__ == "__main__":
    rows: list[tuple[str, str, str]] = []
    for name, smiles_or_inchi, label in EXAMPLES:
        mol = smiles_or_inchi_to_mol(smiles_or_inchi)
        if mol is None:
            rows.append((name, label, "parse_failed"))
            continue
        parity = c5_steroid_parity(mol, _iupac_to_atom(mol))
        parity_text = "-" if parity is None else f"{parity:+d}"
        rows.append((name, label, parity_text))

    name_width = max(len(name) for name, _, _ in rows)
    label_width = max(len("expected"), max(len(label) for _, label, _ in rows))
    print(f"{'molecule':<{name_width}}  |  {'expected':<{label_width}}  |  parity")
    print(f"{'-' * name_width}--+--{'-' * label_width}--+--------")
    for name, label, parity_text in rows:
        print(f"{name:<{name_width}}  |  {label:<{label_width}}  |  {parity_text}")
