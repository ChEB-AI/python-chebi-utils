# extract basic (and not so basic) properties from molecules. This is used to construct
# FOL structures for reasoning tasks on molecules.

import logging

from rdkit import Chem
from rdkit.Chem import Descriptors

MAX_RING_SIZE = 8

# Gonane core SMARTS (C1–C17, IUPAC steroid numbering).
# [#6] matches any carbon; ~ matches any bond — handles unsaturated steroids
# (Δ4, Δ5), ketones, and estrogens (aromatic ring A) without modification.
# C18/C19 methyls and side chains are intentionally excluded so the pattern
# matches all steroid sub-classes, not just fully-saturated ones.
_GONANE_PATTERN = Chem.MolFromSmarts(
    "[#6:13]12~[#6:12]~[#6:11]~[#6:9]3~[#6:10]4~"
    "[#6:1]~[#6:2]~[#6:3]~[#6:4]~[#6:5]4~"
    "[#6:6]~[#6:7]~[#6:8]3~[#6:14]2~"
    "[#6:15]~[#6:16]~[#6:17]1"
)
_GONANE_IDX_TO_IUPAC: dict[int, int] = {
    atom.GetIdx(): atom.GetAtomMapNum()
    for atom in _GONANE_PATTERN.GetAtoms()
    if atom.GetAtomMapNum() > 0
}


def mol_to_fol_atoms(
    mol: Chem.Mol, with_rings=True, with_steroids=True
) -> tuple[dict[str, list], set[str]]:
    """Convert an RDKit ``Mol`` into a first-order logic model at the atom level.

    Returns ``(atom_extensions, mol_extensions)`` where:
    - ``atom_extensions`` is a ``dict[str, list]``: unary predicates map to
      ``list[int]`` of atom indices; binary predicates map to
      ``list[tuple[int, int]]`` of (left, right) index pairs.
    - ``mol_extensions`` is a ``set[str]`` of molecule-level predicate names
      that hold for this molecule (e.g. ``net_charge_positive``).
    """
    atom_extensions: dict[str, list] = {}

    atom_extensions.update(get_atom_properties(mol))
    atom_extensions.update(get_bond_properties(mol))

    if with_rings:
        atom_extensions.update(get_rings(mol))

    if with_steroids:
        atom_extensions.update(get_steroid_positions(mol))

    mol_extensions = get_molecule_level_properties(mol)

    return atom_extensions, mol_extensions


def get_atom_properties(mol: Chem.Mol) -> dict[str, list]:
    try:
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
    except Exception as e:
        logging.error(
            "Failed to assign CIP labels to molecule, skipping chirality-related extensions: %s",
            e,
        )

    atom_extensions: dict[str, list] = {}

    # For each atom: element symbol, charge, hydrogen counts, chirality
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol().lower()
        atom_extensions.setdefault(atom_symbol, []).append(atom_idx)

        charge = atom.GetFormalCharge()
        if charge != 0:
            for pred in [
                f"charge_{'n' if charge < 0 else 'p'}",
                f"charge{'_m' + str(-charge) if charge < 0 else str(charge)}",
            ]:
                atom_extensions.setdefault(pred, []).append(atom_idx)
        else:
            atom_extensions.setdefault("charge0", []).append(atom_idx)

        num_hs = atom.GetTotalNumHs(includeNeighbors=True)
        for pred in [f"has_{num_hs}_hs"] + [f"has_at_least_{n}_hs" for n in range(1, num_hs + 1)]:
            atom_extensions.setdefault(pred, []).append(atom_idx)

        if atom.HasProp("_CIPCode"):
            chiral_code = f"cip_code_{atom.GetProp('_CIPCode')}"
            atom_extensions.setdefault(chiral_code, []).append(atom_idx)

    return atom_extensions


def get_bond_properties(mol: Chem.Mol) -> dict[str, list]:
    # Bond predicates (symmetric)
    atom_extensions: dict[str, list] = {}
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()

        bond_pred = f"b{bond.GetBondType()}"
        atom_extensions.setdefault(bond_pred, []).extend([(left, right), (right, left)])
        atom_extensions.setdefault("has_bond_to", []).extend([(left, right), (right, left)])

        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            stereo_pred = f"b{bond.GetStereo().name}"
            atom_extensions.setdefault(stereo_pred, []).extend([(left, right), (right, left)])
    return atom_extensions


def get_molecule_level_properties(mol: Chem.Mol) -> set[str]:
    # Molecule-level (global) properties (either true or false for the whole molecule)
    mol_extensions: set[str] = set()
    net_charge = Chem.GetFormalCharge(mol)
    if net_charge > 0:
        mol_extensions.add("net_charge_positive")
    elif net_charge < 0:
        mol_extensions.add("net_charge_negative")
    else:
        mol_extensions.add("net_charge_neutral")
    # aliphatic vs aromatic (defined as having at least one aromatic atom)
    if len(list(mol.GetAromaticAtoms())) > 0:
        mol_extensions.add("aromatic")
    else:
        mol_extensions.add("aliphatic")
    return mol_extensions


def get_rings(mol: Chem.Mol) -> dict[str, list]:
    # Rings have two predicates. One for the atom-ring relation and one for the ring itself
    #   ring{N}(A1, …, AN) – A1…AN form an N-membered ring (all permutations)
    #                        Only for N <= MAX_RING_SIZE.
    #   in_ring{N}(A)      – A belongs to some N-membered ring (N <= MAX_RING_SIZE).
    #   in_ring(A)         – A belongs to some ring of any size.
    atom_extensions: dict[str, list] = {}
    in_ring_atoms: set[int] = set()
    in_ringN_atoms: dict[int, set[int]] = {}
    for ring in mol.GetRingInfo().AtomRings():
        n = len(ring)
        in_ring_atoms.update(ring)
        if n <= MAX_RING_SIZE:
            for start_atom in range(n):
                ring_permutation = ring[start_atom:] + ring[:start_atom]
                atom_extensions.setdefault(f"ring{n}", []).append(tuple(ring_permutation))
                atom_extensions[f"ring{n}"].append(tuple(reversed(ring_permutation)))
            in_ringN_atoms.setdefault(n, set()).update(ring)
    if in_ring_atoms:
        atom_extensions["in_ring"] = sorted(in_ring_atoms)
    for n, atoms in in_ringN_atoms.items():
        atom_extensions[f"in_ring{n}"] = sorted(atoms)
    return atom_extensions


def get_steroid_positions(mol: Chem.Mol) -> dict[str, list]:
    # Steroid nucleus positions (steroid_1 … steroid_17)
    atom_extensions: dict[str, list] = {}
    steroid_match = mol.GetSubstructMatch(_GONANE_PATTERN, useChirality=False)
    if steroid_match:
        for pat_idx, atom_idx in enumerate(steroid_match):
            iupac = _GONANE_IDX_TO_IUPAC.get(pat_idx)
            if iupac is not None:
                atom_extensions.setdefault(f"steroid_{iupac}", []).append(atom_idx)
    return atom_extensions


def get_numerical_facts(mol: Chem.Mol) -> dict[str, list]:
    """Molecular weight and ring size as numerical values.

    Expresses "this molecule has weight ..." and "this molecule has a ring of size ..."
    as molecule-integer value relations.
    """
    atom_extensions: dict[str, list] = {}
    atom_extensions["mol_weight"] = [round(Descriptors.MolWt(mol))]
    for ring in mol.GetRingInfo().AtomRings():
        atom_extensions.setdefault("ring_size", []).append(len(ring))
    return atom_extensions
