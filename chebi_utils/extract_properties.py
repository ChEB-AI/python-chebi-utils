"""Extract basic (and not so basic) properties from molecules.

These are used to construct first-order-logic (FOL) structures for reasoning
tasks on molecules.
"""

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

    Aggregates atom, bond, ring, and steroid predicates into a single atom-level
    extension mapping plus the set of molecule-level predicates.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to convert.
    with_rings : bool
        If ``True`` (default), include ring predicates from :func:`get_rings`.
    with_steroids : bool
        If ``True`` (default), include steroid-nucleus position predicates from
        :func:`get_steroid_positions`.

    Returns
    -------
    tuple[dict[str, list], set[str]]
        ``(atom_extensions, mol_extensions)`` where:
        - ``atom_extensions`` maps each predicate name to its extension: unary predicates map
          to ``list[int]`` of atom indices; relation predicates map to ``list[tuple[int, ...]]``
          of atom-index tuples (arity depends on the predicate).
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
    """Extract per-atom unary predicates.

    For each atom, emits predicates for its element symbol, formal charge (both a sign
    predicate ``charge_p``/``charge_n``/``charge0`` and an exact predicate
    ``charge{n}`` for positive charges or ``charge_m{n}`` for negative charges), total
    hydrogen count (exact ``has_{n}_hs`` and cumulative ``has_at_least_{n}_hs``), and
    CIP chirality label (``cip_code_R``/``cip_code_S``) when assigned.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to inspect.

    Returns
    -------
    dict[str, list]
        Mapping from predicate name to the ``list[int]`` of atom indices for
        which the predicate holds.
    """
    try:
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
    except Exception as e:
        logging.error(
            "Failed to assign CIP labels to molecule, skipping chirality-related extensions: %s",
            e,
        )

    atom_extensions: dict[str, list] = {}

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
    """Extract symmetric per-bond binary predicates.

    For each bond, emits a bond-type predicate (e.g. ``bSINGLE``), a generic
    ``has_bond_to`` predicate, and a bond-stereo predicate (e.g.
    ``bSTEREOE``) when the bond carries stereochemistry. Every predicate is
    stored symmetrically, i.e. both ``(left, right)`` and ``(right, left)``.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to inspect.

    Returns
    -------
    dict[str, list]
        Mapping from predicate name to a ``list[tuple[int, int]]`` of ordered
        atom-index pairs for which the predicate holds.
    """
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
    """Extract global (molecule-level) predicates.

    Emits exactly one net-charge predicate
    (``net_charge_positive``/``net_charge_negative``/``net_charge_neutral``) and
    one aromaticity predicate (``aromatic`` if the molecule has at least one
    aromatic atom, otherwise ``aliphatic``).

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to inspect.

    Returns
    -------
    set[str]
        The set of molecule-level predicate names that hold for this molecule.
    """
    mol_extensions: set[str] = set()
    net_charge = Chem.GetFormalCharge(mol)
    if net_charge > 0:
        mol_extensions.add("net_charge_positive")
    elif net_charge < 0:
        mol_extensions.add("net_charge_negative")
    else:
        mol_extensions.add("net_charge_neutral")
    if len(list(mol.GetAromaticAtoms())) > 0:
        mol_extensions.add("aromatic")
    else:
        mol_extensions.add("aliphatic")
    return mol_extensions


def get_rings(mol: Chem.Mol) -> dict[str, list]:
    """Extract ring-membership predicates.

    Emits three kinds of predicate:

    - ``ring{N}(A1, …, AN)`` – ``A1…AN`` form an N-membered ring, stored for all
      rotations in both directions. Only for ``N <= MAX_RING_SIZE``.
    - ``in_ring{N}(A)`` – ``A`` belongs to some N-membered ring
      (``N <= MAX_RING_SIZE``).
    - ``in_ring(A)`` – ``A`` belongs to some ring of any size.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to inspect.

    Returns
    -------
    dict[str, list]
        Mapping from predicate name to its extension: ``ring{N}`` maps to a
        ``list[tuple[int, ...]]`` of atom-index tuples; ``in_ring`` and
        ``in_ring{N}`` map to a sorted ``list[int]`` of atom indices.
    """
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


def _add_angular_methyls(
    mol: Chem.Mol,
    atom_extensions: dict[str, list],
    iupac_to_atom: dict[int, int],
) -> None:
    """Label angular methyls C18 (on C13) and C19 (on C10) when present.

    Looks for the unique carbon neighbour of the attachment atom that is not
    part of the gonane core (C1–C17). Estrogens without C19 simply yield no
    ``steroid_19`` predicate.
    """
    core_atom_indices = set(iupac_to_atom.values())
    for attachment_position, methyl_position in ((13, 18), (10, 19)):
        attachment_atom_idx = iupac_to_atom.get(attachment_position)
        if attachment_atom_idx is None:
            continue
        methyl_candidates = [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(attachment_atom_idx).GetNeighbors()
            if neighbor.GetAtomicNum() == 6 and neighbor.GetIdx() not in core_atom_indices
        ]
        if len(methyl_candidates) == 1:
            atom_extensions.setdefault(f"steroid_{methyl_position}", []).append(
                methyl_candidates[0]
            )



# --- NEW: C17 side chain numbering (C20, C21, …) ---


def _carbon_neighbor_indices(mol: Chem.Mol, atom_idx: int, excluded: set[int]) -> list[int]:
    return [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if neighbor.GetAtomicNum() == 6 and neighbor.GetIdx() not in excluded
    ]


def _reachable_carbon_indices(mol: Chem.Mol, start_idx: int, blocked: set[int]) -> set[int]:
    """Carbon atoms reachable from ``start_idx`` without entering ``blocked``."""
    reachable = set()
    stack = [start_idx]
    while stack:
        atom_idx = stack.pop()
        if atom_idx in reachable or atom_idx in blocked:
            continue
        reachable.add(atom_idx)
        stack.extend(_carbon_neighbor_indices(mol, atom_idx, blocked | reachable))
    return reachable


def _longest_carbon_path(
    mol: Chem.Mol, start_idx: int, allowed: set[int]
) -> list[int]:
    """Longest simple carbon path starting at ``start_idx`` within ``allowed``."""

    def dfs(atom_idx: int, visited: set[int]) -> list[int]:
        best_tail: list[int] = []
        for neighbor_idx in _carbon_neighbor_indices(mol, atom_idx, visited):
            if neighbor_idx not in allowed:
                continue
            tail = dfs(neighbor_idx, visited | {neighbor_idx})
            if len(tail) > len(best_tail):
                best_tail = tail
        return [atom_idx] + best_tail

    return dfs(start_idx, {start_idx})


def _label_steroid_atom(
    atom_extensions: dict[str, list],
    iupac_to_atom: dict[int, int],
    position: int,
    atom_idx: int,
) -> None:
    atom_extensions.setdefault(f"steroid_{position}", []).append(atom_idx)
    iupac_to_atom[position] = atom_idx


def _add_side_chain(
    mol: Chem.Mol,
    atom_extensions: dict[str, list],
    iupac_to_atom: dict[int, int],
) -> None:
    """Label the C17 side chain with IUPAC numbers C20, C21, …

    C20 is the unique non-core carbon neighbour of C17. From there the longest
    carbon path is treated as the main chain; a methyl branch on C20 (if any)
    is C21, then the main chain continues as C22, C23, … Terminal methyls on
    the last main-chain carbon (e.g. C26/C27 of cholestane) get the next
    numbers. Shorter chains (pregnane: C20–C21 only) and missing chains
    (estrone) are handled naturally.
    """
    c17_atom_idx = iupac_to_atom.get(17)
    if c17_atom_idx is None:
        return

    # Core C1–C17 plus any already labeled atoms (C18/C19, …).
    labeled_atom_indices = set(iupac_to_atom.values())
    for atoms in atom_extensions.values():
        labeled_atom_indices.update(atoms)

    c20_candidates = _carbon_neighbor_indices(mol, c17_atom_idx, labeled_atom_indices)
    if len(c20_candidates) != 1:
        return

    c20_atom_idx = c20_candidates[0]
    _label_steroid_atom(atom_extensions, iupac_to_atom, 20, c20_atom_idx)

    side_chain_atoms = _reachable_carbon_indices(mol, c20_atom_idx, labeled_atom_indices)
    side_chain_beyond_c20 = side_chain_atoms - {c20_atom_idx}
    next_position = 21
    c20_side_neighbors = _carbon_neighbor_indices(
        mol, c20_atom_idx, labeled_atom_indices | {c17_atom_idx}
    )

    if not c20_side_neighbors:
        return

    if len(c20_side_neighbors) == 1:
        main_chain = _longest_carbon_path(
            mol, c20_side_neighbors[0], side_chain_beyond_c20
        )
    else:
        neighbor_paths = {
            neighbor_idx: _longest_carbon_path(mol, neighbor_idx, side_chain_beyond_c20)
            for neighbor_idx in c20_side_neighbors
        }
        main_chain_start = max(
            neighbor_paths, key=lambda neighbor_idx: len(neighbor_paths[neighbor_idx])
        )
        main_chain = neighbor_paths[main_chain_start]
        for branch_idx in sorted(
            neighbor_idx
            for neighbor_idx in c20_side_neighbors
            if neighbor_idx != main_chain_start
        ):
            _label_steroid_atom(atom_extensions, iupac_to_atom, next_position, branch_idx)
            next_position += 1

    # Cholestane terminus: longest path may absorb one of C26/C27; stop at C25.
    while len(main_chain) >= 2:
        last_atom_idx = main_chain[-1]
        previous_atom_idx = main_chain[-2]
        leaf_methyls_on_previous = [
            neighbor_idx
            for neighbor_idx in _carbon_neighbor_indices(
                mol, previous_atom_idx, {c20_atom_idx, c17_atom_idx}
            )
            if neighbor_idx in side_chain_atoms
            and neighbor_idx not in main_chain[:-1]
            and len(_carbon_neighbor_indices(mol, neighbor_idx, set())) == 1
        ]
        if last_atom_idx in leaf_methyls_on_previous and len(leaf_methyls_on_previous) >= 2:
            main_chain = main_chain[:-1]
        else:
            break

    for atom_idx in main_chain:
        _label_steroid_atom(atom_extensions, iupac_to_atom, next_position, atom_idx)
        next_position += 1

    if not main_chain:
        return

    labeled_now = set(iupac_to_atom.values())
    for atoms in atom_extensions.values():
        labeled_now.update(atoms)
    terminal_methyls = sorted(
        neighbor_idx
        for neighbor_idx in _carbon_neighbor_indices(mol, main_chain[-1], labeled_now)
        if neighbor_idx in side_chain_atoms
    )
    for atom_idx in terminal_methyls:
        _label_steroid_atom(atom_extensions, iupac_to_atom, next_position, atom_idx)
        next_position += 1


def get_steroid_positions(mol: Chem.Mol) -> dict[str, list]:
    """Extract steroid-nucleus position predicates.

    Matches the molecule against the gonane core and, on a match, labels the
    ring atoms with their IUPAC steroid position as predicates ``steroid_1`` …
    ``steroid_17``. When present, angular methyls are added as ``steroid_18``
    (on C13) and ``steroid_19`` (on C10), and the C17 side chain as
    ``steroid_20``, ``steroid_21``, … (cholestane-style). Molecules without a
    gonane core yield no predicates.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to inspect.

    Returns
    -------
    dict[str, list]
        Mapping from ``steroid_{position}`` to the ``list[int]`` of matched atom
        indices. Empty when the molecule has no steroid nucleus.
    """
    atom_extensions: dict[str, list] = {}
    steroid_match = mol.GetSubstructMatch(_GONANE_PATTERN, useChirality=False)
    if not steroid_match:
        return atom_extensions

    iupac_to_atom: dict[int, int] = {}
    for pat_idx, atom_idx in enumerate(steroid_match):
        iupac = _GONANE_IDX_TO_IUPAC.get(pat_idx)
        if iupac is not None:
            atom_extensions.setdefault(f"steroid_{iupac}", []).append(atom_idx)
            iupac_to_atom[iupac] = atom_idx

    _add_angular_methyls(mol, atom_extensions, iupac_to_atom)
    _add_side_chain(mol, atom_extensions, iupac_to_atom)  # NEW: C20+
    return atom_extensions


def get_numerical_facts(mol: Chem.Mol) -> dict[str, list]:
    """Molecular weight and ring size as numerical values.

    Expresses "this molecule has weight ..." and "this molecule has a ring of
    size ..." as molecule-integer value relations.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to inspect.

    Returns
    -------
    dict[str, list]
        ``{"mol_weight": [<rounded MolWt>], "ring_size": [<size per ring>, …]}``.
        The ``ring_size`` key is present only when the molecule has at least one
        ring.
    """
    atom_extensions: dict[str, list] = {}
    atom_extensions["mol_weight"] = [round(Descriptors.MolWt(mol))]
    for ring in mol.GetRingInfo().AtomRings():
        atom_extensions.setdefault("ring_size", []).append(len(ring))
    return atom_extensions


"""Manual check for steroid numbering (not part of the library API).

Run: python -m chebi_utils.extract_properties
Expect: cholesterol -> steroid_1..27, estrone -> steroid_1..18,
pregnenolone -> steroid_1..21 (with 18/19), benzene -> []
"""
if __name__ == "__main__":
    from chebi_utils.read_molecule import smiles_or_inchi_to_mol

    for name, smiles in {
        "cholesterol": (
            "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
        ),
        "estrone": "C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)CCc4c3ccc(O)c4",
        "pregnenolone": (
            "CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
        ),
        "benzene": "c1ccccc1",
    }.items():
        keys = sorted(
            get_steroid_positions(smiles_or_inchi_to_mol(smiles)),
            key=lambda k: int(k.split("_")[1]),
        )
        print(name, keys)
