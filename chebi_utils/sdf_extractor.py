"""Extract molecule data from ChEBI SDF files."""

from __future__ import annotations

import gzip
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem


def _update_mol_valences(mol: Chem.Mol) -> Chem.Mol:
    """Mark all atoms as having no implicit hydrogens to preserve molfile valences."""
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    return mol


def _parse_molblock(molblock: str, chebi_id: str | None = None) -> Chem.Mol | None:
    """Parse a V2000/V3000 molblock into an RDKit Mol object.

    Uses partial sanitisation to handle ChEBI molecules with unusual valences
    or radicals.

    Parameters
    ----------
    molblock : str
        The molblock string (header + atom/bond table + ``M  END``).
    chebi_id : str or None
        Used only for the warning message when parsing fails.

    Returns
    -------
    Chem.Mol or None
        Parsed molecule, or ``None`` if parsing failed.
    """
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if mol is None:
        warnings.warn(f"Failed to parse molblock for {chebi_id}", stacklevel=2)
        return None
    mol = _update_mol_valences(mol)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=(
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        ),
        catchErrors=True,
    )
    return mol


def _iter_sdf_records(filepath: str | Path):
    """Yield individual SDF records as strings."""
    opener = gzip.open if str(filepath).endswith(".gz") else open
    current_record: list[str] = []

    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            current_record.append(line)
            if line.strip() == "$$$$":
                yield "".join(current_record)
                current_record = []


def _parse_sdf_record(record: str) -> tuple[dict[str, str], str]:
    """Parse a single SDF record.

    Returns
    -------
    tuple[dict[str, str], str]
        ``(props, molblock)`` where *props* is a dict of data-item key/values
        and *molblock* is the raw connection-table string.
    """
    props: dict[str, str] = {}
    lines = record.splitlines(keepends=True)

    # Collect molblock: everything up to (but not including) the first "> <" tag
    molblock_lines: list[str] = []
    data_start = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("> <") or stripped == "$$$$":
            data_start = idx
            break
        molblock_lines.append(line)
    molblock = "".join(molblock_lines)

    # Extract header name (first line of molblock)
    if molblock_lines:
        props["mol_name"] = molblock_lines[0].strip()

    # Parse data items
    i = data_start
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("> <") and line.endswith(">"):
            key = line[3:-1]
            value_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() not in ("", "$$$$"):
                value_lines.append(lines[i].strip())
                i += 1
            props[key] = "\n".join(value_lines)
        else:
            i += 1

    return props, molblock


def extract_molecules(filepath: str | Path) -> pd.DataFrame:
    """Extract molecule data from a ChEBI SDF file.

    Supports both plain (``.sdf``) and gzip-compressed (``.sdf.gz``) files.
    Each molecule is parsed into an RDKit ``Mol`` object stored in the ``mol``
    column.  Molecules that cannot be parsed result in ``None`` in that column.

    Parameters
    ----------
    filepath : str or Path
        Path to the ChEBI SDF (or SDF.gz) file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per molecule. Columns depend on the properties
        present in the file. Common columns (renamed for convenience):
        chebi_id, name, inchi, inchikey, smiles, formula, charge, mass, mol.
    """
    rows = []
    molblocks = []
    for record in _iter_sdf_records(filepath):
        props, molblock = _parse_sdf_record(record)
        rows.append(props)
        molblocks.append(molblock)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    rename_map = {
        "ChEBI ID": "chebi_id",
        "ChEBI Name": "name",
        "InChI": "inchi",
        "InChIKey": "inchikey",
        "SMILES": "smiles",
        "Formulae": "formula",
        "Charge": "charge",
        "Mass": "mass",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    chebi_ids = df["chebi_id"].tolist() if "chebi_id" in df.columns else [None] * len(df)
    df["mol"] = [_parse_molblock(mb, cid) for mb, cid in zip(molblocks, chebi_ids, strict=False)]

    return df
