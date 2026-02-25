"""Extract molecule data from ChEBI SDF files."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd


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


def _parse_sdf_record(record: str) -> dict[str, str]:
    """Parse a single SDF record into a dict of data-item properties."""
    props: dict[str, str] = {}
    lines = record.splitlines()

    if lines:
        props["mol_name"] = lines[0].strip()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("> <") and line.rstrip().endswith(">"):
            key = line.strip()[3:-1]
            value_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() not in ("", "$$$$"):
                value_lines.append(lines[i].strip())
                i += 1
            props[key] = "\n".join(value_lines)
        else:
            i += 1

    return props


def extract_molecules(filepath: str | Path) -> pd.DataFrame:
    """Extract molecule data from a ChEBI SDF file.

    Supports both plain (``.sdf``) and gzip-compressed (``.sdf.gz``) files.

    Parameters
    ----------
    filepath : str or Path
        Path to the ChEBI SDF (or SDF.gz) file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per molecule. Columns depend on the properties
        present in the file. Common columns (renamed for convenience):
        chebi_id, name, inchi, inchikey, smiles, formula, charge, mass.
    """
    records = [_parse_sdf_record(r) for r in _iter_sdf_records(filepath)]

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

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

    return df
