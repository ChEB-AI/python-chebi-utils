"""Extract classes and relations from ChEBI OBO ontology files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parse_obo_stanzas(filepath: str | Path) -> list[dict[str, list[str]]]:
    """Parse an OBO file and return a list of stanza dicts."""
    stanzas: list[dict[str, list[str]]] = []
    current_stanza: dict[str, list[str]] | None = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("["):
                if current_stanza is not None:
                    stanzas.append(current_stanza)
                stanza_type = line.strip("[]")
                current_stanza = {"_type": [stanza_type]}
            elif current_stanza is not None and ":" in line:
                key, _, value = line.partition(":")
                current_stanza.setdefault(key.strip(), []).append(value.strip())

    if current_stanza is not None:
        stanzas.append(current_stanza)

    return stanzas


def extract_classes(filepath: str | Path) -> pd.DataFrame:
    """Extract ontology classes (terms) from a ChEBI OBO file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ChEBI OBO file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, name, definition, is_obsolete.
    """
    stanzas = _parse_obo_stanzas(filepath)
    rows = []
    for stanza in stanzas:
        if stanza.get("_type", [None])[0] != "Term":
            continue
        row = {
            "id": stanza.get("id", [None])[0],
            "name": stanza.get("name", [None])[0],
            "definition": stanza.get("def", [None])[0],
            "is_obsolete": stanza.get("is_obsolete", ["false"])[0] == "true",
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=["id", "name", "definition", "is_obsolete"])


def extract_relations(filepath: str | Path) -> pd.DataFrame:
    """Extract class relations from a ChEBI OBO file.

    Parameters
    ----------
    filepath : str or Path
        Path to the ChEBI OBO file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: source_id, target_id, relation_type.
    """
    stanzas = _parse_obo_stanzas(filepath)
    rows = []

    for stanza in stanzas:
        if stanza.get("_type", [None])[0] != "Term":
            continue
        source_id = stanza.get("id", [None])[0]
        if source_id is None:
            continue

        for is_a_val in stanza.get("is_a", []):
            target_id = is_a_val.split("!")[0].strip()
            rows.append({"source_id": source_id, "target_id": target_id, "relation_type": "is_a"})

        for rel_val in stanza.get("relationship", []):
            parts = rel_val.split()
            if len(parts) >= 2:
                rel_type = parts[0]
                target_id = parts[1].split("!")[0].strip()
                rows.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": rel_type,
                    }
                )

    return pd.DataFrame(rows, columns=["source_id", "target_id", "relation_type"])
