import warnings

from chembl_structure_pipeline.standardizer import update_mol_valences
from rdkit import Chem


def _sanitize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """Sanitize molecule, falling back to the unsanitized molecule on failure"""
    try:
        mol = update_mol_valences(mol)
        Chem.SanitizeMol(mol)
    except Exception as e:
        warnings.warn(f"Failed to sanitize molecule: {e}", stacklevel=2)
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    return mol


def parse_molblock(molblock: str, chebi_id: str | None = None) -> Chem.Mol | None:
    """Parse a V2000/V3000 molblock into an RDKit Mol object.

    Sanitize molecules with the ChEMBL structure pipeline for consistency with ChEBI.
    If sanitization fails, the unsanitized molecule is returned.

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

    return _sanitize_molecule(mol)


def smiles_or_inchi_to_mol(smiles_or_inchi: str) -> Chem.rdchem.Mol | None:
    """Parse a SMILES or InChI string into an RDKit Mol object.

    Sanitize molecules with the ChEMBL structure pipeline for consistency with ChEBI.
    If sanitization fails, the unsanitized molecule is returned.

    Parameters
    ----------
    smiles_or_inchi : str
        The SMILES or InChI string to parse.

    Returns
    -------
    Chem.Mol or None
        Parsed molecule, or ``None`` if parsing failed.
    """

    if smiles_or_inchi.startswith("InChI="):
        mol = Chem.MolFromInchi(smiles_or_inchi, sanitize=False, removeHs=False)
    else:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        params.sanitize = False
        mol = Chem.MolFromSmiles(smiles_or_inchi, params)

    if mol is None:
        warnings.warn(f"RDKit failed at parsing {smiles_or_inchi} (returned None)", stacklevel=2)
        return None

    return _sanitize_molecule(mol)
