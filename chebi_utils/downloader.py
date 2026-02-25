"""Download ChEBI data files from the EBI FTP server."""

from __future__ import annotations

import urllib.request
from pathlib import Path

_CHEBI_LEGACY_VERSION_THRESHOLD = 245


def _chebi_obo_url(version: int) -> str:
    if version < _CHEBI_LEGACY_VERSION_THRESHOLD:
        return f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/chebi_legacy/archive/rel{version}/ontology/chebi.obo"
    return f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel{version}/ontology/chebi.obo"


def _chebi_sdf_url(version: int) -> str:
    if version < _CHEBI_LEGACY_VERSION_THRESHOLD:
        return f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/chebi_legacy/archive/rel{version}/ontology/chebi.obo"
    return f"https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel{version}/SDF/chebi.sdf.gz"


def download_chebi_obo(
    version: int,
    dest_dir: str | Path = ".",
    filename: str = "chebi.obo",
) -> Path:
    """Download a versioned ChEBI OBO ontology file from the EBI FTP server.

    Parameters
    ----------
    version : int
        ChEBI release version number (e.g. 230, 245, 250).
        Versions below 245 are fetched from the legacy archive path.
    dest_dir : str or Path
        Directory where the file will be saved (created if it doesn't exist).
    filename : str
        Name for the downloaded file.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    urllib.request.urlretrieve(_chebi_obo_url(version), dest_path)
    return dest_path


def download_chebi_sdf(
    version: int,
    dest_dir: str | Path = ".",
    filename: str = "chebi.sdf.gz",
) -> Path:
    """Download a versioned ChEBI SDF file from the EBI FTP server.

    Parameters
    ----------
    version : int
        ChEBI release version number (e.g. 230, 245, 250).
        Versions below 245 are fetched from the legacy archive path.
    dest_dir : str or Path
        Directory where the file will be saved (created if it doesn't exist).
    filename : str
        Name for the downloaded file.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    urllib.request.urlretrieve(_chebi_sdf_url(version), dest_path)
    return dest_path
