"""Download ChEBI data files from the EBI FTP server."""

from __future__ import annotations

import urllib.request
from pathlib import Path

CHEBI_OBO_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo"
CHEBI_SDF_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/ChEBI_complete.sdf.gz"


def download_chebi_obo(dest_dir: str | Path = ".", filename: str = "chebi.obo") -> Path:
    """Download the ChEBI OBO ontology file from the EBI FTP server.

    Parameters
    ----------
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
    urllib.request.urlretrieve(CHEBI_OBO_URL, dest_path)
    return dest_path


def download_chebi_sdf(dest_dir: str | Path = ".", filename: str = "chebi.sdf.gz") -> Path:
    """Download the ChEBI SDF file from the EBI FTP server.

    Parameters
    ----------
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
    urllib.request.urlretrieve(CHEBI_SDF_URL, dest_path)
    return dest_path
