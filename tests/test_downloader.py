"""Tests for chebi_utils.downloader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from chebi_utils.downloader import (
    _chebi_obo_url,
    _chebi_sdf_url,
    download_chebi_obo,
    download_chebi_sdf,
)

# --- URL helper tests ---


def test_obo_url_modern_version():
    url = _chebi_obo_url(245)
    assert url == "https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel245/ontology/chebi.obo"


def test_obo_url_legacy_version():
    url = _chebi_obo_url(244)
    assert (
        url
        == "https://ftp.ebi.ac.uk/pub/databases/chebi/archive/chebi_legacy/archive/rel244/ontology/chebi.obo"
    )


def test_sdf_url_modern_version():
    url = _chebi_sdf_url(245)
    assert url == "https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel245/SDF/chebi.sdf.gz"


def test_sdf_url_legacy_version():
    url = _chebi_sdf_url(230)
    assert (
        url
        == "https://ftp.ebi.ac.uk/pub/databases/chebi/archive/chebi_legacy/archive/rel230/ontology/chebi.obo"
    )


# --- download_chebi_obo tests ---


def test_download_chebi_obo_calls_urlretrieve_modern(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve") as mock_retrieve:
        result = download_chebi_obo(version=250, dest_dir=tmp_path)
        mock_retrieve.assert_called_once_with(_chebi_obo_url(250), tmp_path / "chebi.obo")
        assert result == tmp_path / "chebi.obo"


def test_download_chebi_obo_calls_urlretrieve_legacy(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve") as mock_retrieve:
        result = download_chebi_obo(version=230, dest_dir=tmp_path)
        mock_retrieve.assert_called_once_with(_chebi_obo_url(230), tmp_path / "chebi.obo")
        assert result == tmp_path / "chebi.obo"


def test_download_chebi_obo_custom_filename(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        result = download_chebi_obo(version=250, dest_dir=tmp_path, filename="my_chebi.obo")
        assert result == tmp_path / "my_chebi.obo"


# --- download_chebi_sdf tests ---


def test_download_chebi_sdf_calls_urlretrieve_modern(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve") as mock_retrieve:
        result = download_chebi_sdf(version=250, dest_dir=tmp_path)
        mock_retrieve.assert_called_once_with(_chebi_sdf_url(250), tmp_path / "chebi.sdf.gz")
        assert result == tmp_path / "chebi.sdf.gz"


def test_download_chebi_sdf_calls_urlretrieve_legacy(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve") as mock_retrieve:
        result = download_chebi_sdf(version=230, dest_dir=tmp_path)
        mock_retrieve.assert_called_once_with(_chebi_sdf_url(230), tmp_path / "chebi.sdf.gz")
        assert result == tmp_path / "chebi.sdf.gz"


def test_download_chebi_sdf_custom_filename(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        result = download_chebi_sdf(version=250, dest_dir=tmp_path, filename="my_chebi.sdf.gz")
        assert result == tmp_path / "my_chebi.sdf.gz"


# --- shared behaviour tests ---


def test_download_creates_dest_dir(tmp_path):
    new_dir = tmp_path / "subdir" / "nested"
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        download_chebi_obo(version=250, dest_dir=new_dir)
    assert new_dir.is_dir()


def test_download_returns_path_object(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        result = download_chebi_obo(version=250, dest_dir=str(tmp_path))
    assert isinstance(result, Path)
