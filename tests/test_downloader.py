"""Tests for chebi_utils.downloader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from chebi_utils.downloader import (
    CHEBI_OBO_URL,
    CHEBI_SDF_URL,
    download_chebi_obo,
    download_chebi_sdf,
)


def test_download_chebi_obo_calls_urlretrieve(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve") as mock_retrieve:
        result = download_chebi_obo(dest_dir=tmp_path)
        mock_retrieve.assert_called_once_with(CHEBI_OBO_URL, tmp_path / "chebi.obo")
        assert result == tmp_path / "chebi.obo"


def test_download_chebi_sdf_calls_urlretrieve(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve") as mock_retrieve:
        result = download_chebi_sdf(dest_dir=tmp_path)
        mock_retrieve.assert_called_once_with(CHEBI_SDF_URL, tmp_path / "chebi.sdf.gz")
        assert result == tmp_path / "chebi.sdf.gz"


def test_download_chebi_obo_custom_filename(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        result = download_chebi_obo(dest_dir=tmp_path, filename="my_chebi.obo")
        assert result == tmp_path / "my_chebi.obo"


def test_download_chebi_sdf_custom_filename(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        result = download_chebi_sdf(dest_dir=tmp_path, filename="my_chebi.sdf.gz")
        assert result == tmp_path / "my_chebi.sdf.gz"


def test_download_creates_dest_dir(tmp_path):
    new_dir = tmp_path / "subdir" / "nested"
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        download_chebi_obo(dest_dir=new_dir)
    assert new_dir.is_dir()


def test_download_returns_path_object(tmp_path):
    with patch("chebi_utils.downloader.urllib.request.urlretrieve"):
        result = download_chebi_obo(dest_dir=str(tmp_path))
    assert isinstance(result, Path)
