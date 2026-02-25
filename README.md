# python-chebi-utils

Common processing functionality for the ChEBI ontology — download data files, extract classes and relations, extract molecules, and generate stratified train/val/test splits.

## Installation

```bash
pip install chebi-utils
```

For development (includes `pytest` and `ruff`):

```bash
pip install -e ".[dev]"
```

## Features

### Download ChEBI data files

```python
from chebi_utils import download_chebi_obo, download_chebi_sdf

obo_path = download_chebi_obo(dest_dir="data/")   # downloads chebi.obo
sdf_path = download_chebi_sdf(dest_dir="data/")   # downloads chebi.sdf.gz
```

Files are fetched from the [EBI FTP server](https://ftp.ebi.ac.uk/pub/databases/chebi/).

### Extract ontology classes and relations

```python
from chebi_utils import extract_classes, extract_relations

classes = extract_classes("chebi.obo")
# DataFrame: id, name, definition, is_obsolete

relations = extract_relations("chebi.obo")
# DataFrame: source_id, target_id, relation_type  (is_a, has_role, …)
```

### Extract molecules

```python
from chebi_utils import extract_molecules

molecules = extract_molecules("chebi.sdf.gz")
# DataFrame: chebi_id, name, smiles, inchi, inchikey, formula, charge, mass, …
```

Both plain `.sdf` and gzip-compressed `.sdf.gz` files are supported.

### Generate train/val/test splits

```python
from chebi_utils import create_splits

splits = create_splits(molecules, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
train_df = splits["train"]
val_df   = splits["val"]
test_df  = splits["test"]
```

Pass `stratify_col` to preserve class proportions across splits:

```python
splits = create_splits(classes, stratify_col="is_obsolete", seed=42)
```

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check .
ruff format --check .
```

## CI/CD

A GitHub Actions workflow (`.github/workflows/ci.yml`) automatically runs ruff linting and the full test suite on every push and pull request across Python 3.10, 3.11, and 3.12.
