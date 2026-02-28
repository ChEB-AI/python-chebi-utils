# python-chebi-utils

Common processing functionality for the ChEBI ontology — download versioned data files, build an ontology graph, extract molecules, assemble labeled datasets, and generate stratified train/val/test splits.

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

obo_path = download_chebi_obo(version=248, dest_dir="data/")   # downloads chebi.obo
sdf_path = download_chebi_sdf(version=248, dest_dir="data/")   # downloads chebi.sdf.gz
```

A specific ChEBI release `version` (e.g. `230`, `245`, `248`) must be provided.
Files are fetched from the [EBI FTP server](https://ftp.ebi.ac.uk/pub/databases/chebi/).
Versions below 245 are automatically fetched from the legacy archive path.

### Build the ChEBI ontology graph

```python
from chebi_utils import build_chebi_graph

graph = build_chebi_graph("chebi.obo")
# networkx.DiGraph — nodes are string ChEBI IDs (e.g. "1" for CHEBI:1)
# node attributes: name, smiles, subset
# edge attribute:  relation  ("is_a", "has_part", …)
```

Obsolete terms are excluded automatically. `xref:` lines are stripped before
parsing to work around known fastobo compatibility issues in some ChEBI releases.

To obtain only the `is_a` hierarchy as a subgraph:

```python
from chebi_utils.obo_extractor import get_hierarchy_subgraph

hierarchy = get_hierarchy_subgraph(graph)
```

### Extract molecules

```python
from chebi_utils import extract_molecules

molecules = extract_molecules("chebi.sdf.gz")
# DataFrame columns: chebi_id, name, inchi, inchikey, smiles, charge, mass, mol, …
# mol column contains RDKit Mol objects (None when parsing fails)
```

Both plain `.sdf` and gzip-compressed `.sdf.gz` files are supported.
Molecules that cannot be parsed are excluded from the returned DataFrame.

### Build a labeled dataset

```python
from chebi_utils import build_labeled_dataset

dataset, labels = build_labeled_dataset(graph, molecules, min_molecules=50)
# dataset — DataFrame with columns: chebi_id, mol, <label1>, <label2>, …
#            one boolean column per selected ontology class
# labels  — sorted list of ChEBI IDs selected as label classes
```

Each molecule is assigned to every label class that it belongs to directly or
through a chain of `is_a` relationships. Only classes with at least
`min_molecules` descendant molecules are kept as labels.

### Generate stratified train/val/test splits

```python
from chebi_utils import create_multilabel_splits

splits = create_multilabel_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
train_df = splits["train"]
val_df   = splits["val"]
test_df  = splits["test"]
```

Columns 0 and 1 (`chebi_id`, `mol`) are treated as metadata; all remaining
columns are treated as binary label columns. When multiple label columns are
present, `MultilabelStratifiedShuffleSplit` from the
`iterative-stratification` package is used; for a single label column,
`StratifiedShuffleSplit` from scikit-learn is used.

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
