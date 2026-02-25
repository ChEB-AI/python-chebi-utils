This is a Python repository that provides basic functionalities related to ChEBI, an ontology for chemical entities. It is used by machine learning libraries and other Python applications that rely on reproducible datasets generated from ChEBI.

## Code Standards

### Required Before Each Commit
- Run `ruff format` before committing any changes to ensure proper code formatting
- This will run run on all python files to maintain consistent style
- 
## Key Guidelines
1. Follow Python best practices
2. The repository should be installable with uv and contain only a limited number of dependencies.
3. Use `fastobo` for parsing obo files and `rdkit` for parsing molecules.
4. Write unit tests for new functionality
5. Document your code. Give instructions on installation and usage in the README.
