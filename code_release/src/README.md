# Project Description

This project uses **uv** to manage Python dependencies and virtual environments, the main configuration files are:

- `pyproject.toml`: project dependencies and base configuration

For more information about the installation and usage of **uv, please refer to the **uv official documentation**.

## Main program

- `main.py`: runs **five-fold cross-validation**, followed by final training and evaluation on a held-out test set.
- From this directory, run `uv sync --locked`. Follow the [RNA-FM installation instructions](../../data_release/model/README.md) to convert the original checkpoint into the model/tokenizer directory at `data_release/model/rna-fm`.
- Run `uv run check_rnafm.py` before `uv run main.py`. Original `.pth` files cannot be used directly by the current MultiMolecule loading calls.
