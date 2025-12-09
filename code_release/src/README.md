# Project Description

This project uses **uv** to manage Python dependencies and virtual environments, the main configuration files are:

- `pyproject.toml`: project dependencies and base configuration

For more information about the installation and usage of **uv, please refer to the **uv official documentation**.

## Main program

- `main.py`: the main program of the project, which implements the **50% cross validation** and the **complete training and evaluation process using all data at once**.
- Before running the main program, please make sure you have configured your environment correctly and downloaded the rna-fm pre-training model file into the `data_release\model\rna-fm` folder.