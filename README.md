# BSc. DS - Mika Senghaas

This repository is under _active developement_.

## ⚙️ Setup

This project uses virtual environments (called `venv`) to easily replicate the Python environment originally used to produce the results. This includes a Python Version and the list of all Python packages (from PyPi) with their version requirements.

`pyenv-virtualenv` ([GitHub](https://github.com/pyenv/pyenv-virtualenv)) is used to manage both the Python version and external packages. It is a lightweight wrapper around the modern `pyenv` project ([Github](https://github.com/pyenv/pyenv)).

To setup the environment just run the bash script `setup` at the project root. It checks your path for the `pyenv` binaries and installs them from the official repository (using their automatic installation script). It then creates (if not yet present) the virtual environment and installs al dependencies. Re-running the script will not do any good or damage.

```bash
chmod +x setup && ./setup
```

To check that the environment is set up correctly, run the following list of commands.

```bash
pyenv version
```

This checks if `pyenv` is installed and the `bsc` virtual environment is installed.
The output should be `bsc (set by /User/<user>/<path>/bsc/.python-version)`.

```bash
python --version
```

The output should be `Python 3.10.X` (by default the latest stable release of Python `3.10` is installed).

Lastly, check if all package dependencies are installed.

```bash
pip list
```

It lists all packages and their versions. Make sure that you can find all packages listed in the `requirements.txt` file. Alternatively, you can run execute this inline Python expression. If the expressions runs without errors, you are ready to go!

```bash
python -c "import numpy; import torch; import transformers"
```

## Running the Project
