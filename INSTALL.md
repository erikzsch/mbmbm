# Installation for development

### Clone this repository to a location on your computer
```
git clone https://git.io-warnemuende.de/sperlea/microbiome_ML_benchmark.git
```
or 
```
git@git.io-warnemuende.de:sperlea/microbiome_ML_benchmark.git
```

### Install poetry
Full instructions can be found on [https://python-poetry.org/docs/](https://python-poetry.org/docs/).

Short version (Unix/Win-WSL):
```
curl -sSL https://install.python-poetry.org | python3 -
```

Short version (Win-Powershell):
```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Check for success: `poetry --version`.

### Install project
Run `poetry install` in the folder with the `pyproject.toml`.
This will read the `pyproject.toml` file from the current project, resolves the dependencies, 
and installs them to the (possibly newly created) virtual environment in either your project folder 
or a global directory. 

In your IDE use `venv_path/bin/python`.

To activate/deactivate the virtual environment from command line use `poetry shell` and `deactivate`.

### Install dependencies
Instead of `pip install ...` or `conda install ...` use `poetry add ...`.  
Use `poetry add ... -G dev` for adding a development dependency.  
This will check if there are any conflicts with other dependencies 
and (on success) add the dependency to the `pyproject.toml` as well as installing it.

See [here](https://python-poetry.org/docs/cli/#add) for further details, 
e.g. adding repositories or local sources. 