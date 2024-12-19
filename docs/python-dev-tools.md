# Python Getting Started

## Managing environments with pyenv and pyenv-virtualenv
- List Phyton version available to install
	`pyenv install --list | grep " 3\."`
- **Install Python version**
	**`pyenv install -v <version>`**
-  Update pyenv
	`pyenv update`
- Remove Pythonn version
	`pyenv uninstall <version>`
- Display Python versions available
	`pyenv versions` > `*` indicates the currently active pythone version
- **Create Python virtual environment**
	**`pyenv virtualenv <python version> <venv-project-name>`** > 
	**`echo "<venv-project-name>" >> .python-version`** > Venv now gets autoactivated in combination with entry in .zshrc when the folder is entered.
- List virtual environments
	`pyenv virtualenvs`
- Delete virtual environment
	`pyenv uninstall <venv-project-name>`

## Managing packages and dependencies with poetry
+ Init pyproject.toml
	`poetry init`
+ Activate Virtualenv first if you do not want poetry to create its own when installing dependencies. Creates Lock file if it does not exist
	`poetry install [--with <optional-group>] [--sync]`
+ Update lock file, get latest version of dependencies
	`poetry update`
+ Add and remove commands to manage dependecies from the CLI
+ Build source and wheel archives (only pure python wheels are supported)
+ Publish to PyPI 
+ There is some support for pre-commit hooks