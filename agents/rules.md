# Project Rules & Guidelines

## Commit Rules (Conventional Commits)

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files

**Format**: `<type>(<scope>): <description>`

Example: `feat(api): add endpoint for user registration`

---

## Check Rules

Every contribution must pass the following checks:

### 1. Code Formatting (Black)
- **Tool**: `black` with the Jupyter extra (`black[jupyter]`), installed through the project dev extra.
- **Rule**: Python files and committed notebook code cells must be formatted with a line length of 88 characters.
- **Command**: `black .`
- **Config**: `pyproject.toml` must include both Python and notebook files in `[tool.black].include`.

### 2. Linting (Ruff)
- **Tool**: `ruff`
- **Rule**: No linting errors should be present.
- **Rules checked**: Pyflakes, pycodestyle, isort, etc. (see `pyproject.toml`)
- **Command**: `ruff check . --fix`

### 3. Type Checking (Mypy)
- **Tool**: `mypy`
- **Rule**: All public functions must have type hints. Strict mode is partially enabled.
- **Command**: `mypy .`

### 4. Unit Testing (Pytest)
- **Tool**: `pytest`
- **Rule**: All new features must have accompanying tests.
- **Command**: `pytest`

### 5. Documentation
- **Rule**: Any logic change must be reflected in the documentation if applicable.
- **Location**: `docs/` and docstrings in code.
