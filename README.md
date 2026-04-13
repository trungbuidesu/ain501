# ain501

Welcome to the AIN501 project. This project is structured to support clean, modular development for AI and agentic applications.

## Project Structure

- `src/`: Core source code.
- `models/`: Model definitions, architectures, and pre-trained weights.
- `configs/`: Configuration files (YAML, JSON).
- `data/`: Local data directory (ignored by git).
- `tests/`: Unit and integration tests.
- `docs/`: Project documentation.
- `agents/`: Documentation and logic specific to agents.

## Getting Started

### 1. Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
- Python 3.10+.

### 2. Environment Setup
Create and activate the conda environment:
```bash
conda create --name ain501 python=3.10
conda activate ain501
```

### 3. Installation
Install the project in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

## Developer Guidelines
Detailed rules for commits and code checks can be found in [agents/rules.md](agents/rules.md).

### Quick Checks
- **Format**: `black .`
- **Lint**: `ruff check .`
- **Type Check**: `mypy .`
- **Test**: `pytest`