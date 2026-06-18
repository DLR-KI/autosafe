<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>

SPDX-License-Identifier: CC-BY-SA-4.0
-->

<!-- markdownlint-disable MD033 -->
<h1 align="center">
<img src="./docs/_static/autoSAFE.svg" alt="autoSAFE logo" width="300">
</h1><br>
<!-- markdownlint-enable MD033 -->

[![The latest version of autosafe can be found on PyPI.](https://img.shields.io/pypi/v/autosafe.svg)](https://pypi.python.org/pypi/autosafe)
[![Information on what versions of Python autosafe supports can be found on PyPI.](https://img.shields.io/pypi/pyversions/autosafe.svg)](https://pypi.python.org/pypi/autosafe)
[![Python tests (pytest)](https://github.com/DLR-KI/autosafe/actions/workflows/pytest.yaml/badge.svg)](https://github.com/DLR-KI/autosafe/actions/workflows/pytest.yaml)
[![Docs status](https://readthedocs.org/projects/autosafe/badge/)](https://autosafe.readthedocs.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/DLR-KI/autosafe)
[![REUSE status](https://api.reuse.software/badge/github.com/DLR-KI/autosafe)](https://api.reuse.software/info/github.com/DLR-KI/autosafe)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

Reference implementation of the autoSAFE specification.

autoSAFE can derive semantically correct Operational Design Domains (ODDs) purely from data.
This allows to automatically generate ODDs for machine learning-based functions.
The use cases range from data-driven ODD definition over ODD monitoring to retrofitting existing functions with ODDs.
Moreover, if no ODD is given for a certain dataset, autoSAFE can derive one automatically to ensure safe operations of the resulting AI-based system

## Installation

We recommend using [uv](https://astral.sh/uv/) to manage the virtual environment and dependencies.
Simply running `uv sync` will create a virtual environment and install all dependencies as specified in `pyproject.toml`.

### CVXOPT

Users might want to optionally install [CVXOPT](https://pypi.org/project/cvxopt/) for enhanced performance.
However, as CVXOPT is licensed under GPL-v3, it is not included as a dependency by default.
Moreover, CVXOPT is not compatible with Python versions >= 3.14 or free-threaded builds as of now.

## Usage

### CLI

autoSAFE provides a command-line interface (CLI) for easy interaction.
After installing the package, you can use the `autosafe` command in your terminal:

```shell
uv run autosafe --help
```

This will list all available commands and options.
They are described in detail in the [documentation](https://autosafe.readthedocs.io/).

### Common CLI Commands

Here are the most frequently used commands:

#### Monte Carlo Sampling and Evaluation

```bash
# Run Monte Carlo sampling with default 2D box ODD
autosafe montecarlo sample --dim 2 --odd-limits 5.0 --samples 1000 --filename mc_results.json

# Evaluate results and generate precision/recall plots
autosafe montecarlo evaluate mc_results.json
```

#### ODD Comparison Methods

```bash
# Quick comparison with default parameters
autosafe comparison quick data/WineQT.csv --export results.json

# Full comparison with custom parameters
autosafe comparison evaluate data/WineQT.csv \
    --methods knn kmeans density \
    --knn-k 5 --kmeans-clusters 4 --density-gamma 0.01
```

#### Custom ODD with Inequalities

```bash
# Use YAML config for polytope with inequality constraints
autosafe montecarlo sample --odd-config odd_config.yaml
```

### API

autoSAFE also offers a Python API for more advanced usage.
To start, you can read your data from a CSV file and create an autoSAFE ODD as follows:

```python
import autosafe as af
import numpy as np

# Load data from a CSV file
odd = af.from_csv(af.ROOT_FOLDER / "data" / "iris.csv")

# Query the ODD for a new data point
affinity_threshold = 0.8
data_point = np.array([[5.1, 3.5, 1.4, 0.2]])
is_within_odd = odd.contains(data_point) >= affinity_threshold
print(f"The data point is within the ODD: {is_within_odd}")
```

## Developing

### Pre-Commit Hooks

We use [prek](https://github.com/j178/prek) to manage pre-commit hooks for code quality and consistency.
To install the pre-commit hooks, run the following command from the project root:

```shell
uv run prek install
```

### Tests

We use [pytest](https://pytest.org/) for testing.
To run the test suite, execute the following command from the project root:

```shell
uv run pytest
```

## License

Copyright and license information are provided in accordance to the [REUSE Specification 3.3](https://reuse.software/spec-3.3/).
