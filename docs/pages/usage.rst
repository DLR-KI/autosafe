.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

Usage
=====

autoSAFE exposes four CLI tool groups:

- ``montecarlo``
- ``evaluate``
- ``comparison``
- ``experiments``

List top-level commands:

.. code-block:: console

    autosafe --help

Most users should start with ``evaluate dataset``.

Quick start
-----------

.. code-block:: console

    autosafe evaluate dataset data/vcas_state_variables.csv

If ``data/vcas_state_variables.yml`` exists, it is automatically used as
ground-truth ODD.

Monte Carlo config-file workflow
--------------------------------

The config-file based Monte Carlo entry point is still supported.
Both JSON and YAML files are accepted.

.. code-block:: console

    autosafe montecarlo sample --config-file sample_config.json
    autosafe montecarlo sample --config-file experiments/dim_2d/sampling_config.yaml

This runs sampling with the full JSON configuration and overrides
command-line shape parameters.

Detailed references
-------------------

For complete command options, method definitions, and workflow semantics, see:

- :doc:`tools`
- :doc:`comparison_methods`

Python API
----------

A minimal API example:

.. code-block:: python

    import autosafe as af
    import polars as pl

    df = pl.read_csv("data/vcas_state_variables.csv")
    odd = af.from_polars(df, closest_sample_mode="per_dimension", kernel_cls="RBF")

    # Affinity for a batch of points
    points = df.head(10).to_numpy()
    affinity = odd(points)
    print(affinity)
