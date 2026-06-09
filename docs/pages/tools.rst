.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

Tools and CLI Reference
=======================

This page documents all high-level autoSAFE tools exposed through the CLI:

- ``autosafe montecarlo``
- ``autosafe evaluate``
- ``autosafe comparison``
- ``autosafe experiments``

The tools can be composed in workflows, but each can be used independently.


autosafe evaluate
-----------------

The evaluate tool is the primary path for metric generation.

dataset
^^^^^^^

Command:

.. code-block:: console

    autosafe evaluate dataset <dataset_path> [options]

Behavior:

- Loads data and applies robust normalization for kernel-affinity stability.
- Builds affinity ODD (or reuses ``--odd-json``).
- Samples evaluation points around the affinity ODD.
- Sweeps affinity thresholds and computes confusion-matrix metrics.
- Compares affinity ODD against reference methods.

Ground truth YAML behavior:

- If ``--ground-truth-yaml`` is provided, it is used.
- Otherwise, a sibling ``.yml``/``.yaml`` next to the dataset is auto-detected.
- If YAML exists, the evaluation includes ``ground_truth`` in addition to baselines.

Supported baseline references:

- ``hull_single``: single convex hull over all reference points
- ``hull_clustered``: union of convex hulls from clustered subregions
- ``knn``: nearest-neighbor threshold monitor
- ``kmeans``: k-means cluster boundary method
- ``density_single``: single KDE superlevel-set boundary
- ``density_clustered``: clustered KDE superlevel-set union
- ``dbscan_cluster``: DBSCAN-core density cluster boundary

Alias compatibility:

- ``hull`` -> ``hull_single``
- ``density`` -> ``density_single``

sampling-results
^^^^^^^^^^^^^^^^

Command:

.. code-block:: console

    autosafe evaluate sampling-results --file <json> [--file <json> ...] [options]

Behavior:

- Reads Monte Carlo sampling result JSON files.
- Uses stored ``affinity`` and ``in_odd`` values.
- Evaluates selected references across threshold grids.
- Writes per-threshold metric rows to CSV.


autosafe comparison
-------------------

Independent method comparison on arbitrary datasets.

evaluate
^^^^^^^^

Command:

.. code-block:: console

    autosafe comparison evaluate <dataset_path> [--methods ...]

Supported methods:

- ``hull_single``
- ``knn``
- ``kmeans``
- ``density_single``
- ``hull_clustered``
- ``density_clustered``
- ``dbscan_cluster``

quick
^^^^^

Runs all methods above with default parameters for fast diagnostics.

info
^^^^

Prints method summaries and intended use-cases.


autosafe montecarlo
-------------------

sample
^^^^^^

Generates Monte Carlo sampling data. Supports standard box settings and custom ODD YAML constraints.
Config files can be provided as JSON or YAML via ``--config-file``.
Folders passed with ``--config-file-folder`` may contain ``.json``, ``.yaml``, and ``.yml`` files.

Examples:

.. code-block:: console

    autosafe montecarlo sample --dim 2 --odd-limits 5 --samples 1000
    autosafe montecarlo sample --config-file sample_config.json
    autosafe montecarlo sample --config-file-folder configs/

evaluate
^^^^^^^^

Legacy compatibility command for MC-result evaluation. Prefer
``autosafe evaluate sampling-results`` for full parameter control.


autosafe experiments
--------------------

Batch runner for structured specs.

run-spec
^^^^^^^^

Runs multi-item evaluation specs with optional resume state and stop-on-error behavior.

Supported spec modes:

- ``mc-sample``: run Monte Carlo sampling from a config file
    (``config_file``)
- ``dataset``: run dataset evaluation
- ``mc-results``: evaluate one or more Monte Carlo results files


Metric outputs
--------------

CSV output rows include, per source/reference/threshold:

- confusion matrix values (TP, FP, TN, FN)
- precision, recall, specificity, F1, accuracy
- additional derived rates from the shared metric engine


Notes on normalization and stability
------------------------------------

Kernel-affinity ODD construction is sensitive to strongly heterogeneous feature ranges.
For stability, numeric dataset columns are robustly normalized before affinity ODD fitting.
Ground-truth YAML containment is evaluated directly from the YAML polytope definition.
