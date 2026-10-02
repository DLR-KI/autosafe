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
- ``gmm``: Gaussian-mixture superlevel-set boundary, component count selected by BIC over a configurable range
- ``oneclass_svm``: one-class SVM boundary (RBF kernel)
- ``svdd``: Support Vector Data Description boundary (Tax & Duin, 2004); under an RBF kernel it is mathematically equivalent to ``oneclass_svm`` with matched hyperparameters (do not report both as independent evidence in that case -- use ``kernel="poly"`` for a genuinely distinct comparison)

``DEFAULT_DATASET_BASELINES`` -- used whenever a dataset spec item does not set ``references:`` -- is ``hull_single``, ``hull_clustered``, ``knn``, ``kmeans``, ``density_single``, ``density_clustered``, ``dbscan_cluster``, and ``gmm``.
``oneclass_svm`` and ``svdd`` are deliberately **not** in the default and must be requested explicitly via ``references:``: every comparison monitor here is fit on the *full* anchor set (chunking only affects evaluation), and at aviation scale (622k anchors) OC-SVM is :math:`O(n^2)`--:math:`O(n^3)` via libsvm while SVDD builds a dense :math:`n \times n` kernel matrix for its QP -- about 3 TB at that size.
Since spec items such as ``eval-vcas-rbf`` / ``eval-hcas-rbf`` do not override ``references:``, putting either in the default would make those runs unrunnable.
GMM is cheap enough to include by default (EM is :math:`O(n \cdot k \cdot d^2)` per iteration).
Adding a reference only adds *rows* to the dataset CSV (one row per ``source, reference, affinity_space, threshold``); it never alters existing rows, so previously reported baseline numbers are unaffected.

Alias compatibility:

- ``hull`` -> ``hull_single``
- ``density`` -> ``density_single``

``baselines_only``:

- When ``true`` on a spec item, the affinity ODD is loaded from ``odd_json`` instead of being rebuilt, and requires that cached ODD to match the requested kernel settings exactly -- it never silently refreshes or rebuilds one. Use this to add new baseline-reference rows to an existing evaluation without re-incurring the (potentially expensive) ODD build, and with the autoSAFE affinity column guaranteed byte-identical to the run that produced ``odd_json``.

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

Generates Monte Carlo sampling data.
Supports standard box settings and custom ODD YAML constraints.
Config files can be provided as JSON or YAML via ``--config-file``.
Folders passed with ``--config-file-folder`` may contain ``.json``, ``.yaml``, and ``.yml`` files.

Examples:

.. code-block:: console

    autosafe montecarlo sample --dim 2 --odd-limits 5 --samples 1000
    autosafe montecarlo sample --config-file sample_config.json
    autosafe montecarlo sample --config-file-folder configs/

evaluate
^^^^^^^^

Legacy compatibility command for MC-result evaluation.
Prefer ``autosafe evaluate sampling-results`` for full parameter control.


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
