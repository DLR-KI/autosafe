.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

Comparison Methods
==================

This page documents all currently supported comparison references used by ``autosafe evaluate`` and ``autosafe comparison``.

Method set
----------

Single-shape references
^^^^^^^^^^^^^^^^^^^^^^^

- ``hull_single``

    Single convex hull over all reference points.

- ``density_single``

    Single KDE superlevel-set boundary using :class:`autosafe.odd.comparison.SuperlevelSetMonitor`.

- ``gmm``

    Gaussian-mixture superlevel-set boundary using :class:`autosafe.odd.comparison.GaussianMixtureBoundary` (subclasses ``SuperlevelSetMonitor``, replacing the KDE score with the log-likelihood of a fitted Gaussian mixture).
    Component count is selected by BIC over a configurable range rather than fixed.

Clustered references
^^^^^^^^^^^^^^^^^^^^

- ``hull_clustered``

    Clustered convex hull union using :class:`autosafe.odd.comparison.ClusteredConvexHulls`.

- ``density_clustered``

    Clustered superlevel-set union using :class:`autosafe.odd.comparison.ClusteredSuperlevelSetMonitor`.

- ``dbscan_cluster``

    DBSCAN-based clustered boundary using :class:`autosafe.odd.comparison.DBSCANCluster`.

Neighborhood/partition references
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``knn``

    Nearest-neighbor threshold boundary using :class:`autosafe.odd.comparison.KNNMonitor`.

- ``kmeans``

    K-means boundary method using :class:`autosafe.odd.comparison.KMeansBoundaries`.

Opt-in references (not in the default set)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are registered, supported comparison methods, but must be requested explicitly via ``references:`` -- see `Default baseline set`_ below for why.

- ``oneclass_svm``

    One-class SVM boundary (RBF kernel) using :class:`autosafe.odd.comparison.OneClassSVMBoundary`.

- ``svdd``

    Support Vector Data Description boundary (Tax & Duin, 2004) using :class:`autosafe.odd.comparison.SVDDBoundary`; under an RBF kernel it is mathematically equivalent to ``oneclass_svm`` with matched hyperparameters (do not report both as independent evidence in that case -- use ``kernel="poly"`` for a genuinely distinct comparison).

Alias compatibility
-------------------

- ``hull`` is treated as ``hull_single``.
- ``density`` is treated as ``density_single``.

Default baseline set
---------------------

``DEFAULT_DATASET_BASELINES`` (``autosafe.tools.evaluate.dataset.baselines``) is used whenever a dataset spec item does not set ``references:``: ``hull_single``, ``hull_clustered``, ``knn``, ``kmeans``, ``density_single``, ``density_clustered``, ``dbscan_cluster``, and ``gmm``.

``oneclass_svm`` and ``svdd`` are deliberately excluded.
Every comparison monitor here is fit on the *full* anchor set (chunking only affects evaluation, not fitting), and at aviation scale (622k anchors) OC-SVM is :math:`O(n^2)`--:math:`O(n^3)` via libsvm while SVDD builds a dense :math:`n \times n` kernel matrix for its QP -- about 3 TB at that size.
Spec items such as ``eval-vcas-rbf`` / ``eval-hcas-rbf`` do not override ``references:``, so putting either method in the default would make those runs unrunnable.
GMM is included by default because EM is cheap in comparison: :math:`O(n \cdot k \cdot d^2)` per iteration.

Adding a method to ``references:`` only adds *rows* to the dataset CSV (one row per ``source, reference, affinity_space, threshold``); it never alters existing rows, so previously reported baseline numbers are unaffected.

Evaluation behavior
-------------------

Dataset evaluation (``autosafe evaluate dataset``):

- If YAML ground truth is present (explicitly via ``--ground-truth-yaml`` or
    auto-detected as sibling ``.yml/.yaml``), results include:

    - all methods in the default baseline set (see `Default baseline set`_ above), plus any explicitly requested ``references:``, and
    - ``ground_truth`` membership from YAML containment.

- If YAML ground truth is absent, only requested baseline methods are used.

Monte Carlo result evaluation (``autosafe evaluate sampling-results``):

- Uses ``in_odd`` from MC JSON as ``ground_truth`` when requested.
- Baseline methods are evaluated from anchors and sampled points.

Parameter reference and helpers
--------------------------------

``KNNMonitor`` accepts:

+---------------+-----------------------+-------------------------------------------------+
| Parameter     | Type / Default        | Description                                     |
+===============+=======================+=================================================+
| **k**         | ``int = 3``           | Number of nearest neighbors.                    |
+---------------+-----------------------+-------------------------------------------------+
| **gamma**     | ``Optional[float]``   | Distance threshold (auto-detected if ``None``). |
+---------------+-----------------------+-------------------------------------------------+
| **metric**    | ``str = "euclidean"`` | Distance metric for the ``KDTree``.             |
+---------------+-----------------------+-------------------------------------------------+
| **leaf_size** | ``int = 40``          | ``KDTree`` optimization parameter.              |
+---------------+-----------------------+-------------------------------------------------+

``KMeansBoundaries`` exposes a cluster-count helper (:func:`autosafe.odd.comparison.auto_detect_optimal_k`) and per-cluster diagnostics:

.. code-block:: python

    import polars as pl

    from autosafe.odd.comparison import (
        KMeansBoundaries,
        KNNMonitor,
        auto_detect_optimal_k,
    )

    df = pl.read_csv("data/iris.csv")
    data = df.to_numpy().T  # shape: (n_features, n_samples)

    knn = KNNMonitor(k=5)
    knn.fit(data)
    knn.compute_conservatism_metric()  # KNNMonitor: conservatism is a method

    k = auto_detect_optimal_k(data)  # sqrt(n_samples), capped at max_k_upper=10
    kmeans = KMeansBoundaries(n_clusters=k)
    kmeans.fit(data)
    kmeans.get_cluster_info()["silhouette"]  # cluster-quality score
    kmeans.conservatism  # KMeansBoundaries: conservatism is an attribute set by fit()

Conservatism is surfaced differently per method: ``KNNMonitor`` via ``compute_conservatism_metric()``, ``KMeansBoundaries`` via the ``conservatism`` attribute set during ``fit()``, and ``SuperlevelSetMonitor`` only through its ``decision_boundary`` property (there is no public ``compute_conservatism_metric`` on the latter two).

``SuperlevelSetMonitor.fit`` always recalibrates ``gamma`` to the 75th percentile of the reference PDF (:func:`~autosafe.odd.comparison.density.SuperlevelSetMonitor.suggest_reasonable_gamma`), overriding whatever value was passed to the constructor.
Call ``suggest_reasonable_gamma(percentile=...)`` directly after fitting for a different threshold.

Example
-------

.. code-block:: console

    autosafe evaluate dataset data/hcas_state_variables.csv \
        --references hull_single hull_clustered knn kmeans \
            density_single density_clustered dbscan_cluster gmm
