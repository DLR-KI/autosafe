.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

Comparison Methods
==================

This page documents all currently supported comparison references used by
``autosafe evaluate`` and ``autosafe comparison``.

Method set
----------

Single-shape references
^^^^^^^^^^^^^^^^^^^^^^^

- ``hull_single``

    Single convex hull over all reference points.

- ``density_single``

    Single KDE superlevel-set boundary using
    :class:`autosafe.odd.comparison.SuperlevelSetMonitor`.

Clustered references
^^^^^^^^^^^^^^^^^^^^

- ``hull_clustered``

    Clustered convex hull union using
    :class:`autosafe.odd.comparison.ClusteredConvexHulls`.

- ``density_clustered``

    Clustered superlevel-set union using
    :class:`autosafe.odd.comparison.ClusteredSuperlevelSetMonitor`.

- ``dbscan_cluster``

    DBSCAN-based clustered boundary using
    :class:`autosafe.odd.comparison.DBSCANCluster`.

Neighborhood/partition references
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``knn``

    Nearest-neighbor threshold boundary using
    :class:`autosafe.odd.comparison.KNNMonitor`.

- ``kmeans``

    K-means boundary method using
    :class:`autosafe.odd.comparison.KMeansBoundaries`.

Alias compatibility
-------------------

- ``hull`` is treated as ``hull_single``.
- ``density`` is treated as ``density_single``.

Evaluation behavior
-------------------

Dataset evaluation (``autosafe evaluate dataset``):

- If YAML ground truth is present (explicitly via ``--ground-truth-yaml`` or
    auto-detected as sibling ``.yml/.yaml``), results include:

    - all baseline methods listed above, and
    - ``ground_truth`` membership from YAML containment.

- If YAML ground truth is absent, only requested baseline methods are used.

Monte Carlo result evaluation (``autosafe evaluate sampling-results``):

- Uses ``in_odd`` from MC JSON as ``ground_truth`` when requested.
- Baseline methods are evaluated from anchors and sampled points.

Example
-------

.. code-block:: console

    autosafe evaluate dataset data/hcas_state_variables.csv \
        --references hull_single hull_clustered knn kmeans \
            density_single density_clustered dbscan_cluster
