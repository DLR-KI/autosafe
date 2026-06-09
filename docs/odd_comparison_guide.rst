.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

ODD Comparison Methods Guide
==============================

.. currentmodule:: autosafe.odd.comparison

This guide documents the comprehensive ODD comparison framework that extends beyond
traditional convex hull validation. The framework supports four methods:

1. **k-Nearest Neighbors (KNN)** with threshold-based membership
2. **k-Means Clustering** with silhouette-optimized convex hulls per cluster
3. **Superlevel Set Analysis** using kernel density estimation
4. **Hierarchical Convex Hulls** for sub-cluster boundaries

Quick Start
-----------

.. code-block:: python

    from autosafe.odd.comparison import KNNMonitor, KMeansBoundaries, SuperlevelSetMonitor
    from autosafe.tools.experiments import load_dataset
    import autosafe

    # Load your dataset
    df, _ = load_dataset("data/WineQT.csv")
    data = df.to_numpy().T  # Shape: (n_features, n_samples)

    # Create RBF kernel affinity ODD representation
    odd = autosafe.from_polars(df, kernel_cls="RBF")

    # Compare with different boundary methods
    methods = {
        "knn": KNNMonitor(k=3, gamma=None),  # Auto-detects reasonable gamma
        "kmeans": KMeansBoundaries(n_clusters=3),  # Finding 3 clusters
        "density": SuperlevelSetMonitor(gamma=0.01),  # PDF > 0.01 threshold
    }

    # Fit all methods to reference data
    for name, method in methods.items():
        method.fit(data)

    # Test dataset quality using multiple perspectives
    test_points = generate_test_grid(odd)  # Create evaluation points
    results = {}

    for name, method in methods.items():
        decisions = np.array([method(p) for p in test_points])
        results[name] = {
            "coverage": decisions.mean(),
            "conservatism": method.compute_conservatism_metric(data)
        }

    # Now you can analyze how different methods view your ODD quality



Conservatism Control
^^^^^^^^^^^^^^^^^^^^

The Python implementation provides additional conservatism analysis:

.. code-block:: python

    # Automatically determine reasonable gamma based on data
    knn = KNNMonitor(k=5)  # gamma will be auto-detected
    knn.fit(data)

    # Manually set gamma for specific conservatism requirements
    conservative_knn = KNNMonitor(k=5, gamma=0.2)  # Exclude outliers
    liberal_knn = KNNMonitor(k=5, gamma=0.8)  # Include more regions

    # Analyze current gamma value's impact
    conservatism_score = knn.compute_conservatism_metric(data)
    print(f"Conservatism level: {conservatism_score:.2%}")

Parameter Guide
^^^^^^^^^^^^^^^

+--------------------+----------------------+-----------------------------------------------+
| Parameter          | Type / Default       | Description                                   |
+====================+======================+===============================================+
| **k**              | ``int = 3``         | Number of nearest neighbors                    |
+--------------------+----------------------+-----------------------------------------------+
| **gamma**          | ``Optional[float]``  | Distance threshold (auto-detected if None)     |
+--------------------+----------------------+-----------------------------------------------+
| **metric**         | ``str = "euclidean"`` | Distance metric for KDTree                    |
+--------------------+----------------------+-----------------------------------------------+
| **leaf_size**      | ``int = 40``         | KDTree optimization parameter                 |
+--------------------+----------------------+-----------------------------------------------+

KMeansBoundaries: Cluster-Based Analysis
----------------------------------------

.. autoclass:: KMeansBoundaries
    :members:
    :undoc-members:

The k-means approach outperforms single hull methods by identifying natural data
distributions and validating clusters individually.

.. code-block:: python

    from autosafe.odd.comparison.kmeans_boundaries import KMeansBoundaries

    # Standard approach
    kmeans = KMeansBoundaries(n_clusters=3)
    kmeans.fit(data)

    # Get cluster information
    cluster_info = kmeans.get_cluster_info()
    print(f"Silhouette Score: {cluster_info['silhouette']:.3f}")

    # Conservative validation (exclude small clusters)
    conservative = KMeansBoundaries(n_clusters=4, min_cluster_size=20)

Cluster Size Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the auto-detection helper:

.. code-block:: python

    from autosafe.odd.comparison.utils import auto_detect_optimal_k

    optimal_k = auto_detect_optimal_k(data)
    print(f"Optimal cluster count: {optimal_k}")

Silhouette Validation
^^^^^^^^^^^^^^^^^^^^^^

Automatically calculates silhouette score (0.7 = good separation):

.. code-block:: python

    kmeans = KMeansBoundaries(n_clusters=optimal_k)
    kmeans.fit(data)

    conservatism = kmeans.compute_conservatism_metric()
    print(f"Silhouette-based conservatism: {conservatism:.2%}")

SuperlevelSetMonitor: Density Analysis
--------------------------------------

.. autoclass:: SuperlevelSetMonitor
    :members:
    :undoc-members:

Extends assessment to include probability analysis, particularly valuable when data
forms complex manifolds rather than simple clusters.

.. code-block:: python

    from autosafe.odd.comparison.density import SuperlevelSetMonitor

    # Auto-detect gamma based on PDF distribution
    density_monitor = SuperlevelSetMonitor(gamma=None)  # Auto-detects 75th percentile
    density_monitor.fit(data)

    # Custom gamma values
    conservative = SuperlevelSetMonitor(gamma=0.005)  # Very selective
    liberal = SuperlevelSetMonitor(gamma=0.1)       # More permissive

PDF Visualization
^^^^^^^^^^^^^^^^^

Powerful visualization for identifying optimal thresholds:

.. code-block:: python

    import matplotlib.pyplot as plt
    from numpy import linspace, meshgrid

    # Create grid for contour plot
    bounding_box = (data.min(axis=1), data.max(axis=1))
    grid_points, grid_pdf = density_monitor.create_visualization_grid(bounding_box, 50)

    # Plot PDF landscape with threshold
    plt.figure(figsize=(10, 6))
    plt.contourf(grid_points[0], grid_points[1], grid_pdf.reshape(50, 50), levels=20)
    plt.contour(grid_points[0], grid_points[1], grid_pdf.reshape(50, 50),
                levels=[density_monitor.gamma], colors='navy', linewidths=2)
    plt.title(f"Superlevel Set ODD (gamma={density_monitor.gamma:.4f})")
    plt.colorbar(label="Probability Density")
    plt.show()

Integration with Existing Workflow
==================================

The comparison methods integrate seamlessly with both traditional ODD analysis and the Monte Carlo affinity framework.

Monte Carlo with Inequality Constraints
----------------------------------------

autoSAFE supports custom ODD definitions using inequality constraints through YAML configuration.

Basic Usage
^^^^^^^^^^^

To define an ODD with linear inequality constraints:

1. Create a YAML configuration file (e.g., ``custom_odd.yaml``):

.. code-block:: yaml

    type: polytope
    dim: 2
    constraints:
        # Constraint: x1 >= x2 + 4
        - type: linear
            coefficients: [1.0, -1.0]  # 1*x1 - 1*x2
            relation: ">="
            bound: 4.0

        # Additional constraints
        - type: linear
            coefficients: [1.0, 0.0]
            relation: "<="
            bound: 10.0  # x1 <= 10

2. Use the configuration with Monte Carlo sampling:

.. code-block:: bash

    autosafe montecarlo sample --odd-config custom_odd.yaml

Supported Constraint Types
^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, only **linear constraints** are supported:

- **Format**: :math:`\mathbf{a} \cdot \mathbf{x} \leq b`
- **Examples**:
    - ``x1 >= x2 + 4`` (converted to ``x1 - x2 >= 4``)
    - ``2*x1 - x2 <= 5``
    - ``x1 + x2 + x3 >= 0``

**Unsupported**: Quadratic, cubic, nonlinear, or trigonometric constraints.

Constraint Specification
^^^^^^^^^^^^^^^^^^^^^^^^

Each constraint in the YAML file has the following fields:

+-------------------+---------------------+-------------------------------------------------------------+
| Field             | Type               | Description                                                 |
+===================+=====================+=============================================================+
| ``type``          | string (linear)    | Type of constraint (currently only linear is supported)     |
+-------------------+---------------------+-------------------------------------------------------------+
| ``coefficients``  | list[float]        | Coefficient vector :math:`\mathbf{a}` for :math:`\mathbf{a} \cdot \mathbf{x}` |
+-------------------+---------------------+-------------------------------------------------------------+
| ``relation``      | string             | Inequality relation: ``<=``, ``>=``, ``<``, ``>``           |
+-------------------+---------------------+-------------------------------------------------------------+
| ``bound``         | float              | Scalar bound :math:`b` in :math:`\mathbf{a} \cdot \mathbf{x} \leq b` |
+-------------------+---------------------+-------------------------------------------------------------+

Examples of Complex ODD Definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**1. 3D polytope with multiple constraints**:

.. code-block:: yaml

    type: polytope
    dim: 3
    constraints:
        - type: linear
            coefficients: [1.0, 1.0, 0.0]
            relation: ">="
            bound: 5.0  # x1 + x2 >= 5
        - type: linear
            coefficients: [0.0, 1.0, -2.0]
            relation: "<="
            bound: 3.0  # x2 - 2*x3 <= 3

**2. Bounded region with inequality relationships**:

.. code-block:: yaml

    type: polytope
    dim: 2
    constraints:
        - type: linear
            coefficients: [1.0, -2.0]
            relation: ">="
            bound: 0.0  # x1 >= 2*x2
        - type: linear
            coefficients: [2.0, 1.0]
            relation: "<="
            bound: 15.0 # 2*x1 + x2 <= 15

**3. Traditional box constraints expressed as inequalities**:

.. code-block:: yaml

    type: polytope
    dim: 2
    constraints:
        - type: linear
            coefficients: [1.0, 0.0]
            relation: ">="
            bound: -5.0  # x1 >= -5
        - type: linear
            coefficients: [1.0, 0.0]
            relation: "<="
            bound: 5.0   # x1 <= 5
        - type: linear
            coefficients: [0.0, 1.0]
            relation: ">="
            bound: -3.0  # x2 >= -3
        - type: linear
            coefficients: [0.0, 1.0]
            relation: "<="
            bound: 3.0   # x2 <= 3

Convex Hull Comparison
----------------------

Replace standard hull analysis with comprehensive multi-method validation:

.. code-block:: python

    from autosafe.tools.monte_carlo._evaluate import evaluate_comprehensive_odd

    # Add new methods to evaluation pipeline
    results = evaluate_comprehensive_odd(
        autosafe_odd=odd,
        comparison_methods=["hull", "knn", "kmeans", "density"],
        ref_points=data
    )

Analysis Dashboard
------------------

Compile results across methods for comprehensive validation:

.. code-block:: python

    def analyze_comparison_results(results):
        """Create comparison dashboard showing trade-offs"""
        import pandas as pd

        metrics = []
        for method_name, method_results in results.items():
            metrics.append({
                'method': method_name,
                'coverage': method_results.get('coverage_ratio'),
                'conservatism': method_results.get('conservatism'),
                'precision': method_results.get('precision')
            })

        return pd.DataFrame(metrics).sort_values('conservatism')

Method Comparison Guide
-----------------------

+----------------------+--------+-----------------------------+
| Method               | Complexity | Best Use Case              |
+======================+========+=============================+
| **Convex Hull**      | Low    | Simple boundary validation   |
| **k-Nearest Neighbors**| Medium| Outlier rejection             |
| **k-Means Clusters** | Medium| Natural data segmentation     |
| **Superlevel Sets**  | High   | Complex probability regions  |
| **Hierarchical Hulls**| High   | Multi-scale data distributions|
+----------------------+--------+-----------------------------+

Best Practices
-------------

1. Start with all methods enabled (`comparison_methods=["hull", "knn", "kmeans", "density"]`)
2. Use conservatism metrics to identify appropriate gamma/k parameter values
3. Visualize density landscapes to identify meaningful probability thresholds
4. Compare silhouette scores when validating cluster definitions

Troubleshooting
---------------

**Import Issues**: Ensure scikit-learn is installed for clustering methods:

.. code-block:: bash

    pip install scikit-learn

**Memory Issues**: Consider smaller sample sizes for large datasets, especially with density methods which compute full covariance matrices.

**Conservatism Not Matching Expectations**: Use auto-detection for initial parameter guidance, then fine-tune manually based on confusion matrix analysis.

Validation
----------

Compare with existing MC融eval_data僧

.. code-block:: python

    # Run Monte Carlo evaluation on reference datasets (HCAS, VCAS)
    from autosafe.tools.monte_carlo import evaluate_mc_integration

    # Validate against historical data
    results = evaluate_mc_integration("data/WineQT.csv", methods=["knn", "kmeans"])

Benchmark
---------

Compare performance across methods during development:

.. code-block:: python

    import time

    methods = [
        KNNMonitor(k=3),
        KMeansBoundaries(n_clusters=4),
        SuperlevelSetMonitor(gamma=0.01)
    ]

    for method in methods:
        start = time.time()
        method.fit(data)
        fit_time = time.time() - start

        start = time.time()
        results = method.evaluate_batch(data)
        eval_time = time.time() - start

        print(f"{method.method_type}: fit={fit_time:.4f}s, eval={eval_time:.4f}s")
