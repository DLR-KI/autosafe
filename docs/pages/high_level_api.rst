.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

High-level ODD API
==================

``AutoSafeODD.fit`` constructs a queryable ODD while preserving the input
data, requested configuration, and all data-derived parameters. It accepts
either typed configuration objects or a mapping:

.. code-block:: python

    import autosafe

    config = autosafe.AutoSafeConfig.from_mapping(
        {
            "gamma": 1.0,
            "s": 3.0,
            "lambda_rel": 1e-4,
            "zeta": 0.7,
            "xi": 0.2,
            "c": 0.9,
        }
    )

    odd = autosafe.AutoSafeODD.fit(
        in_distribution_data=id_anchors,
        out_of_distribution_data=observed_ood,
        config=config,
        feature_names=feature_names,
    )

    inside = odd.contains(query_rows)
    affinity = odd.affinity(query_rows)

The ID anchors and observed OOD points are separate inputs. Providing OOD
points requires an OOD configuration, and configuring OOD consistency
requires OOD points. All data are transformed by the same normalizer fitted
on ID anchors.

Parameter names and paper symbols
---------------------------------

Every tuning parameter has one canonical descriptive name. The mapping API
also accepts ASCII and paper-symbol aliases:

.. list-table::
    :header-rows: 1

    * - Canonical name
      - Paper symbol
      - Common ASCII alias
      - Meaning
    * - ``kernel.decay_per_median_gap``
      - ``γ``
      - ``gamma``
      - Calibrated decay per median nearest-neighbor gap.
    * - ``kernel.width_in_median_gaps``
      - ``s``
      - ``s``
      - Calibrated maximum width in median gaps.
    * - ``kernel.relative_variance_floor``
      - ``λ_rel``
      - ``lambda_rel``
      - Variance floor relative to ``κ``.
    * - ``kernel.maximum_variance``
      - ``κ``
      - ``kappa``
      - Manual maximum variance.
    * - ``kernel.distance_decay_rate``
      - ``η``
      - ``eta``
      - Manual distance-decay rate.
    * - ``kernel.variance_floor``
      - ``λ``
      - ``lambda`` or ``lam``
      - Manual absolute variance floor.
    * - ``membership.affinity_threshold``
      - ``ζ``
      - ``zeta``
      - Fixed ODD membership threshold.
    * - ``membership.target_false_exclusion_rate``
      - ``ε``
      - ``epsilon`` or ``eps``
      - Split-conformal false-exclusion target.
    * - ``ood.max_affinity``
      - ``ξ``
      - ``xi``
      - Maximum affinity at every observed OOD point.
    * - ``ood.covariance_shrink_factor``
      - ``c``
      - ``shrink_factor``
      - OOD covariance shrink factor.
    * - ``evaluation.local_noise_multiple``
      - ``m``
      - ``local_noise_multiple``
      - Evaluation displacement in median gaps.

The registry is available as ``autosafe.PARAMETER_SPECS``. Unknown names,
duplicate aliases, and mixing manual and calibrated kernel parameters raise
an error. ``calibration_c`` remains a deprecated alias for ``gamma``;
unqualified ``c`` always means the OOD covariance shrink factor.

Calibrated and manual kernels
-----------------------------

Calibrated mode is selected by ``gamma``/``s`` or explicitly by
``kernel.mode: calibrated``. With median positive full-space
nearest-neighbor distance ``d_tilde``, fitting resolves:

.. code-block:: text

    kappa = (s * d_tilde) ** 2
    eta = gamma / d_tilde
    lambda = lambda_rel * kappa

It requires at least two ID anchors and uses global nearest-neighbor
assignment.

Manual mode is selected by ``kappa``/``eta`` or explicitly by
``kernel.mode: manual``. Scalars or one value per feature are accepted.
Define either the absolute ``lambda`` or ``lambda_rel``, but not both.

Membership and OOD ordering
---------------------------

Use exactly one membership mode:

- ``zeta`` selects a fixed affinity threshold.
- ``epsilon`` selects split-conformal calibration and requires held-out ID
  rows through ``calibration_data``.

Fitting first constructs the kernels, then enforces ``affinity <= xi`` at
all observed OOD points, and only then calibrates a conformal membership
threshold. The resolved log-survival threshold is authoritative for
membership decisions near affinity one. If OOD consistency is enabled, the
final requirement is ``0 < xi < zeta < 1``.

Requested and resolved configuration
------------------------------------

``odd.config`` retains the requested typed configuration.
``odd.resolved_config`` additionally records ``d_tilde``, ``kappa``,
``eta``, ``lambda``, the resolved ``zeta``, its log-survival threshold,
and conformal calibration size when applicable.

Configurations can be converted to plain dictionaries:

.. code-block:: python

    canonical = config.to_mapping(style="canonical")
    paper = config.to_mapping(style="paper")

Both forms can be passed back to ``AutoSafeConfig.from_mapping`` without
losing configuration values.

ASAM OpenODD 1.0.0 YAML export
------------------------------

A fitted model can be exported as validated ASAM OpenODD 1.0.0 YAML.
Map each model dimension to either a numeric OpenODD concept or a
categorical concept with encoded prototypes:

.. code-block:: python

    features = (
        autosafe.OpenODDNumericFeature(
            index=0,
            concept_id="ambient_temperature",
            unit_type="temperature",
            unit="C",
        ),
        autosafe.OpenODDCategoricalFeature(
            indices=(1, 2),
            concept_id="surface_state",
            literals={
                "dry": (1.0, 0.0),
                "wet": (0.0, 1.0),
            },
        ),
    )

    path = autosafe.write_openodd_yaml(
        odd,
        "derived-odd.yaml",
        features=features,
        metadata=autosafe.OpenODDExportMetadata(
            title="Derived operating conditions",
            description="ODD derived from observed in-distribution data",
        ),
    )

The YAML ODD is a conservative inner approximation of the fitted kernel
level set. It is represented as a root ``ODD`` module referring to a union
of standard modules. Those modules contain only standard range and
categorical expressions over concepts declared in ``TAXONOMY``.

See :doc:`openodd_derivation` for the complete construction, its
conservativeness argument, and its time and space complexity. In particular,
the exporter does not randomly sample the fitted space: each fitted ID anchor
deterministically produces at most one candidate OpenODD region.

autoSAFE thresholds, requested configuration, paper notation, resolved
data-derived values, and approximation details are written only to the root
module's standard ``METADATA`` field. They never become ODD conditions.

Every document is checked before writing. The public
``OPENODD_1_0_YAML_SCHEMA`` contains the derived Draft 2020-12 structural
schema. ``validate_openodd_document`` additionally validates typed
expressions, categorical literals, identifier collisions, module
references, forbidden root dependencies, and dependency cycles.
``load_openodd_yaml`` rejects invalid documents and duplicate YAML keys.
