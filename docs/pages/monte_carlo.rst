.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

Monte Carlo Sampling
=====================

``autosafe montecarlo sample`` generates synthetic validation data for an *analytically specified* ground-truth ODD: it draws anchor points uniformly from that ODD, derives the kernel-based (autoSAFE) ODD from those anchors, then draws a larger set of validation points and records, for each one, both its ground-truth membership and its autoSAFE affinity.
This is the mechanism behind the paper's Monte Carlo validation (Section "Monte Carlo Validation", ``subsec:MonteCarloValidation``) and the two-to-twelve-dimensional synthetic sweep summarized under "Synthetic Benchmarks" (``subsec:MCSampling``); this page documents the configuration format and does not restate those results -- see the paper for the published precision/recall numbers.

This page covers how to *describe* the ground-truth ODD for that sweep: the ``box``/``polytope`` configuration schema, its three entry points, and the errors the validator raises.
It does not cover the evaluation side (threshold sweeps, confusion matrices, CSV export) -- see :doc:`tools` for ``autosafe evaluate sampling-results``.

Taxonomy, ontology, and the sampled region
-------------------------------------------

The paper defines an ODD as the structure :math:`\mathcal{O} = (X, R_1^\mathcal{O}, \dots, R_r^\mathcal{O}, f^\mathcal{O}, \Omega^\mathcal{O})`, where :math:`X \subseteq \mathbb{R}^n` is the domain (*taxonomy*) and each :math:`R_i^\mathcal{O} = \{\bm{x} \in X \mid R_i(\bm{x}) = 1\}` is the set of points satisfying one predicate (*ontology*); their conjunction is written :math:`\mathcal{R}^\mathcal{O} \equiv \bigcap_{i=1}^r R_i^\mathcal{O}`.
The Monte Carlo sampler mirrors this split directly:

- The outer sampling box (``odd_lower_limits``/``odd_upper_limits``, or ``--odd-limits`` on the CLI) always defines an axis-aligned taxonomy :math:`X`.
- A ``custom_odd_config`` (below) adds ontology predicates on top of it.
  When its ``type`` is ``polytope``, each ``constraints`` entry is one linear half-space predicate :math:`R_i`; the polytope library ANDs them together internally, so the resulting region is already :math:`\mathcal{R}^\mathcal{O} = \bigcap_i R_i^\mathcal{O}`.
  When its ``type`` is ``box``, the region is itself another axis-aligned taxonomy-like set -- useful standalone (see :ref:`mc-third-entry-point`), but redundant with the outer box if combined with it.
- Ground truth for a validation point is membership in :math:`X` intersected with the custom region, matching :math:`X \cap \mathcal{R}^\mathcal{O}`.

Read on for the exact schema; :ref:`mc-known-issue` documents one place where the *implementation* of this intersection currently does not match this description for the ``autosafe montecarlo sample`` pipeline specifically.

The ``box`` and ``polytope`` config types
-------------------------------------------

A custom ODD configuration is a mapping validated by ``autosafe.tools.monte_carlo.inequality_utils.validate_odd_config``.
Both types require ``type`` and ``dim``:

- ``type: box`` -- an axis-aligned region given directly by ``lower_bounds``/``upper_bounds``.
  Prefer this whenever the region *is* a product of per-dimension intervals: it is shorter, and the bounds read directly as the taxonomy ranges.
  The previous, now-removed draft of this page expressed every box as four separate ``polytope`` inequalities (:math:`x_1 \geq -5`, :math:`x_1 \leq 5`, ...); that still works, but ``type: box`` says the same thing in two fields instead of eight.
- ``type: polytope`` -- one or more linear half-space ``constraints``.
  Use this for genuine ontology predicates that are not axis-aligned, such as :math:`x_1 - x_2 \geq 4` (coupling two dimensions) -- exactly the constraint used in ``experiments/dim_2d/sampling_config.yaml`` through ``dim_12d``.

.. code-block:: yaml

    # box: an axis-aligned taxonomy region
    type: box
    dim: 2
    lower_bounds: [-5, -3]
    upper_bounds: [5, 3]

.. code-block:: yaml

    # polytope: one ontology predicate, x1 - x2 >= 4
    type: polytope
    dim: 2
    constraints:
      - type: linear
        coefficients: [1.0, -1.0]
        relation: ">="
        bound: 4.0

Field reference
-----------------

Top-level fields:

.. list-table::
    :header-rows: 1
    :widths: 15 15 70

    * - Field
      - Required
      - Description
    * - ``type``
      - Yes
      - ``"box"`` or ``"polytope"``.
        Any other value raises ``ValueError``.
    * - ``dim``
      - Yes
      - Number of dimensions.
        Every constraint's ``coefficients`` (polytope) or every bounds list (box) must match this length.
    * - ``lower_bounds``, ``upper_bounds``
      - Only for ``box``
      - Per-dimension bound lists (or a single scalar broadcast to all dimensions when used via :class:`~autosafe.tools.monte_carlo.inequality_utils.ODDFactory`).
    * - ``constraints``
      - Only for ``polytope``
      - Non-empty list of constraint mappings (below).
        Missing, non-list, or empty raises ``ValueError``.

Each entry in ``constraints``:

.. list-table::
    :header-rows: 1
    :widths: 15 15 70

    * - Field
      - Required
      - Description
    * - ``type``
      - No (default ``"linear"``)
      - Only ``"linear"`` is supported; any other value raises ``ValueError``.
        There is no quadratic, cubic, or trigonometric constraint type.
    * - ``coefficients``
      - Yes
      - Length-``dim`` coefficient vector :math:`\bm{a}`.
        A length that does not equal ``dim`` raises ``ValueError``.
    * - ``relation``
      - Yes
      - One of ``"<="``, ``"<"``, ``">="``, ``">"``.
        Anything else raises ``ValueError``.
    * - ``bound``
      - Yes
      - Scalar :math:`b`, coerced with ``float()``.

Every constraint is normalized to the canonical half-space form :math:`\bm{a} \cdot \bm{x} \leq b` before being handed to the ``polytope`` library.
For ``relation`` in ``{"<=", "<"}`` this is a no-op; for ``{">=", ">"}`` both the coefficients and the bound are negated (:math:`\bm{a} \cdot \bm{x} \geq b \iff -\bm{a} \cdot \bm{x} \leq -b`), so ``coefficients: [1.0, -1.0], relation: ">=", bound: 4.0`` (i.e. :math:`x_1 - x_2 \geq 4`) is stored internally as coefficients ``[-1.0, 1.0]`` with bound ``-4.0``.
The strict/non-strict distinction (``<`` vs. ``<=``) is not enforced separately -- both compile to the same closed half-space.

Entry points
-------------

All three entry points below resolve to the same validation and polytope construction (``validate_odd_config``, then :class:`~autosafe.tools.monte_carlo.inequality_utils.ODDFactory`; see "Programmatically" below for a runnable example of calling it directly).

1. CLI flag
^^^^^^^^^^^^

``--odd-config`` on ``autosafe montecarlo sample`` accepts a YAML file path:

.. code-block:: console

    autosafe montecarlo sample --dim 2 --odd-anchors 5 --samples 10 \
        --odd-config custom_odd.yaml --filename results.json

where ``custom_odd.yaml`` is one of the ``box``/``polytope`` documents above.

2. ``odd:`` block in a sampling config file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An ``odd:`` (or ``odd_config:``) key inside a full sampling config YAML is renamed to ``custom_odd_config`` on load (``autosafe.tools.monte_carlo._sample._load_sampling_config_file``), so it takes the same schema as the CLI flag, embedded alongside the box/anchor/ sample settings:

.. code-block:: yaml

    dim: 2
    odd_type: box
    odd_lower_limits: [-10, -10]
    odd_upper_limits: [10, 10]
    box_lower_limits: [-20, -20]
    box_upper_limits: [20, 20]
    odd_anchors: 20
    samples: 50
    filename: mc_odd_config_example-results.json
    odd:
      type: polytope
      dim: 2
      constraints:
        - type: linear
          coefficients: [1.0, -1.0]
          relation: ">="
          bound: 4.0

.. code-block:: console

    autosafe montecarlo sample --config-file sampling_config.yaml

This is the format used by ``experiments/dim_2d/sampling_config.yaml`` through ``experiments/dim_12d/sampling_config.yaml`` -- see :ref:`mc-canonical-examples`.

.. _mc-third-entry-point:

3. Programmatically
^^^^^^^^^^^^^^^^^^^^^

``custom_odd_config`` accepts either a YAML path string or a mapping, both via :func:`~autosafe.tools.monte_carlo._sample.create_config`'s ``odd_config`` parameter:

.. code-block:: python

    from autosafe.tools.monte_carlo._sample import create_config
    from autosafe.tools.monte_carlo.inequality_utils import (
        ODDFactory,
        validate_odd_config,
    )

    inline_config = {
        "type": "polytope",
        "dim": 2,
        "constraints": [
            {
                "type": "linear",
                "coefficients": [1.0, -1.0],
                "relation": ">=",
                "bound": 4.0,
            },
        ],
    }
    validate_odd_config(inline_config)  # raises on a malformed top-level shape

    mc_config = create_config(
        dim=2,
        odd_anchors=5,
        samples=10,
        odd_config=inline_config,  # a mapping; a "*.yaml" path string also works
        filename="mc_programmatic_example-results.json",
    )
    assert mc_config["custom_odd_config"] == inline_config

    # The same schema, used directly without the sampling pipeline:
    region, description = ODDFactory(inline_config).create_odd()
    print(description)

This last form -- calling ``ODDFactory`` directly -- is also the reliable way to get the constructed region *today*; see :ref:`mc-known-issue`.

.. _mc-canonical-examples:

Canonical examples: the paper's 2D-12D sweep
-----------------------------------------------

``experiments/dim_2d/sampling_config.yaml`` through ``experiments/dim_12d/sampling_config.yaml`` are the real configuration files that produced the paper's Monte Carlo results from two to twelve dimensions (``subsec:MonteCarloValidation``: "The experiments cover ODD structures from two to twelve dimensions").
They are run via the ``mc-sample-2d`` through ``mc-sample-12d`` entries in ``experiments/run_all_spec.yaml``, through ``autosafe experiments run-spec``.
Each file couples adjacent dimensions with one linear ontology predicate (:math:`x_1 - x_2 \geq 4`, :math:`x_2 - x_3 \geq 4`, ...) inside an ``odd:`` polytope block, layered on an outer box taxonomy that doubles in each dimension for the validation region (``box_lower_limits``/``box_upper_limits`` at :math:`\pm 20`) relative to the anchor region (``odd_lower_limits``/``odd_upper_limits`` at :math:`\pm 10`) -- the same box-doubling pattern as the paper's representative 2D figure (``fig:2D-MCM-ODD``).
Read these files directly for the exact, reproducible per-dimension setup rather than a restated copy here.

Errors you will hit
---------------------

.. list-table::
    :header-rows: 1
    :widths: 40 20 40

    * - Situation
      - Raised by
      - Exception
    * - Missing ``type`` or ``dim``
      - ``validate_odd_config``
      - ``ValueError``
    * - ``type`` not ``"box"``/``"polytope"``
      - ``validate_odd_config``
      - ``ValueError``
    * - ``polytope`` without a non-empty ``constraints`` list
      - ``validate_odd_config``
      - ``ValueError``
    * - ``box`` without ``lower_bounds``/``upper_bounds``
      - ``validate_odd_config``
      - ``ValueError``
    * - Constraint ``coefficients`` length :math:`\neq` ``dim``
      - ``ODDFactory.create_odd`` (via ``_normalize_constraint``)
      - ``ValueError``
    * - Constraint ``relation`` not one of ``<=``, ``<``, ``>=``, ``>``
      - ``ODDFactory.create_odd``
      - ``ValueError``
    * - Constraint ``type`` other than ``"linear"``
      - ``ODDFactory.create_odd``
      - ``ValueError``
    * - Constraint missing ``coefficients``/``relation``/``bound``
      - ``ODDFactory.create_odd``
      - ``ValueError``
    * - A ``constraints`` entry that is not a mapping
      - ``ODDFactory.create_odd``
      - ``TypeError``

``validate_odd_config`` only checks the *top-level* shape (``type``, ``dim``, and that ``constraints``/bounds are present in the right form); the per-constraint checks in the second half of the table are raised later, when ``ODDFactory.create_odd()`` normalizes each constraint.
A YAML file that passes ``load_yaml_odd_config`` (which calls ``validate_odd_config``) can still fail once a region is actually built from it.

.. code-block:: python

    from autosafe.tools.monte_carlo.inequality_utils import ODDFactory

    bad_config = {
        "type": "polytope",
        "dim": 2,
        "constraints": [
            {"coefficients": [1.0], "relation": "<=", "bound": 1.0},  # length 1, not 2
        ],
    }
    try:
        ODDFactory(bad_config).create_odd()
    except ValueError as exc:
        print(exc)  # "Coefficient dimension 1 does not match space dimension 2"

.. _mc-fixed-2026-09-02:

Fixed in 2026-09-02: custom ODD constraints are now applied
-----------------------------------------------------------

Until 2026-09-02 the constraints configured through any of the entry points above were built and stored correctly but never reached the sampled region.
``autosafe.tools.monte_carlo.sample._resolve_odd`` guarded the merge with ``hasattr(config, "custom_odd_config")``; ``config`` is a ``MonteCarloConfig``, which is a ``typing.TypedDict`` and therefore a plain ``dict`` at runtime, and a ``dict`` never exposes its keys as *attributes*.
The condition was always ``False``, so every ``type: polytope`` ODD silently fell back to the outer box.

The guard now tests the key itself.
A configured half-space adds one row to the taxonomy's four box rows, and points on its far side are correctly excluded:

.. code-block:: python

    import numpy as np
    import polytope as pc

    from autosafe.tools.monte_carlo.sample import _resolve_odd

    taxonomy = pc.Region([pc.box2poly(np.array([[-10.0, 10.0], [-10.0, 10.0]]))])
    config = {
        "odd_type": "box",
        "custom_odd_config": {
            "type": "polytope",
            "dim": 2,
            "constraints": [
                {
                    "type": "linear",
                    "coefficients": [1.0, -1.0],
                    "relation": ">=",
                    "bound": 4.0,
                },
            ],
        },
    }

    odd, description = _resolve_odd(config, taxonomy)

    assert len(odd.list_poly[0].A) == 5  # four box rows plus the half-space
    assert np.array([5.0, 0.0]) in odd  # x1 - x2 = 5 >= 4
    assert np.array([0.0, 5.0]) not in odd  # x1 - x2 = -5 < 4

Result artifacts produced before this fix carry only the box: their stored ``odd.list_poly[0].b`` has four rows and ``odd_description`` is ``null``, even though ``config.custom_odd_config`` is populated.
Regenerate any ``experiments/dim_*/`` result you intend to rely on.
The paper's reported two- to twelve-dimensional figures are not derived from this path -- they come from the benchmark suite in ``experiments/benchmark/``, which builds its ground-truth regions directly and never calls ``ODDFactory``.

A second defect fixed at the same time: ``ODDFactory.create_odd()``'s description string substituted the canonicalized (negated) coefficients back into the original relation symbol, so a ``>=`` constraint printed as the opposite half-space.
``x1 - x2 >= 4`` now describes itself as ``(1x1 -1x2) >= 4`` rather than ``(-1x1 1x2) >= -4``.
