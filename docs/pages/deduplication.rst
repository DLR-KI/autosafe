.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

De-duplication
===============

Resolution-cell de-duplication (``src/autosafe/deduplication.py``) removes repeated visits to the same acquisition cell from the anchor pool before the ODD is fit.

What it is, and why
--------------------

The recovered ODD represents *support*: which regions of state space the system actually observed operating safely, not how often each region was revisited.
A sensor logging at a fixed rate parked at a red light contributes thousands of near-identical rows for one operating condition.
Left in the anchor pool, that multiplicity does not add support -- it just increases local sampling density, and the noisy-OR affinity construction lets sampling density distort the recovered geometry: a region visited often looks more "in-distribution" than one visited once, even when both are equally valid operating conditions.

De-duplication groups anchor candidates into resolution cells and keeps one observed representative per cell.
Repeated visits are retained as *provenance* -- the per-cell multiplicity and full list of source-record ids are written to a side artifact (see `Provenance artifact`_ below) -- but never as kernel weight.
A cell visited 1000 times and a cell visited once contribute the same one anchor to the fitted ODD.

Disabled by default
--------------------

**De-duplication ships disabled by default.** ``evaluate_dataset_mode``, ``from_csv``/``from_polars``/``from_numpy``, and the ``dataset`` spec mode all take an optional policy argument that defaults to ``None`` (no policy / ``dedup_resolution`` absent).
With no policy, none of ``src/autosafe/deduplication.py`` or ``src/autosafe/tools/evaluate/dataset/dedup_integration.py`` is even imported, and the existing pipeline runs byte-identically to before R3.

This matters because **enabling de-duplication changes the anchor set**, and therefore any previously computed result: fewer anchors, different kernel placements, a different fitted ODD, different cache files.
Turning it on for a dataset that has already been evaluated is not a metadata-only change -- it invalidates the comparison to prior numbers for that dataset unless both are re-run under the same (disabled or enabled) setting.
See `Cache implications`_ below for how the pipeline prevents a stale cache from masking this silently.

The policy is a documented assurance input, not a hyperparameter
------------------------------------------------------------------

``resolution`` and ``origin`` describe the *acquisition system's* grid -- sensor quantization step, timestamp granularity, or another recorded acquisition resolution from the source system's metadata.
They are not free parameters to search over:

- Never choose or tune ``resolution``/``origin`` against evaluation metrics (precision, recall, coverage, or anything else computed downstream).
  Doing so would let the de-duplication grid itself become an unaudited lever on the reported numbers.
- Document where a chosen ``resolution`` comes from (the sensor spec, the logging interval, etc.) alongside any result that used it.
- When no coarser acquisition resolution is justified for a dataset, use :func:`~autosafe.deduplication.exact_equality_policy`, which builds a policy with a cell width far below any real sensor's precision (default ``epsilon=1e-12``).
  It only ever merges records that are exact duplicates (up to floating-point/round-tripping noise), so it is always a safe, conservative default that never invents a coarser grid than the data already has.

The deterministic rule
------------------------

Fixed by the paper's Algorithm 1, not a design choice made in this module:

- **Cell assignment**: componentwise ``floor((x - o) / q)`` for the resolution vector ``q`` and grid origin ``o`` (the source system's own quantizer mapping, when available, in place of this formula).
- **Representative selection**: the cell's representative is an *actually observed* member minimizing the dimensionless resolution-scaled distance ``‖(x - mean) / q‖₂`` -- never the synthetic cell mean itself.
  The representative is always a row that was really recorded.
- **Tie-break**: ties in that minimization break lexicographically, first by coordinates, then by a stable source-record identifier.
- Cell means are computed in canonical (lexicographically sorted) member order, in float64.

Because every step of grouping, representative selection, and tie-breaking is defined in terms of a canonical sort key that does not depend on input row position, the output (representatives, their order, multiplicities, and provenance groups) is **order-independent**: permuting the input rows never changes the result.

Pipeline placement
--------------------

Within ``evaluate_dataset_mode`` (see ``src/autosafe/tools/evaluate/dataset/dedup_integration.py``, :func:`~autosafe.tools.evaluate.dataset.dedup_integration.run_dataset_deduplication`), the fixed order is:

1. Load raw data with stable record ids (the row index).
2. **Reserve split-conformal calibration records** -- the last ``dedup_n_calibration_reserved`` rows by row index are held out *before* de-duplication and never de-duplicated or instantiated as kernels.
   They keep their original observed frequency, because a calibration set's job is to reflect the true operational distribution, not the deduplicated support set.
3. Subtract OOD rows (exact match, in raw coordinates) from the remaining ID candidates.
4. De-duplicate the ID candidates and, independently, the OOD set, under the same policy.
5. Fit (or apply, if given externally) the normalizer on the retained ID representatives only -- fitting on the full, undeduplicated pool would let sampling density back into the geometry through the normalization statistics.
6. Re-check ID/OOD disjointness in the normalized space used by the OOD consistency loop.
7. **Anchor subsampling happens after de-duplication**: ``subsample_anchors`` is applied to the de-duplicated representative set inside ``_build_or_load_affinity_odd``, not to the raw candidate pool.
   If subsampling ran first, it would draw from a pool whose density still reflected coverage rather than support.

Provenance artifact
----------------------

:func:`~autosafe.tools.evaluate.dataset.dedup_integration.write_dedup_provenance` writes two files beside the dataset, both named with the same ``-dedup<digest>`` tag as the cache (see `Cache implications`_):

- ``<dataset>-dedup-provenance<tag>.parquet``: one row per (cell, source-record), for every cell and every reserved calibration record.
  Columns are ``label`` (``"id"``, ``"ood"``, or ``"calibration"``), ``cell_index`` (``-1`` for calibration rows, which are not de-duplicated), ``member_record_id``, ``member_rank`` (position within the cell's canonical member order), and ``is_representative``.
  This is the full group membership -- potentially large -- so it is kept out of the ODD JSON entirely.
- ``<dataset>-dedup-summary<tag>.json``: a small human-inspectable summary -- the resolved policy, ``id_n_input``/``id_n_output``/ ``id_n_duplicates``, the cell-multiplicity histogram (``id_cell_count_distribution``), input/output digests, the OOD result's input/output counts (``null`` when no OOD file was given), the label conflict count and a sample of conflicting cells, the calibration-reserved count, and the Parquet file's own path and SHA-256 digest.

Cache implications
---------------------

De-duplication changes the anchor set, but neither of the two existing cache-identity checks would notice that on their own: the nearest-neighbor index loader validates only mode and shape, and the ODD cache spec (``_odd_matches_cache_spec``) compares only kernel settings, never anchors.
Without an explicit tag, a cache built from the *undeduplicated* pool could be silently reused for a de-duplicated run (or vice versa), producing results that silently mix two different anchor sets.

To prevent that, every de-duplication-enabled run adds a ``-dedup<digest>`` tag to the ODD cache path and the neighbor-index cache path.
The tag (``_dedup_filename_tag`` in ``src/autosafe/tools/evaluate/dataset/odd_cache.py``) is a SHA-256 digest (first 8 hex characters) of the de-duplication algorithm version, the fully resolved policy, and the retained representative coordinates -- the same ``-ex<digest>``/``-ood<digest>`` precedent already used for OOD exclusion tagging (see ``docs/ood-consistency.md``).
A different policy, or the same policy applied to different underlying data, always resolves to a different cache file.

ID/OOD label conflicts
--------------------------

Two distinct checks guard against ID and OOD anchors disagreeing about the same region of state space:

- **Exact coordinate coincidence** between an ID and an OOD representative is always a hard error (:class:`~autosafe.exceptions.OODAnchorCoincidenceError`), raised inside ``run_dataset_deduplication`` after normalization.
  It is not specific to de-duplication -- an anchor's affinity is 1 for any covariance, so the OOD consistency loop could never converge if an OOD point sat exactly on an anchor.
- **Resolution-cell co-membership** without exact coincidence (:func:`~autosafe.deduplication.check_label_conflicts`) is weaker evidence: it means an ID and an OOD point were close enough to land in the same acquisition cell, which may or may not indicate an actual labeling problem.
  By default this is only *reported* -- the conflicting cell ids are recorded in the provenance summary (``label_conflicts_count``/``label_conflicts_sample``) for a data owner to adjudicate -- and only raises (:class:`~autosafe.exceptions.DeduplicationLabelConflictError`) when ``dedup_strict_label_conflict``/``strict_label_conflict`` is explicitly set.

Worked API example
----------------------

The low-level API operates on plain arrays and is independent of any kernel, dataset loader, or CLI.
``data/iris.csv`` genuinely contains three exact-duplicate rows, so :func:`~autosafe.deduplication.exact_equality_policy` collapses them without any invented resolution:

.. code-block:: python

    import polars as pl

    from autosafe.deduplication import deduplicate_points, exact_equality_policy

    points = pl.read_csv("data/iris.csv").to_numpy().astype(float)
    policy = exact_equality_policy(n_dims=points.shape[1])
    result = deduplicate_points(points, policy)

    result.n_input                    # 150
    result.n_output                   # 147
    result.n_duplicates               # 3
    result.cell_count_distribution()  # {1: 145, 2: 1, 3: 1}

``from_csv``/``from_polars``/``from_numpy`` accept the same policy directly, producing an ODD fit on the de-duplicated representatives:

.. code-block:: python

    import autosafe as af

    odd_plain = af.from_csv("data/iris.csv")
    odd_dedup = af.from_csv("data/iris.csv", dedup_policy=policy)

    len(odd_plain.samples)  # 150
    len(odd_dedup.samples)  # 147

YAML spec keys
------------------

The ``dataset`` spec mode (``autosafe experiments run-spec``) resolves the same policy from spec-item keys, via ``_resolve_dedup_policy`` in ``src/autosafe/cli/experiments.py``.
De-duplication stays off unless ``dedup_resolution`` is present:

.. code-block:: yaml

    experiments:
      - id: iris-dedup-demo
        mode: dataset
        dataset_path: data/iris.csv
        evaluation_samples: 200
        closest_sample_mode: global
        kernel_type: RBF
        csv_output: iris-dedup-demo.csv
        odd_json_out: iris-dedup-demo-odd.json
        dedup_resolution: [1e-12, 1e-12, 1e-12, 1e-12]
        dedup_origin: [0.0, 0.0, 0.0, 0.0]
        dedup_coordinate_space: raw

Running this spec against ``data/iris.csv`` reproduces the same 150 -> 147 collapse as the API example above, and additionally writes the ``-dedup<digest>``-tagged provenance Parquet/JSON pair beside the dataset.
The full set of de-duplication spec keys:

- ``dedup_resolution`` (list of floats, required to enable de-duplication): per-dimension cell width ``q``.
- ``dedup_origin`` (list of floats, defaults to all zeros): grid origin ``o``.
- ``dedup_coordinate_space`` (``"raw"`` or ``"normalized"``, defaults to ``"raw"``): the coordinate system ``resolution``/``origin`` are expressed in.
- ``dedup_n_calibration_reserved`` (int, defaults to ``0``): rows reserved for split-conformal calibration before de-duplication (see `Pipeline placement`_).
- ``dedup_strict_label_conflict`` (bool, defaults to ``false``): raise instead of only reporting ID/OOD resolution-cell co-membership (see `ID/OOD label conflicts`_).

``representative_metric`` and ``tie_break`` are not exposed as spec keys: they are fixed by the paper's Algorithm 1 (see `The deterministic rule`_) and are not meant to vary per run.
