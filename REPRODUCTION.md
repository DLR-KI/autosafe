<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>

SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Reproduction

This document reproduces every quantitative figure and table in the paper from this repository.
It covers the environment, the committed data, the archive of pre-computed results (the fast path), the two commands that regenerate those results from scratch (the slow path), a per-figure/table mapping down to the exact artifact and command, seeds/configurations, and a pointer to the de-duplication rules.

## 1. Environment

```shell
uv sync --frozen
```

This creates a virtual environment from the committed `uv.lock`, so the exact dependency versions the results were produced with are reproduced byte-for-byte (`uv.lock` is also included in the archive bundle, see below).

## 2. Data

Every dataset the paper's experiments read is committed to this repository, with its licence recorded in `data/REUSE.toml`:

| File                                                             | What it is                                                                          | Licence         |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------- |
| `data/iris.csv`                                                  | Fisher's Iris dataset                                                               | CC0-1.0         |
| `data/WineQT.csv`                                                | Wine quality dataset (Kaggle)                                                       | CC0-1.0         |
| `data/breast-cancer-wisconsin.csv`                               | Breast Cancer Wisconsin dataset                                                     | CC-BY-NC-SA-4.0 |
| `data/hcas_state_variables.csv`                                  | HCAS state-variable samples                                                         | CC-BY-4.0       |
| `data/vcas_state_variables.csv`                                  | VCAS state-variable samples                                                         | CC-BY-4.0       |
| `data/vcas_ood_hole.csv`                                         | VCAS anchors inside the excluded near-CPA region (the OOD-consistency held-out set) | CC-BY-4.0       |
| `data/hcas_state_variables.yml`, `data/vcas_state_variables.yml` | Ground-truth ODD limit definitions for HCAS/VCAS                                    | CC-BY-4.0       |

`iris.csv`, `WineQT.csv`, and `breast-cancer-wisconsin.csv` back the working-example spec items (`eval-iris`, `eval-WineQT`) that are **not** used in the paper -- see the run-spec comment header in `experiments/run_all_spec.yaml`.

Run `uv run reuse lint` at any time to re-verify every licence annotation.

## 3. Fast path: the results archive (recommended)

The archive record for the benchmark results is:

> **DOI: TBD** -- the Zenodo record does not exist yet; this placeholder will be replaced once the archive is published (upload is a manual, post-review step -- see the project plan).

Once the DOI exists:

1. Download the archive and unpack it. It contains the results tree (minus debug PNGs), the merged run spec, `uv.lock`, a `LICENSE` (CC-BY-4.0), and a `README` naming the exact repository revision the results came from (`experiments/benchmark/make_archive_bundle.py` builds this bundle).
2. Point the exporter at the unpacked `results/` directory:

    ```shell
    uv run python -m experiments.benchmark.export_paper_data \
        --outdir /tmp/paper-data --results-dir /path/to/unpacked/results
    ```

3. Compare against the camera-ready copies used in the paper (optional, read-only):

    ```shell
    uv run python -m experiments.benchmark.export_paper_data \
        --outdir /tmp/paper-data --verify paper/graphics/data/benchmark
    ```

    This reports a numeric, cell-by-cell match count (18/18 as of this writing) and exits non-zero on any real difference; formatting differences (the camera-ready files were partly hand-typed with inconsistent zero-padding) are ignored by design.

### The slow path, stated honestly

Without the archive, every result must be regenerated locally:

- All twelve synthetic benchmark experiments (`experiments/benchmark/run_all.py`, see command 1 below): **up to ~30 minutes each** at full size.
- The real-data (HCAS/VCAS) evaluations in `experiments/run_all_spec.yaml` (command 2 below), specifically `eval-vcas-rbf`, `eval-hcas-rbf`, and the OOD/subsample items: **on the order of a day each** (622k anchors, global closest-sample mode).
  These are not something to run in a dev session or CI; they are Johann's to run.

## 4. The two run commands

**1. The synthetic benchmark suite** (twelve experiments backing every figure/table not tied to real aviation data):

```shell
# Validate the whole path in seconds:
uv run python -m experiments.benchmark.run_all --quick

# Full-size run (up to ~30 min per experiment; do not run in CI):
uv run python -m experiments.benchmark.run_all --seed 42
```

This writes to `experiments/benchmark/results/` and tracks progress in `experiments/benchmark/run_all.state.local.json` (resumable: a rerun skips anything already completed, matching the contract in point 2 below).

**2. The real-data (HCAS/VCAS) evaluations**, via the experiment manager:

```shell
uv run autosafe experiments run-spec experiments/run_all_spec.yaml
```

This writes to `experiments/run_all_spec.state.local.json`, tracking `completed` ids, `failed` ids with their error message, and an update timestamp; a rerun resumes rather than repeating completed work.
Both state files are local-only (the root `.gitignore` `*.json` rule covers them; the `.local` suffix is documentation, not a mechanism) -- do not expect them to appear in git status.

## 5. Figure / table -> artifact -> command

Every entry below ends at the synthetic-benchmark exporter unless noted otherwise (the two real-data aviation tables end at the manager's own `results.csv`/`config.json`, since they do not go through `experiments/benchmark/`).
Purely illustrative figures with no underlying data (domain-geometry diagrams, the Monte Carlo setup sketch) are omitted.

| Figure / Table                                                      | Paper data artifact                                                                             | Results-tree source                                                                  | Command                                            |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------- |
| Fig. MCM-PerN (precision/recall vs. anchor count)                   | `mcm_per_n.dat`                                                                                 | `anchor_count_sweep/pr_n<N>.dat` (eleven values of N)                                | `run_all` -> `anchor_count_sweep` -> exporter      |
| (aggregate precision/recall averaged over N, by zeta)               | `mcm_anchor_count_mean.dat`                                                                     | derived from `mcm_per_n.dat`                                                         | exporter (no separate script)                      |
| Fig. HaloAnchorCount (halo + precision vs. N)                       | `halo_wide.dat`                                                                                 | `halo_vs_anchor_count/halo.dat`                                                      | `run_all` -> `halo_vs_anchor_count` -> exporter    |
| (halo audit table, verbatim)                                        | `halo.dat`                                                                                      | `halo_vs_anchor_count/halo.dat`                                                      | `run_all` -> `halo_vs_anchor_count` -> exporter    |
| Tab. BaselineAUPR                                                   | `baselines.dat`                                                                                 | `baseline_comparison/aupr_vs_n_<ds>.dat`                                             | `run_all` -> `baseline_comparison` -> exporter     |
| Tab. BaselineFixedThreshold                                         | `fixed_zeta.dat` (results tree only; not one of the 18 exported files)                          | `baseline_comparison/fixed_zeta.dat`                                                 | `run_all` -> `baseline_comparison`                 |
| Fig. PerNValidation (AUPR + precision-curve agreement vs. N)        | `per_n.dat`                                                                                     | `anchor_count_sweep/aupr_vs_n.dat` + `curve_r2.dat`                                  | `run_all` -> `anchor_count_sweep` -> exporter      |
| Fig. ParameterSensitivity (linear/annulus/5D heatmaps)              | `sensitivity_linear_n1000.dat`, `sensitivity_annulus_n1000.dat`, `sensitivity_poly5d_n1000.dat` | `parameter_sensitivity/sensitivity_<ds>_n1000.dat`                                   | `run_all` -> `parameter_sensitivity` -> exporter   |
| Tab. HeldOutHole (VCAS held-out false-positive rate)                | `hole.dat`                                                                                      | `held_out_vcas_hole/hole_fp.dat`                                                     | `run_all` -> `held_out_vcas_hole` -> exporter      |
| Fig. DuplicateSensitivity (density + de-duplication IoU)            | `density.dat`, `dedup.dat`                                                                      | `duplicate_sensitivity/density_stability.dat`, `duplicate_sensitivity/dedup.dat`     | `run_all` -> `duplicate_sensitivity` -> exporter   |
| Fig. SubsetStability                                                | `subset.dat`                                                                                    | `anchor_subset_stability/subset_stability.dat`                                       | `run_all` -> `anchor_subset_stability` -> exporter |
| Fig. CovarianceStructure (probe + deployed vs. N)                   | `covariance_n100.dat`, `deployed_covariance.dat`                                                | `covariance_structure/anisotropy.dat`                                                | `run_all` -> `covariance_structure` -> exporter    |
| Fig. TruncationScalability (accuracy + runtime)                     | `truncation_error.dat`, `truncation_latency.dat`                                                | `kernel_truncation/truncation_error.dat`, `kernel_truncation/truncation_latency.dat` | `run_all` -> `kernel_truncation` -> exporter       |
| Conformal calibration discussion (false-exclusion rate vs. eps)     | `conformal_n1000.dat`                                                                           | `conformal_threshold/conformal_coverage.dat`                                         | `run_all` -> `conformal_threshold` -> exporter     |
| OOD-consistency termination discussion (text only, no figure/table) | --                                                                                              | `ood_adjustment/ood_iterations.dat`, `ood_adjustment/ood_collateral.dat`             | `run_all` -> `ood_adjustment`                      |
| Permutation-stability discussion (text only, no figure/table)       | --                                                                                              | `permutation_stability/permutation_report.json`                                      | `run_all` -> `permutation_stability`               |
| Tab. AviationReferenceAgreement                                     | -- (real data)                                                                                  | `eval-vcas-rbf` / `eval-hcas-rbf` `results.csv`                                      | `autosafe experiments run-spec`                    |
| Tab. AviationSubsetStability                                        | -- (real data)                                                                                  | `eval-vcas-rbf-sub5000-s1` / `-s2` `results.csv`                                     | `autosafe experiments run-spec`                    |
| Fig. VCAS-PR (VCAS precision/recall curves)                         | -- (real data)                                                                                  | `eval-vcas-rbf` `results.csv`                                                        | `autosafe experiments run-spec`                    |

## 6. Seeds and configurations

Every experiment's exact seed and configuration is recorded next to its output: `experiments/benchmark/results/<experiment>/config.json` for the twelve synthetic experiments (also included in the archive bundle), and the corresponding entry in `experiments/run_all_spec.yaml` plus each run's `results.csv`/manager output for the real-data evaluations.
`run_all.py`'s own `--seed` (default 42) is forwarded to every synthetic experiment; the real-data spec items pin their own `seed` field per entry (see `eval-vcas-rbf-ood`, `eval-vcas-rbf-sub5000-s1`, `-s2`).

## 7. De-duplication and canonical ordering

Anchor de-duplication ships disabled by default, so every cache key, anchor set, and published number above reproduces byte-identically regardless of whether de-duplication is available.
Once implemented, its policy type, result type, and pure function live in `src/autosafe/deduplication.py` -- see that module directly for its API; this document does not duplicate it.
