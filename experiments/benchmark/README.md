<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
SPDX-License-Identifier: MIT
-->
# Benchmark experiments

Standalone experiments backing the paper's empirical claims.
They use the autoSAFE public API directly; the only autoSAFE-internal call is `Samples._find_closest_samples` (to assign each kernel's nearest neighbor), mirroring the production construction path.

Results (long-format `results.csv`, pgfplots-ready `.dat`, `config.json` with git hash and wall time) live in `experiments/benchmark/results/<exp>/`.
They are committed to this repository as produced, licensed CC-BY-4.0 (see `results/REUSE.toml`), and must not be regenerated in place.
Numbers quoted below are from those files.
`plots/` is not tracked in git (see `.gitignore` in this directory); plotting is TikZ/PGFPlots in the paper, so the matplotlib scripts under `plots/` are local debugging aids only.

## Provenance of the committed results

Each experiment's `config.json` (for permutation stability, `permutation_report.json`) records the `git_hash` of the code that produced it.
The anchor-count sweep was produced at `8db2a0a4`, the halo-vs-anchor-count, OOD-adjustment, and parameter-sensitivity experiments at `c5442046`, and all others at `89d350d4`.
Those commits belong to the pre-publication history and are not part of this repository.
On 2026-10-02 the committed tree was verified to rebuild all 18 paper data files exactly (see [Exporting the paper's data files](#exporting-the-papers-data-files)).
Code changes since then that can affect a re-run: the scripts now use `scipy.spatial.KDTree` (default `leafsize=10`) instead of `cKDTree` (`leafsize=16`), which can shift the kernel-truncation latencies.

## Running

Every script takes `--seed` (default 42), `--quick` (a tiny config that finishes in seconds, used by the smoke tests), and `--outdir`.
Invoke from the repo root with the module form (the package uses absolute imports):

```bash
uv run python -m experiments.benchmark.run_baseline_comparison --quick
uv run python -m experiments.benchmark.run_halo_vs_anchor_count   # full config
```

`run_all.py` runs all twelve in the fixed order below, with a resumable local state file (`run_all.state.local.json`, same completed/failed/timestamp contract as `ExperimentManager`):

```bash
uv run python -m experiments.benchmark.run_all --quick   # validate the path, seconds
uv run python -m experiments.benchmark.run_all           # full run, up to ~30 min per experiment
```

Smoke tests: `uv run pytest tests/benchmark/ -q`.

## The experiments

Each section states what the experiment does, the paper artifact it backs, and its outputs.

### Baseline comparison (`run_baseline_comparison.py`)

- **Backs:** the baseline AUPR and fixed-threshold tables.
- **What it does:** scores the full synthetic suite (`linear2d`, `annulus2d`, `twoblobs2d`, `banana2d`, `poly5d`; N ∈ {10, 100, 1000}; 10⁵ validation points on the 2× bbox) with autoSAFE (calibrated, γ=1, s=3) against KDE (best bandwidth by AUPR), OC-SVM, k-NN distance, Isolation Forest, GMM (BIC), and the convex hull.
  Order-independence: max score deviation over 5 anchor permutations — measured for autoSAFE only; the baselines carry NaN, so no non-determinism numbers may be quoted for them.
- **Headline (N=1000):** autoSAFE matches the strongest baselines on 2D (0.973–0.993 AUPR; best method within 0.014) and beats the hull decisively on non-convex cases (annulus 0.991 vs 0.901, twoblobs 0.987 vs 0.482).
  **Honest caveat:** on sparse 5D (`poly5d`) the default s=3 collapses to **0.079** while GMM/k-NN reach 0.85–0.88 — the same s-sensitivity the parameter-sensitivity experiment measures (s=1 recovers ≈0.91). autoSAFE permutation deviation ≤ 5.6e-15.
- **Outputs:** `aupr_vs_n_<dataset>.dat`, `fixed_zeta.dat`.

### OOD adjustment (`run_ood_adjustment.py`)

- **Backs:** the OOD handling section and its termination proposition (text only — no figure or table).
- **What it does:** first empirical exercise of `enforce_ood_consistency`: `linear2d`/`annulus2d`, N_ID=1000 (calibrated), M ∈ {10, 100, 1000} uniform OOD points on the 2× bbox, grid ξ ∈ {0.1, 0.3, 0.5} × c ∈ {0.5, 0.8, 0.9, 0.99}, ζ=0.6; records iterations, post-adjustment max OOD affinity, and collateral damage on ID precision/recall.
- **Headline:** the bound α ≤ ξ held in **all 72 configs** (max post-adjustment OOD affinity 0.49990 at ξ=0.5), empirically confirming the termination proposition.
  Iterations T ∈ [0, 28037]; worst case ξ=0.1, c=0.99 with M=1000 (295 kernels touched); across the grid a single kernel was rescaled at most 541×; per-c maxima of T: 462 / 1291 / 2697 / 28037 for c = 0.5 / 0.8 / 0.9 / 0.99. Collateral: ≤ 2.2 % of ID points dropped below ζ, recall −≤0.018, precision +≤0.102.
- **Outputs:** `ood_iterations.dat`, `ood_collateral.dat`.

### Parameter sensitivity (`run_parameter_sensitivity.py`)

- **Backs:** the parameter sensitivity figure.
- **What it does:** calibrated grid γ ∈ {0.5, 1, 2} × s ∈ {1, 2, 3, 5} plus m ∈ {1, 3, 5} on `linear2d`/`annulus2d`/`poly5d`, N ∈ {100, 1000}; secondary fixed-parameter grid κ × η ∈ {0.1, 0.5, 1, 2, 5}² on normalized `linear2d`.
  Metric: AUPR.
- **Headline:** 2D is robust (AUPR spread ≤ 2.5 % across the γ×s grid at N=1000); sparse 5D is s-dominated (s=1 → 0.87–0.91, s=3 → 0.05–0.16, s=5 → ≈0.02 ≈ prevalence — boundary over-inflation, the two-regime halo mechanism).
  The fixed grid never exceeds AUPR 0.60 on normalized data — no fixed (κ, η) is robust under normalization, supporting the calibration argument.
- **Outputs:** `sensitivity_<dataset>_n<N>.dat`, `sensitivity_fixed_linear2d.dat`.

### Halo vs. anchor count (`run_halo_vs_anchor_count.py`) — the headline experiment

- **Backs:** the halo-vs-N figure and the associated asymptotic propositions.
- **What it does:** `linear2d` on raw [−5,5]² (κ=η=1 on normalized data is the degenerate saturating case; the calibrated arms are scale-invariant, so raw units keep halo widths comparable).
  Arms: `fixed` (κ=η=1), `calibrated` (γ=1, s=3), `calibrated_sN` (s=√(ln N)); N ∈ {30, …, 30000}; halo width measured on a 50 000-point band outside the true boundary (signed distance is closed-form), ζ=0.5.
- **Headline (N = 30 → 3·10⁴):** fixed halo (max) **grows** 1.03 → 3.43, precision 0.90 → 0.33; calibrated halo **shrinks** to 0.10, precision → 0.97; the s_N arm matches the calibrated arm at large N (0.102 vs 0.101).
  `config.json` acceptance: `halo_trend_fixed > 0`, `halo_trend_calibrated* ≤ 0` — both hold.
- **Outputs:** `halo.dat`.

### Anchor count sweep (`run_anchor_count_sweep.py`)

- **Backs:** the Monte Carlo per-N precision/recall figure and its curve-R² validation.
- **What it does:** paper-exact MCM config — `linear2d`, fixed κ=η=1, λ_rel=e⁻¹⁰, raw [−5,5]², no normalization/calibration, validation 10⁵ points uniform on [−10,10]²; N ∈ {3, …, 10⁴}, 5 seeds.
  Curve-R² between the ODD-referenced and hull-referenced precision (resp. recall) curves as ζ sweeps, with the ODD curve as the regression target.
- **Headline (evaluated with the stable log-space score):** precision R² = 0.971 (N=10³), 0.997 (3·10³), 0.99965 (10⁴).
  Recall R² is 1.0 for N ≥ 10³, but is not informative there because both recall curves are nearly constant; report this degeneracy with the value.
  Mean AUPR at N=5/25/50 is 0.734±0.056 / 0.903±0.020 / 0.949±0.012.
- **Outputs:** `pr_n<N>.dat`, `aupr_vs_n.dat`, `curve_r2.dat`.

### Kernel truncation (`run_kernel_truncation.py`)

- **Backs:** the truncation scalability figure and table (Appendix H).
- **What it does:** full `affinity_dual` vs. K-nearest-anchor truncation (reuses the SAME per-anchor diagonal covariance) on `poly5d`, N ∈ {1e3, 1e4, 1e5} (+`--huge` for 6e5), K ∈ {8, 32, 128, 512}; error max/mean/p99, ζ-flip rate, query latency + memory, and exact-vs-approximate (FAISS-IVF) nearest-neighbor determinism.
- **Headline:** error/flip decrease monotonically with K; the truncated affinity is bit-exact vs. full at K=N; exact nearest-neighbor search is bit-reproducible; the approximate index flips 4 % of memberships at nprobe=1, 0 % at nprobe=8 — the reason certification mode uses exact search.
- **Outputs:** `truncation_error.dat`, `truncation_latency.dat`, `truncation_determinism.json`.

### Permutation stability (`run_permutation_stability.py`)

- **Backs:** the determinism / order-independence claim (text only — no figure or table).
- **What it does:** full pipeline including `enforce_ood_consistency` on `annulus2d` (N_ID=1000, M_OOD=200), 20 simultaneous ID/OOD permutations; σ diagonals compared keyed by anchor coordinate (indices change under permutation), α compared on a fixed 10⁴-point grid.
- **Headline:** all 20 permutations pass without any canonical anchor sort: σ parameters bit-identical (max deviation 0.0), affinities within a few ulps (max deviation 1.67e-15, ~7 ulp — the residual is non-associative floating-point summation in the batched path).
  No library default was changed.
- **Outputs:** `permutation_report.json`.

### Held-out VCAS hole (`run_held_out_vcas_hole.py`)

- **Backs:** the held-out hole false-positive table.
- **What it does:** `--source auto` picks the real VCAS anchor set when `data/vcas_state_variables.csv` exists (default `n_id=20000` anchor subsample, hole = `h∈[−300,300] ∧ τ∈[0,8]` as an excluded near-CPA regime, constant `hdot_int` dropped → 4 effective dims), else a `poly5d` synthetic fallback; `--quick` always forces synthetic so the smoke tests stay hermetic.
  Compares (i) autoSAFE without OOD, (ii) autoSAFE with the removed in-hole points as OOD samples, (iii) convex hull.
  The false-positive-in-hole metric is computed on a held-out set of real in-hole points (disjoint from the OOD set): uniform corner points would understate the hull's leakage because the VCAS mass is concentrated and its min/max box has empty corners.
- **Headline (VCAS-with-hole):** false-positive rate inside the hole: convex hull **0.547**, autoSAFE without OOD **0.037**, with OOD **0.0115**.
  The AUPR column is low for *all* methods (box ground truth, ~6 % prevalence) — the headline metric is `fp_rate_in_hole`, not AUPR.
- **Outputs:** `hole_fp.dat`, `hole_fixed_zeta.dat`.

### Conformal threshold (`run_conformal_threshold.py`)

- **Backs:** the threshold calibration subsection.
- **What it does:** `ζ̂ = ε(n+1)-th smallest ID calibration affinity`, guaranteeing `P(α(X_test) < ζ̂) ≤ ε`; measures empirical ID false-exclusion on a disjoint test set and specificity on true-outside points, over datasets/N/ε, 20 repeats.
- **Headline:** `linear2d` at ε=0.1 gives ζ̂≈0.985, specificity_out 0.993 (non-trivial).
- **Outputs:** `conformal_coverage.dat`, `zeta_vs_eps_<dataset>_n<N>.dat`.

### Duplicate sensitivity (`run_duplicate_sensitivity.py`)

- **Backs:** the duplicate sensitivity figure.
- **What it does:** (A) calibrated density → convergence (IoU vs. truth), (B) duplicate injection with IoU-vs-baseline for fixed vs. calibrated plus a controlled check of the noisy-OR formula `a → 1−(1−a)^m`, (C) a dedup remedy.
- **Key finding:** the check reproduces `α→1−(1−a)^m` exactly (needs a fixed-width kernel to isolate it); autoSAFE actually has two duplicate responses — the noisy-OR inflation and the σ-law reacting to `d_i*→0`.
  Density-adaptive calibration plus dedup-at-ingestion is the answer, not naive weighting.
- **Outputs:** `duplicate.dat`, `dedup.dat`, `formula_check.dat`, `density_stability.dat`.

### Anchor subset stability (`run_anchor_subset_stability.py`)

- **Backs:** the subset stability figure.
- **What it does:** from one fixed master pool of ID points, draws K independent subsamples at each size N, rebuilds the calibrated ODD per draw (including its own normalizer — the deployed pipeline), and measures AUPR and level-set IoU vs. ground truth (mean ± std over draws) plus the mean/min/max pairwise IoU between the K level sets at the same size (selection stability at fixed size).
- **Outputs:** `subset_stability.dat`.

### Covariance structure (`run_covariance_structure.py`)

- **Backs:** the diagonal-covariance appendix.
- **What it does:** a `corr2d(rho)` rotated band plus an iso/diag/full ablation built from the same k-NN local covariance and s² scaling (isolates covariance *structure*, not the sigma-law's per-anchor width); AUPR, IoU, and cross-band halo vs. N and ρ, plus a `deployed` arm that is the real sigma-law kernel via `build_odd(mode="calibrated")`.
- **Headline:** at ρ=0 the isotropic and full arms agree; the full-vs-iso IoU gap grows with ρ, and `deployed` tracks the diagonal arm as expected (the deployed kernel is diagonal, not isotropic).
- **Outputs:** `anisotropy.dat`, `gap_vs_rho.dat`.

## Configuration notes (read before citing numbers)

- **Coordinate frames differ by experiment, on purpose.**
  - The anchor-count-sweep and halo-vs-anchor-count experiments use raw `[−5,5]²` coordinates with the fixed `κ=η=1` parameters to match the paper's Monte Carlo Figure 2 and show the `√(ln N)` halo growth.
    `κ=η=1` on *normalized* data is the degenerate saturating case, so those two experiments deliberately skip normalization.
  - The baseline-comparison, OOD-adjustment, parameter-sensitivity, permutation-stability, and held-out-VCAS-hole experiments normalize per-dimension to `[−1, 1]` (min–max on ID points) and use the calibrated parameters.
- **λ floor is the camera-ready relative one everywhere.**
  `build_odd("fixed")` uses `λ = λ_rel·κ = e⁻¹⁰` (with κ=1) — not an absolute floor.
  Regenerated Figure-2 numbers therefore reflect this relative floor.
- **Ground-truth membership is `R ∩ X`** (ontology AND taxonomy box), enforced in `synthetic_odds.SyntheticODD.contains`.
  Validation points on the 2× box that satisfy `R` but fall outside `X` are correctly labeled *not* in the ODD.
- **The anchor-count-sweep curve-R²** is between the ODD-referenced and convex-hull-referenced precision (resp. recall) curves as ζ sweeps.
  The stable-score evaluation avoids saturated linear-affinity ties: precision R² is 0.971 at `N=1000` and 0.99965 at `N=10⁴`; recall R² is numerically 1.0 at `N≥1000` but degenerate because both recall curves are nearly constant.
- **Halo-vs-anchor-count acceptance:** `config.json` reports `halo_trend_fixed` (should be `> 0`) and `halo_trend_calibrated*` (should be `≤ 0`).
  At N = 30 → 3·10⁴: fixed halo (max) 1.03 → 3.43, calibrated 3.49 → 0.10.
- **Permutation stability** passes without the optional `canonical_order` sort, so no library default was changed. σ parameters are bit-identical under input permutations; affinities are identical to within a few units in the last place (`max_alpha_dev ≈ 1.7e-15`).

## Exporting the paper's data files

`export_paper_data.py` reconstructs the eighteen `.dat` files behind the paper's benchmark figures and tables from `results/` (pivots, joins, and the row filters and label shortenings documented next to each recipe in the script).

```bash
uv run python -m experiments.benchmark.export_paper_data --outdir /tmp/paper-data
```

On 2026-09-02 and again on 2026-10-02 (from the committed results) the output was checked cell by cell against the data files used in the paper: 18/18 match.
`tests/benchmark/test_results_tree.py` checks on every test run that all 18 files rebuild from the committed results.

## Plotting (`plots/`, PEP 723)

Self-contained matplotlib scripts (inline `# /// script` deps), one per experiment that produces a figure (permutation stability and anchor-subset-stability do not have plotting scripts — the former is a scalar report, the latter is table-only).
Run e.g. `uv run experiments/benchmark/plots/plot_kernel_truncation.py`; each reads `results/<exp>/*.dat` and writes PNGs alongside them, which git ignores.
Re-run after the full experiment runs.
These are local debugging aids, not published artifacts — all paper figures are TikZ/PGFPlots.

## Approximate runtimes (desktop CPU)

`--quick` configs: seconds each.
Full configs target `< 30 min` each; the baseline-comparison and anchor-count-sweep experiments at the largest N and validation counts are the heaviest.
Full-size HCAS/VCAS datasets are **not** run here.
