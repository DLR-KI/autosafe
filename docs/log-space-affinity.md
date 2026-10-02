<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>

SPDX-License-Identifier: CC-BY-SA-4.0
-->
# Log-Space Evaluation of the autoSAFE Affinity

## 1. Definitions

For anchors $x_1, \dots, x_N$ with RBF kernels $k_i(x) = \exp(-\tfrac12 m_i(x))$, $m_i(x) = (x - x_i)^\top \Sigma_i^{-1} (x - x_i)$, the affinity and its **log-survival** are

$$
\alpha(x) = 1 - \prod_{i=1}^N \big(1 - k_i(x)\big),
\qquad
S(x) := \log\big(1 - \alpha(x)\big) = \sum_{i=1}^N \log\big(1 - k_i(x)\big) \;\le\; 0 .
$$

$S$ and $\alpha$ carry identical information: $\alpha = 1 - e^{S}$, and $t \mapsto \log(1 - t)$ is a strictly decreasing bijection $[0, 1] \to [-\infty, 0]$.
The decision rule transforms accordingly:

**Proposition 1 (threshold equivalence).**
For every $\zeta \in [0, 1]$,
$$
\alpha(x) \ge \zeta \;\Longleftrightarrow\; S(x) \le \log(1 - \zeta),
$$
with the convention $\log 0 = -\infty$.
In particular, at $\zeta = 1$ the rule becomes $S(x) = -\infty$, which holds iff $k_i(x) = 1$ for some $i$, i.e. iff $x$ coincides exactly with an anchor.
*Proof:* monotonicity of $\log(1 - \cdot)$; the $\zeta = 1$ case from $\alpha = 1 \Leftrightarrow \prod_i (1 - k_i) = 0 \Leftrightarrow \exists i: k_i = 1$.
$\square$

This matches the intended semantics of the threshold sweep: at $\zeta = 1$, no sampled point should be classified inside the ODD except an (almost surely impossible) exact anchor hit.

## 2. Failure modes of the linear-space evaluation in IEEE-754 double precision

Let $\varepsilon = 2^{-53} \approx 1.11 \times 10^{-16}$ (unit roundoff) and let $\operatorname{fl}(\cdot)$ denote evaluation in double precision.
The direct evaluation $\alpha = 1 - \prod_i (1 - k_i)$ fails in three distinct ways:

- **(F1) Absorption.**
  If $k_i < \varepsilon$, then $\operatorname{fl}(1 - k_i) = 1$: the contribution of anchor $i$ is silently dropped.
  Individually negligible contributions are discarded *before* they can accumulate.
- **(F2) Product underflow.**
  If $\prod_i (1 - k_i) < 2^{-1074}$ (smallest subnormal), the product flushes to $0$ and $\operatorname{fl}(\alpha) = 1$ exactly.
  Since $\prod_i (1 - k_i) \le e^{-K}$ with $K = \sum_i k_i$, underflow is guaranteed once $K \gtrsim 745$.
- **(F3) Saturation at the rounding step.**
  Far earlier than (F2): if $P := \prod_i (1 - k_i) < 2^{-54}$, then $\operatorname{fl}(1 - P) = 1$.
  By $P \le e^{-K}$ this is triggered already for $K \gtrsim 54 \ln 2 \approx 37.4$.

Consequence: whenever $K(x) \gtrsim 37$, the stored affinity is **exactly** $1.0$ and the decision $\operatorname{fl}(\alpha) \ge \zeta$ is constant in $\zeta$ on all of $[0, 1]$---including $\zeta = 1$.
For the HCAS dataset ($N \approx 5.8 \times 10^5$, measured $K \approx 10^4$ across the evaluation region) *every* test point saturates, which reproduces the observed constant confusion matrices.
Note that $1 - \zeta$ itself is computed exactly for $\zeta \in [0.5, 1]$ (Sterbenz's lemma), so the loss of information is entirely on the affinity side.

## 3. The log-space reformulation

Evaluate $S$ instead of $\alpha$:

$$
S(x) = \sum_{i=1}^N \log\big(1 - e^{-z_i}\big), \qquad z_i = \tfrac12 m_i(x) \ge 0 .
$$

### 3.1 Per-term stability

The naive form $\log(1 - \operatorname{fl}(e^{-z}))$ cancels catastrophically as $z \to 0$ ($k \to 1$).
Following Mächler (2015), we use the branchwise-stable

$$
\operatorname{log1mexp}(z) =
\begin{cases}
\log\big(-\operatorname{expm1}(-z)\big), & 0 \le z \le \ln 2,\\[2pt]
\operatorname{log1p}\big(-e^{-z}\big), & z > \ln 2,
\end{cases}
$$

where `expm1` and `log1p` are the standard correctly-rounded library kernels.
Each branch evaluates its argument without cancellation, giving relative error $O(\varepsilon)$ over the entire range $z \in (0, \infty)$.
The limit $z = 0$ (exact anchor hit) correctly yields $-\infty$.

### 3.2 Accumulation accuracy

All summands are non-positive, so the sum has **condition number 1** (no cancellation).
Standard summation analysis (Higham 2002, §4.2) bounds the relative error of recursive summation by $(N - 1)\varepsilon + O(\varepsilon^2)$; for $N = 5.8 \times 10^5$ this is $\le 6.5 \times 10^{-11}$, and pairwise/chunked summation (as used by the implementation's tile-wise accumulation) improves this to $O(\varepsilon \log N)$.
There is no underflow: $S$ values like $-10^4$ (HCAS interior) are ordinary doubles, whereas the linear branch would need to represent $e^{-10^4}$.

In effect, the representable decision range grows from $1 - \alpha \in [2^{-54}, 1]$ (i.e. $|S| \lesssim 37$) to $|S| \lesssim 1.8 \times 10^{308}$---every degree of "depth inside the ODD" remains distinguishable.

### 3.3 Exact thresholding, including ζ = 1

By Proposition 1 the implementation compares $S(x) \le \operatorname{log1p}(-\zeta)$.
For $\zeta \in [0.5, 1)$ the right-hand side is computed from the exactly-representable $1 - \zeta$ (Sterbenz), so the comparison is exact up to the $O(\varepsilon)$ accuracy of $S$.
At $\zeta = 1$, $\operatorname{log1p}(-1) = -\infty$ and the rule selects exactly the anchor hits---the behavior that the linear branch structurally cannot deliver (it reports $\operatorname{fl}(\alpha) = 1.0 \ge 1$ for every saturated point).

A practical corollary: near $\zeta = 1$ a *linear* grid of thresholds carries no information beyond $\zeta \le 1 - 2^{-53}$; in log space one may sweep $\zeta = 1 - 10^{-r}$ for arbitrary $r$, enabling meaningful ROC analysis in the high-confidence regime.

## 4. Dual evaluation and comparison methodology

Both quantities are produced in a single pass over the anchor tiles: the Mahalanobis distances $m_i$ and kernel values $k_i$ are computed once, then accumulated as a running product $\prod (1 - k_i)$ (linear branch, bit-identical to the legacy implementation) and a running sum $\sum \operatorname{log1mexp}(m_i / 2)$ (log branch).
The marginal cost is one extra accumulator per evaluation point.

The OOD consistency adjustment consumes $S$ directly for the same reason: it selects the most-violated OOD point as $\arg\min_y S(y)$ rather than $\arg\max_y \alpha(y)$, because under failure mode (F1) a whole block of points shares $\operatorname{fl}(\alpha) = 1.0$ and `argmax` over that block---or over a `NaN` produced by a saturated covariance---silently degrades its documented "first maximum" tie rule.
$S$ additively decomposes over kernels, which also makes the per-iteration update exact and $O(1)$ per point; see docs/ood-consistency.md §4.

For the empirical comparison we report, per dataset:

1. **False-saturation fraction:** the share of test points with $\operatorname{fl}(\alpha_{\text{lin}}) = 1.0$ but $S > -\infty$ (points the linear branch cannot distinguish from anchor hits; under default bandwidths on HCAS this is 100%).
2. **Agreement off saturation:** $\max |\alpha_{\text{lin}} - (1 - e^{S})|$ over non-saturated points (expected $O(N \varepsilon)$).
3. **Confusion-matrix divergence:** TP/FP/TN/FN per threshold for both branches (`affinity_space ∈ {linear, log}` in the evaluation CSV); they coincide wherever neither failure mode (F1)–(F3) is active and diverge exactly in the saturated regime, which localizes the numerical artifact in the metric curves.

## 5. CSV output and the threshold grid

**(a) One CSV, two affinity spaces, and what the filename suffix means.**
Dataset-mode evaluation writes a single CSV per dataset containing both `affinity_space = "linear"` and `affinity_space = "log"` rows for every threshold and reference.
The filename suffix (e.g. `vcas_state_variables-evaluation-edges.csv`) names the **threshold spacing mode** used to build the sweep grid, not the affinity space---both spaces are always present in the same file, distinguished by the `affinity_space` column.

**(b) Linear rows saturate at ζ = 1.0 by design; paper numbers come from log rows.**
As shown in §2, once $K(x) \gtrsim 37$ the stored `affinity` is exactly `1.0` in float64, so `linear` rows report TP/FP > 0 even at the strictest threshold ζ = 1.0. This is not a bug---it is the documented failure mode (F3) of the linear representation, and the CSV keeps these rows intentionally so the artifact is visible next to the correct log-space numbers.
**All reported results should be computed from `affinity_space == "log"` rows only**; `linear` rows exist for comparison and regression-testing purposes.

**(c) The adaptive threshold grid (`build_threshold_pairs`).**
Dataset mode no longer sweeps a fixed `linspace(0, 1, count)` grid, because that wastes almost all of its resolution where the affinity distribution has no mass: the ODD boundary is sharp (the transition shell is about one kernel width thick), so most sampled points have either $\alpha \approx 0$ or $\alpha \approx 1$, and a uniform grid spends ~98% of its thresholds on the empty middle.
Instead, each threshold is generated as an exact analytic pair $(\zeta, S) = (\zeta, \log(1-\zeta))$, with $\zeta = 10^{-t}$ for the lower edge and $\zeta = 1 - 10^{-t}$ (equivalently $S = -t \ln 10$) for the upper edge, $t$ ranging linearly between decade 1 and an adaptive depth derived from the data (the smallest positive observed affinity for the lower edge, the most negative observed survival value for the upper edge; both floored at 16 decades).
A linear middle section over $\zeta \in [0.1, 0.9]$ covers datasets whose affinity spreads smoothly.
Because $S$ is carried alongside $\zeta$ from construction rather than re-derived from the (possibly saturated) float $\zeta$, the log-space decision rule stays exact to whatever depth the data requires, even where many thresholds share the same printed `affinity_threshold == 1.0`.
The CSV therefore also carries a `survival_threshold` column, and rows are sorted by descending survival (not by `affinity_threshold`) so saturated rows stay in the correct order.
The legacy fixed `"linear"`/`"log"` threshold modes (`build_affinity_thresholds`) remain available for the Monte Carlo results path only.

## References

- M. Mächler.
  *Accurately Computing log(1 − exp(−|a|))*.
  R project vignette, 2012.
- N. J. Higham.
  *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
- D. Goldberg.
  *What Every Computer Scientist Should Know About Floating-Point Arithmetic*.
  ACM Computing Surveys, 1991.
