<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
SPDX-License-Identifier: MIT
-->
# The OOD Consistency Adjustment: Preconditions, Numerics, and Cost

This note documents the OOD consistency adjustment (paper Def. "OOD Consistency Constraint" and Algorithm 1) as **implemented** in `Samples.enforce_ood_consistency`.
It states the hypothesis the termination proof depends on and where the implementation enforces it (§2), the float64 limits of repeated covariance scaling (§3), the incremental update that makes the procedure usable at production scale (§4), the optional closed-form jump and the regime in which it applies (§5), the resulting complexity---which differs from the statement currently in the paper (§6)---the interaction with the $\lambda$ floor (§7), and what the loop reports while it runs (§8).

The behavior documented here was derived while diagnosing a production run that could not terminate.

## 1. Setting and notation

Let $A = \{x_1, \dots, x_N\} \subseteq \mathcal{D}_{\mathrm{ID}}$ be the anchor points, each carrying an RBF kernel

$$
k_i(x) = \exp\!\big(-\tfrac12 m_i(x)\big),
\qquad
m_i(x) = (x - x_i)^\top \Sigma_i^{-1} (x - x_i),
$$

and let the global affinity and its log-survival be

$$
\alpha(x) = 1 - \prod_{i=1}^N \big(1 - k_i(x)\big),
\qquad
L(x) := \log\big(1 - \alpha(x)\big) = \sum_{i=1}^N \log\big(1 - k_i(x)\big) \le 0 .
$$

$L$ is the quantity `Samples.affinity_dual` returns as its second element; see `docs/log-space-affinity.md`.

Given an OOD set $\mathcal{D}_{\mathrm{OOD}} = \{y_1, \dots, y_M\}$ and a bound $\xi \in (0,1)$, Algorithm 1 repeats:

1. $x^\ast \leftarrow \arg\max_{y \in \mathcal{D}_{\mathrm{OOD}}} \alpha(y)$;
2. $i^\ast \leftarrow \arg\max_i k_i(x^\ast)$;
3. $\Sigma_{i^\ast} \leftarrow c\,\Sigma_{i^\ast}$ with $c \in (0,1)$;

until $\max_y \alpha(y) \le \xi$.

**Implementation note.**
Step 1 is performed as $\arg\min_y L(y)$, not $\arg\max_y \alpha(y)$.
The two are equivalent in exact arithmetic because $\alpha = 1 - e^{L}$ is strictly decreasing in $L$, but in float64 $\alpha$ saturates at exactly $1.0$ once roughly 36 kernels are near-1 (`docs/log-space-affinity.md` §2), and `argmax` over a block of exact ties---or over `NaN`---silently degrades the documented "first maximum" tie rule.
$L$ stays discriminative over the whole range.

## 2. The disjointedness precondition

The paper's termination proposition requires $q_{i^\ast} = (x^\ast - x_{i^\ast})^\top \Sigma_{i^\ast}^{-1} (x^\ast - x_{i^\ast}) > 0$ and obtains it from $\mathcal{D}_{\mathrm{ID}} \cap \mathcal{D}_{\mathrm{OOD}} = \emptyset$.
That hypothesis is not decorative.

> **Proposition 1 (an anchor is never OOD-consistent).**
> Let $x \in A$, say $x = x_j$.
> Then $\alpha(x) = 1$ for every choice of positive-definite covariances $\Sigma_1, \dots, \Sigma_N$.
> Consequently, if $A \cap \mathcal{D}_{\mathrm{OOD}} \neq \emptyset$ then $\max_y \alpha(y) = 1 > \xi$ for every $\xi \in (0,1)$, the exit condition of Algorithm 1 is unsatisfiable, and the loop does not terminate.

*Proof.*
$m_j(x_j) = 0^\top \Sigma_j^{-1} 0 = 0$, so $k_j(x_j) = \exp(0) = 1$ irrespective of $\Sigma_j$.
The factor $\big(1 - k_j(x_j)\big) = 0$ annihilates the product, giving $\alpha(x_j) = 1 - 0 = 1$.
Step 3 rescales $\Sigma_{i^\ast}$ but $m_j(x_j) = 0$ for every scaling, so no iteration can change this value.
$\square$

Equivalently, in log space $L(x_j) = -\infty$: the point is at the bottom of the selection order forever, so Algorithm 1 selects it, shrinks its dominant kernel, and re-selects it, indefinitely.

Proposition 1 is a statement about the *input*, not about the algorithm.
The theorem is correct and states its hypothesis; what was missing was any enforcement between the theorem and a running job.
Two mechanisms now provide it:

- **Construction.**
  `evaluate_dataset_mode` subtracts the `ood_path` rows from the anchor pool before subsampling (`_build_or_load_affinity_odd(exclude_points=…)`), reproducing what `experiments/benchmark/run_held_out_vcas_hole.py::_vcas_with_hole` already did (`id_pts = region[~in_hole(region)]`).
  Because the anchor set then *depends on* the OOD file, the ODD JSON and the nearest-neighbor `.npz` both carry an `-ex<digest>` tag keyed to that file---neither cache validates the anchor set it was built from, so an untagged run would silently reuse a cache built from the unfiltered pool.
- **Verification.**
  `Samples.enforce_ood_consistency` tests $A \cap \mathcal{D}_{\mathrm{OOD}} = \emptyset$ exactly, before the loop, and raises `OODAnchorCoincidenceError` naming the count and the first offending index.
  The test is `rows_in`, one lexsort over the $(N + M) \times n$ stack: $O((N+M)\log(N+M)\,n)$ once, against a loop whose alternative is unbounded.

Row equality is compared **by value**, not by bit pattern: $-0.0$ matches $0.0$ (they give the same $q$), and a row containing `NaN` matches nothing.
This matters because the anchor path and the OOD path apply the *same fitted normalizer object* to the *same source row*, so coincidences are bit-identical by construction---a hand-rolled re-implementation of the normalization compares unequal in the last ulp and reports a spurious all-clear.

### 2.1 Near-coincidence

Points that are merely *close* to an anchor are legal but expensive: §6 shows the required number of scalings grows like $\log(1/q)$.
The implementation raises `NearAnchorOODWarning` when any OOD point starts at $L = -\infty$ (numerically saturated but not exactly coincident), because those runs may need very many iterations, and §3 bounds how many are available before the arithmetic breaks down.

## 3. Float64 limits of repeated scaling

Repeated application of step 3 exhausts the float64 range.
After $t$ scalings of the same kernel, starting from $\sigma_0 = \min_k \Sigma_{kk}$ and $\nu_0 = \max_k (\Sigma^{-1})_{kk}$:

$$
\Sigma_{kk} \text{ underflows to } 0 \iff t > \frac{\log(\sigma_{\min}/\sigma_0)}{\log c},
\qquad
(\Sigma^{-1})_{kk} \text{ overflows to } \infty \iff t > \frac{\log(F_{\max}/\nu_0)}{\log(1/c)},
$$

with $\sigma_{\min} \approx 4.94\cdot10^{-324}$ and $F_{\max} \approx 1.798\cdot10^{308}$.
For the measured VCAS ODD ($\Sigma_{kk} \in [3.99\cdot10^{-9},\, 8.79\cdot10^{-5}]$, so $\nu_0 \approx 2.5\cdot10^{8}$) and $c = 0.9$ this gives $t \approx 6\,550$ for overflow and $t \approx 6\,880$ for underflow.

The incident's implementation applied step 3 as two *independent* updates, $\Sigma \leftarrow c\,\Sigma$ and $\Sigma^{-1} \leftarrow c^{-1}\Sigma^{-1}$.
Two consequences, both observed:

1. **The `max_iterations = 10^6` cap permits far more iterations than the arithmetic supports.**
  Past $t \approx 6.6\cdot10^3$ the loop computes $0 \cdot \infty = \texttt{NaN}$, `np.argmax` over an array containing `NaN` returns the first `NaN` index, `NaN <= xi` is `False`, and the loop continues on corrupted state.
  Note that `sigma_inv` is *not* "finite for any finite iteration count", as was once assumed: it is finite only for $t \lesssim 6.6\cdot10^3$.
2. **$\Sigma$ and $\Sigma^{-1}$ stopped being consistent inverses long before either saturated**, because $c$ and $c^{-1}$ are not exact reciprocals in binary floating point, so scaling the two arrays independently lets them drift apart.

### 3.1 What the implementation does instead

**$\Sigma^{-1}$ is recomputed from the shrunken $\Sigma$, not rescaled alongside it** (`_invert_covariance`).
Rescaling would be marginally cheaper, but consequence 2 above is a direct result of it, and the drift is unbounded in the iteration count while the kernel reads only $\Sigma^{-1}$ and the serializer persists both.
For the diagonal $\Sigma$ that the $\sigma$-law always produces, recomputation is one reciprocal per dimension---the cost is negligible, and the pair is consistent by construction at every iteration.
A dense $\Sigma$ falls back to `np.linalg.inv`, with a singular result reported as saturation.

Both candidate arrays are computed *before* being committed, and `KernelSaturationError` is raised if either is non-finite or if any $\Sigma_{kk} \le 0$, naming the kernel, the offending OOD index, and near-coincidence as the likely cause.
A run that would previously have spent months on `NaN` now fails in seconds.

Note that $\Sigma^{-1}$ overflowing is not the only way to lose a point: the squared distance itself can underflow.
At $\|x - x_i\| \lesssim 10^{-162}$, $\|x - x_i\|^2$ is $0$ in float64, so $L(x) = -\infty$ even though $x \neq x_i$ and the exact-coincidence check of §2 passes.
This is what `NearAnchorOODWarning` reports.

## 4. Incremental log-survival update

Each iteration changes exactly one kernel, $i^\ast$.
Recomputing $\alpha$ for all $M$ OOD points against all $N$ kernels therefore does $O(MNn)$ work to absorb an $O(Mn)$ change.

> **Proposition 2 (exact incremental update).**
> Let $L$ be the log-survival before an iteration and $L'$ after it, and let $k_{i^\ast}^{\mathrm{old}}, k_{i^\ast}^{\mathrm{new}}$ be the values of the adjusted kernel before and after Step 3. Then, for every $y$,
> $$
> L'(y) = L(y) - \log\big(1 - k_{i^\ast}^{\mathrm{old}}(y)\big)
>              + \log\big(1 - k_{i^\ast}^{\mathrm{new}}(y)\big),
> $$
> and $\alpha'(y) = -\operatorname{expm1}\big(L'(y)\big)$.
> The update costs $O(Mn)$ and is exact in exact arithmetic.

*Proof.*
$L$ is a sum over kernels and only the $i^\ast$ term changed; subtract the old term and add the new one.
$\square$

Together with the $O(Nn)$ evaluation of $k_i(x^\ast)$ needed for Step 2, one iteration costs $O\big((M + N)n\big)$ instead of $O(MNn)$.

Three numerical caveats govern the implementation:

- **Cancellation.**
  $\log(1 - k)$ with $k = e^{-z}$ is computed with the branch $\log(-\operatorname{expm1}(-z))$ for $z \le \log 2$ and $\operatorname{log1p}(-e^{-z})$ above, mirroring the JAX tile in `_affinity._build_tile("diag_dual")` so the incremental path and the exact refresh agree to round-off.
- **Drift.**
  Errors accumulate over many updates, so $L$ is recomputed exactly every `refresh_interval` iterations (default 1000).
  This amortizes to one full sweep per 1000 iterations.
- **Non-invertibility.**
  $L = -\infty$ (a saturated kernel) cannot be undone by subtraction.
  Whenever the updated $L$ contains a non-finite entry the implementation falls back to a full recomputation for that iteration.

Additionally, the loop **never exits on an incrementally-maintained estimate**: when the exit test passes on a drifted $L$, $L$ is recomputed exactly and the test is repeated.
The reported `max_ood_affinity` is therefore backed by an exact sweep, and the returned `exact_recomputations` counts how many such sweeps occurred.

### 4.1 Measured effect

Disjoint synthetic ID/OOD sets ($n = 5$, $\xi = 0.3$, $c = 0.9$), comparing the measured loop against $T \times$ the measured cost of one full $M \times N$ sweep---the pre-fix per-iteration cost:

|   $N$ |    $M$ |     $T$ |   loop | ms/iter | exact sweeps | one sweep | pre-fix estimate |   speedup |
| ----: | -----: | ------: | -----: | ------: | -----------: | --------: | ---------------: | --------: |
| 2 000 |  5 000 |  77 493 | 31.5 s |    0.41 |           79 |    6.6 ms |            512 s | **16.2×** |
| 8 000 | 15 000 | 343 020 |  327 s |    0.95 |          345 |   41.1 ms |         14 109 s | **43.1×** |

Both runs verify $\max_y \alpha(y) = 0.29997 \le \xi$ against an independent evaluation.
The per-iteration cost is dominated by fixed NumPy/Python overhead rather than arithmetic---$(M+N)n$ is only $4\cdot10^{5}$ element operations at the second row---so the speedup is *overhead*-limited and grows with $MN/(M+N)$ (1 429 and 5 217 for the two rows) rather than tracking it exactly.
Extrapolating the ~1 ms/iteration figure to the VCAS configuration of the incident ($M = 62\,192$, $N = 20\,000$) and the $T \lesssim 1.3\cdot10^{6}$ bound of §6 puts the whole loop under an hour, against the 46–190 day estimate measured for the pre-fix implementation.

Note $T \approx N$ in both rows---every kernel was adjusted at least once---which is the regime §5 identifies as the one where no single-kernel closed form applies.

## 5. Optional closed-form jump

While the selected point $x^\ast$ and its dominant kernel $i^\ast$ stay fixed, repeated applications of Step 3 can be collapsed into a single scaling $c^{t}$.
Write $q = -2\log k_{i^\ast}(x^\ast) > 0$ for the current Mahalanobis distance and $L_{\mathrm{rest}} = L(x^\ast) - \log\big(1 - k_{i^\ast}(x^\ast)\big)$ for the survival contributed by all other kernels.
Requiring $L(x^\ast) \ge \log(1-\xi)$ after the scaling gives, with $r := \log(1-\xi) - L_{\mathrm{rest}}$,

$$
\log\big(1 - k^{\mathrm{new}}\big) \ge r
\;\Longleftrightarrow\;
k^{\mathrm{new}} \le -\operatorname{expm1}(r)
\;\Longleftrightarrow\;
c^{t} \le \frac{q}{2b},
\quad b := -\log\big(-\operatorname{expm1}(r)\big) > 0,
$$

so $t \ge \log\!\big(q/(2b)\big) / \log c$ (the sense flips because $\log c < 0$).

**Applicability.**
The derivation requires $r < 0$, i.e. $e^{L_{\mathrm{rest}}} > 1 - \xi$: the kernels *other* than $i^\ast$ must not already violate the constraint at $x^\ast$ on their own.
When $r \ge 0$ no single-kernel scaling can fix the point---several different kernels must shrink---and there is no closed form; the implementation falls back to a plain $t = 1$ step.
This is the common case in a dense superposition.
Measured on the two-blob test fixture ($N = 40$, $\xi = 0.05$), the $r \ge 0$ branch fires on 368 of 370 iterations and the jump saves nothing; on a separated fixture where each OOD point sits beside one isolated anchor, the same run collapses from 175 iterations to 5, one per OOD point.

Two caps are applied: $t$ never makes $i^\ast$ non-dominant at $x^\ast$ (which is where unbatched Algorithm 1 would switch kernels), and never drives $\Sigma$ or $\Sigma^{-1}$ past the float64 limits of §3.

**Fidelity.**
Even with the dominance cap this is an approximation of Algorithm 1: the *selected point* may change part-way through a segment that the closed form treats as uniform, so the adjusted kernel can end up smaller than Algorithm 1 would have made it, and the resulting ODD tighter.
The constraint $\max_y \alpha(y) \le \xi$ is always still satisfied.
For this reason the jump is opt-in (`batch_jump=False` by default, spec key `ood_batch_jump`) and all reported paper numbers use the default path.

## 6. Complexity

Let $T$ be the total number of covariance adjustments.
The shipped procedure costs

$$
O\big(N^2 n \;+\; T\,(M + N)\,n\big)
$$

time---$N^2 n$ for the nearest-neighbor pass that defines the kernels, plus $T$ adjustment iterations---in $O(Nn)$ space.
The naive form of the loop, which recomputes the full sweep each iteration, costs $O(N^2n + T M N n)$.

$T$ is bounded by the paper's own iteration-count argument.
With $\xi_N := 1 - (1-\xi)^{1/N}$ the per-kernel budget that guarantees the global constraint, a single kernel needs at most

$$
K \;\ge\; \frac{\log\big(-2\log \xi_N / q\big)}{\log(1/c)}
$$

scalings, and $T \le \sum_i K_i \le N K$.
For the VCAS configuration ($N = 20\,000$, $\xi = 0.3$, $c = 0.9$): $\xi_N \approx 1.783\cdot10^{-5}$, $-2\log\xi_N \approx 21.87$, and with $\Sigma_{kk} \approx 4.2\cdot10^{-5}$, $K$ ranges from $\approx 21$ at anchor–OOD separation $10^{-2}$ to $\approx 108$ at $10^{-4}$.
Taking $K \approx 65$ gives $T \lesssim 1.3\cdot10^{6}$---**above** the former `max_iterations` default of $10^{6}$, which is why the cap is now a spec key (`ood_max_iterations`) and the VCAS item sets it to $2\cdot10^{6}$.

The paper currently states the cost as $O(N^2 n + MNn)$ (`main.tex` L546, L548, L1187).
The $MNn$ term is a **single** pass over the OOD set; the procedure performs one such pass per adjustment.
A correction has been drafted for the paper text.

## 7. Interaction with the $\lambda$ floor

The $\sigma$-law floors each diagonal entry at $\lambda = \texttt{SIGMA\_FLOOR\_RATIO} \cdot \kappa$ at construction time (`docs/bandwidth-calibration.md` §5).
The adjustment loop scales $\Sigma$ **freely below that floor** and must: the strict decrease $\alpha_{i^\ast}^{\mathrm{new}}(x^\ast) = \exp(-q/2c)$ that the termination proof relies on would be destroyed by re-flooring after each shrink.
The conditioning bound $\operatorname{cond}(\Sigma) \le 1/\texttt{SIGMA\_FLOOR\_RATIO}$ therefore holds only for **pre-adjustment** kernels; §3 bounds how far below the floor the arithmetic remains valid.

## 8. Observability

The incident's proximate cost was not the bug but the silence: a loop with a $10^6$ cap, a day-scale runtime, and no output at all, so "is it stuck or is it working?" required process forensics.
The implementation now:

- shows a `tqdm.rich` progress bar for the duration of the loop.
  The bar's total is `max_iterations`, which is a cap rather than an expectation, so the percentage is not meaningful---the elapsed time and the iteration rate are, and those are what separate "working" from "stuck";
- writes a per-iteration progress line every `log_interval` iterations through `tqdm.write`, so it interleaves with the bar rather than corrupting it;
- logs the start and end of the run (sizes, parameters, final counts) to the `loguru` logger;
- returns `iterations`, `max_ood_affinity`, `adjusted_kernels` and `exact_recomputations` for the caller to record.
  `evaluate_dataset_mode` writes all of them, plus `ood_anchor_coincidences` (which must be 0), into the run's sidecar JSON.

## References

- `docs/log-space-affinity.md`---the log-survival representation and its numerics.
- `docs/bandwidth-calibration.md`---the $\sigma$-law, $\kappa$, $\eta$, $\lambda$.
