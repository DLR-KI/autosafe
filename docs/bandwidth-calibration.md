<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>

SPDX-License-Identifier: CC-BY-SA-4.0
-->
# Scale-Calibrated Bandwidths for autoSAFE RBF Kernels

## 1. Setting and notation

Let $X = \{x_1, \dots, x_N\} \subset \mathbb{R}^D$ be the anchor set, expressed in the normalized coordinate system that the pipeline constructs from the dataset's declared variable ranges (each dimension mapped affinely to $[-1, 1]$). autoSAFE assigns each anchor a Gaussian RBF kernel with diagonal covariance $\Sigma_i = \operatorname{diag}(\sigma_{i1}, \dots, \sigma_{iD})$,

$$
k_i(x) = \exp\Big(-\tfrac12 (x - x_i)^\top \Sigma_i^{-1} (x - x_i)\Big),
\qquad
\alpha(x) = 1 - \prod_{i=1}^N \big(1 - k_i(x)\big),
$$

and declares $x$ inside the ODD iff $\alpha(x) \ge \zeta$.

Let $x_{\mathrm{nn}}(i) = \arg\min_{i' \ne i} \lVert x_i - x_{i'} \rVert_2$ denote the full-space nearest neighbor of $x_i$, let $d_i = \lVert x_i - x_{\mathrm{nn}}(i) \rVert_2$, and let $d_{ij} = |x_{i,j} - x_{\mathrm{nn}}(i)_j|$ be its per-dimension components.
The bandwidth law is

$$
\sigma_{ij} = (\kappa_j - \lambda_j)\, e^{-\eta_j d_{ij}} + \lambda_j,
\tag{1}
$$

with parameters $\kappa_j > 0$ (units length$^2$), $\eta_j > 0$ (units length$^{-1}$), and a lower bound $0 < \lambda_j \ll \kappa_j$ (units length$^2$), set relative to $\kappa$ as $\lambda_j = \lambda_{\mathrm{rel}}\, \kappa_j$ with $\lambda_{\mathrm{rel}} = e^{-10}$ by default (Section 5).

## 2. Why fixed parameters fail, and why the calibration scale must be the

full-space neighbor distance

Since the exponent of (1) must be dimensionless, $[\eta] = \text{length}^{-1}$ and $[\kappa] = \text{length}^2$: fixing $\kappa = \eta = 1$ implicitly selects a unit of length, so the estimated ODD changes under the very normalization the pipeline performs.
Quantitatively, $\sigma_{ij} \to \kappa$ as $d_{ij} \to 0$; for the HCAS dataset ($N \approx 5.8 \times 10^5$ anchors in $[-1,1]^7$) nearest-neighbor distances are tiny, so with $\kappa = 1$ every kernel spans the whole domain.
The elementary bounds (from $1 - t \le e^{-t}$ and the union bound), with $K(x) = \sum_i k_i(x)$,

$$
1 - e^{-K(x)} \le \alpha(x) \le \min\{1, K(x)\},
\tag{2}
$$

then give $\alpha = 1 - e^{-10^4} = 1$ to any floating-point precision everywhere in the evaluation region: the threshold sweep is constant---the observed pathology.

The calibration scale must also be chosen from the right geometry.
Two candidates:

- **Per-dimension 1D projection distances** $\min_{i'} |x_{i,j} - x_{i',j}|$: these scale like $\operatorname{span}_j / N$ (and collapse to 0 for quantized variables whose values repeat).
  They measure the spacing of *projections*, not of points; in $D > 1$ they are asymptotically unrelated to the local point density.
  Calibrating to them yields kernels of radius $O(N^{-1})$---measured on HCAS: radii $\sim 10^{-7}$ of the domain, $\alpha \approx 0$ everywhere.
  Degenerate.
- **Full-space nearest-neighbor distance** $d_i$: scales like $N^{-1/D}$---the actual gap between data points that kernels must bridge for the ODD interior to be connected.
  This is the scale we use.

## 3. The calibration rule

Let

$$
\tilde d = \operatorname{median}\{\, d_i : d_i > 0 \,\}
$$

(the median positive full-space NN distance; exact duplicate anchors are excluded).
We set, identically for all dimensions,

$$
\boxed{\;\eta = \frac{c}{\tilde d}, \qquad \kappa = (s\, \tilde d)^2\;}
\tag{3}
$$

with dimensionless defaults $c = 1$, $s = 3$, and $\lambda = \lambda_{\mathrm{rel}}\, \kappa$.
The law (1) becomes

$$
\sigma_{ij} = (s \tilde d)^2 \Big[\big(1 - \lambda_{\mathrm{rel}}\big)\,
e^{-c\, d_{ij}/\tilde d} + \lambda_{\mathrm{rel}}\Big]:
$$

bandwidths depend on the data only through the dimensionless ratios $d_{ij}/\tilde d$, scaled by the squared local length unit $\tilde d^{\,2}$.
$\kappa$ is isotropic in the normalized space; the per-dimension anisotropy enters through the components $d_{ij}$ of the neighbor offset---dimensions in which the nearest neighbor agrees (e.g. shared discrete values, $d_{ij} = 0$) keep the full width $\kappa$, dimensions in which it is distant are narrowed.

## 4. Invariance properties

**Proposition 1 (equivariance requirement).**
For a per-dimension affine map $T(x) = Ax + b$, $A = \operatorname{diag}(a)$, $a_j > 0$, the affinity built on $\{T(x_i)\}$ satisfies $\alpha'(T(x)) = \alpha(x)$ iff $\Sigma_i' = A \Sigma_i A$.
*Proof:* the Mahalanobis form is invariant under this joint transformation, and each $k_i$ determines $\alpha$; conversely a mismatch in any $\sigma_{ij}$ is exposed by points displaced along dimension $j$.
$\square$

**Proposition 2 (pipeline invariance).**
The full pipeline is invariant under any per-dimension affine re-expression of the *raw* data (changes of units, offsets), because normalization is a canonicalization: raw coordinates and their declared ranges transform together, so the normalized anchor set---and everything derived from it---is bit-identical.
$\square$

**Proposition 3 (in-space equivariance, isotropic maps).**
Within the normalized space, under a similarity transformation $T(x) = a x + b$ ($a > 0$ scalar): the L2 nearest-neighbor assignment is preserved, $d_i \mapsto a\, d_i$, hence $\tilde d \mapsto a \tilde d$, $\eta \mapsto \eta / a$, $\kappa \mapsto a^2 \kappa$, $\lambda = \lambda_{\mathrm{rel}} \kappa \mapsto a^2 \lambda$, and therefore $\sigma_{ij} \mapsto a^2 \sigma_{ij}$---exactly the condition of Proposition 1, so the calibrated affinity is invariant.
(This is why the lower bound must be specified *relative to $\kappa$*: an absolute $\lambda$ would not transform and would break the equivariance.)
$\square$

*Remark (anisotropic maps).*
Under anisotropic rescaling of the normalized space the L2 neighbor assignment itself may change, so exact equivariance holds only conditionally on a fixed assignment.
This is immaterial in practice by Proposition 2: the calibration always runs in the one canonical coordinate system.

## 5. Interpretation, the lower bound $\lambda$, and saturation control

- **Dense regions** ($d_{ij} \ll \tilde d$): $\sigma_{ij} \to \kappa = (s\tilde d)^2$ exactly at $d_{ij} = 0$---kernel standard deviation capped at $s$ median neighbor gaps: wide enough to bridge the gaps between neighboring anchors (connected ODD interior), no wider.
- **Median anchor** ($d_{ij} = \tilde d$): standard deviation $\approx s\, e^{-c/2} \tilde d \approx 1.8\, \tilde d$ at the defaults (the $\lambda$ term is negligible there).
- **Isolated anchors** ($d_{ij} \gg \tilde d$): widths shrink exponentially toward $\lambda$---outliers cannot inflate the ODD (conservative, appropriate for safety).

**Why a lower bound, and why this form.**
The isolation ratios $d_{ij}/\tilde d$ are heavy-tailed in real data; the pure law $\kappa e^{-\eta d}$ then underflows to numerical zero for the most isolated anchors, producing singular covariance matrices $\Sigma_i$ (verified empirically: without a bound, the most isolated HCAS anchors trigger the non-invertibility repair path).
Two remedies were considered:

1. a hard exponent clamp $\sigma = \kappa\, e^{-\min(\eta d,\, E)}$, and
2. the affine bound of Eq. (1), $\sigma = (\kappa - \lambda)e^{-\eta d} + \lambda$.

Both floor $\sigma$ and agree asymptotically when $\lambda = \kappa e^{-E}$, but (1) is preferable: it is smooth in $d$ (the clamp has a derivative discontinuity at $\eta d = E$), it gives exact bounds $\sigma_{ij} \in [\lambda, \kappa]$ and hence a guaranteed condition number $\kappa/\lambda = 1/\lambda_{\mathrm{rel}}$ for $\Sigma_i$, and it preserves $\sigma(0) = \kappa$ exactly.
Two requirements govern the choice of $\lambda$:

- $\lambda$ must scale with $\kappa$ ($\lambda = \lambda_{\mathrm{rel}} \kappa$): Proposition 3 fails for an absolute $\lambda$.
- $\lambda_{\mathrm{rel}}$ must be small enough that floored kernels are negligible at evaluation scale: with $\lambda_{\mathrm{rel}} = e^{-10}$, the floored standard deviation is $e^{-5} \approx 0.7\%$ of $s\tilde d$, so a floored kernel decays within a sub-percent fraction of a neighbor gap.
  Results are insensitive to $\lambda_{\mathrm{rel}}$ over a wide range (any $\lambda_{\mathrm{rel}} \lesssim e^{-5}$ leaves floored kernels evaluation-negligible while keeping $\Sigma_i$ well-conditioned).

Saturation is now local: $K(x)$ in (2) is dominated by anchors within a few multiples of $s \tilde d$, so $\alpha \to 1$ deep inside the support (the intended "certainly inside" semantics, reported separately by the log-space branch; see docs/log-space-affinity.md) and the transition $\alpha: 1 \to 0$ occupies a boundary layer of width $O(s \tilde d)$ around the data support.
The threshold sweep resolves this boundary geometry.

**The $\lambda$ floor is a construction-time bound only.**
The OOD consistency adjustment (paper Algorithm 1, `Samples.enforce_ood_consistency`) scales $\Sigma_i$ *below* $\lambda$ by design: the strict decrease $\alpha_{i^\ast}(x^\ast) = \exp(-q/2c)$ that its termination proof relies on would be destroyed by re-flooring after each shrink.
The conditioning bound $\operatorname{cond}(\Sigma_i) \le 1/\lambda_{\mathrm{rel}}$ above therefore applies to **pre-adjustment** kernels only.
How far below the floor float64 arithmetic remains valid---and the precondition $\mathcal{D}_{\mathrm{ID}} \cap \mathcal{D}_{\mathrm{OOD}} = \emptyset$ without which the adjustment cannot terminate at all---is treated in docs/ood-consistency.md.

## 6. Evaluation sampling at the calibrated scale

The boundary layer has width $O(s \tilde d)$, so the test-point generator must place points at that resolution: test points are drawn half uniformly in the (slightly expanded) variable box---far-field negatives---and half as anchors perturbed by isotropic Gaussian noise with per-dimension standard deviation

$$
\sigma_{\text{noise}} = \frac{m\, \tilde d}{\sqrt{D}}
\qquad (\text{so } \mathbb{E}\lVert \Delta x \rVert_2 \approx m \tilde d),
$$

default $m = 3$.
A span-proportional noise scale (legacy $0.05 \cdot \text{span}$) probes only the far field once kernels are density-scaled: on HCAS it places every perturbed point dozens of kernel widths off the data manifold, so all affinities vanish and the sweep is again uninformative.
Matching the sampling scale to the calibrated kernel scale is therefore part of the method, not a tuning trick; both $m$ and the sampling mode are recorded in the run's parameter sidecar.

Empirically (HCAS, 20 000-anchor subsample, $m = 3$): $\approx 19\%$ of test points saturate ($\alpha = 1$ to double precision), $\approx 30\%$ fall in the informative mid-range, the rest are far-field negatives---a well-conditioned operating curve.

## 7. Choice of constants

$c = 1$: one $e$-fold of variance decay per median spacing of isolation.
$s = 3$: kernels reach $\approx 3$ median gaps ("three-sigma" neighbor coverage); $s \lesssim 1$ fragments the interior, large $s$ re-approaches global saturation.
$\lambda_{\mathrm{rel}} = e^{-10}$: see Section 5. Recommended ablations: $s \in \{1, 2, 3, 5\}$, $c \in \{0.5, 1, 2\}$, $m \in \{1, 3, 5\}$, $\lambda_{\mathrm{rel}} \in \{e^{-12}, e^{-10}, e^{-7}, e^{-5}\}$.

## 8. Limitations

- $\tilde d$ is a single global scale; strongly multi-scale data may warrant per-anchor quantile calibration (future work).
- $\kappa$ is isotropic in the normalized space; anisotropy enters only through the neighbor components in (1).
  Full local covariance estimation is out of scope.
- The median excludes exact duplicates; datasets dominated by duplicated rows (>50%) would need de-duplication first.
