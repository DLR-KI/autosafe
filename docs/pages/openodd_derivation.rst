.. SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
..
.. SPDX-License-Identifier: CC-BY-SA-4.0

OpenODD derivation
==================

This page describes how a fitted :class:`autosafe.AutoSafeODD` is converted
into a validated ASAM OpenODD 1.0.0 YAML document. The exporter constructs a
conservative inner approximation of the fitted autoSAFE membership region:
every condition emitted to the OpenODD is accepted by the fitted model, but
the OpenODD may omit some accepted points.

The construction is deterministic. There is no Monte Carlo, grid, or boundary
sampling during export. In this page, a *sample* is an observed
in-distribution (ID) anchor used to fit one kernel. Each fitted anchor produces
at most one candidate OpenODD region.

Notation
--------

Let:

* :math:`n` be the number of fitted ID anchors;
* :math:`d` be the number of encoded model dimensions;
* :math:`x_i \in \mathbb{R}^d` be anchor :math:`i`;
* :math:`\Sigma_i` be its positive-definite RBF covariance;
* :math:`\zeta \in (0,1)` be the resolved membership threshold; and
* :math:`R \leq n` be the number of distinct exported regions.

The exporter always uses
``odd.resolved_config.membership.affinity_threshold``. Thus, :math:`\zeta`
can either be the fixed value requested by the caller or the value resolved
from held-out calibration data in conformal membership mode.

From data to fitted kernels
---------------------------

Fitting first normalizes the ID anchors. Each normalized anchor becomes the
center of one RBF kernel:

.. math::

    k_i(x)
    =
    \exp\left(
        -\frac{1}{2}
        (x-x_i)^\mathsf{T}
        \Sigma_i^{-1}
        (x-x_i)
    \right).

The default calibrated mode finds the exact nearest neighbour of every anchor.
It uses the median positive full-space nearest-neighbour distance
:math:`\widetilde d` to resolve:

.. math::

    \kappa = (s\widetilde d)^2,
    \qquad
    \eta = \frac{\gamma}{\widetilde d},
    \qquad
    \lambda = \lambda_{\mathrm{rel}}\kappa.

Manual mode instead accepts :math:`\kappa`, :math:`\eta`, and :math:`\lambda`
directly. Scalars or one value per encoded dimension are supported.

For anchor :math:`i` and encoded dimension :math:`j`, the fitted diagonal
variance is:

.. math::

    \sigma_{ij}
    =
    (\kappa_j-\lambda_j)
    \exp\left(
        -\eta_j
        \left|x_{\operatorname{NN}(i),j}-x_{ij}\right|
    \right)
    +\lambda_j.

If optional OOD consistency was applied during fitting, it may subsequently
shrink some fitted covariances. Export uses the final covariances, so the
generated OpenODD reflects those adjustments. OOD points are not otherwise
read or embedded by the exporter.

Combined affinity and membership
--------------------------------

The kernel set combines individual affinities as:

.. math::

    A(x) = 1-\prod_{i=1}^{n}\left(1-k_i(x)\right).

The fitted ODD accepts :math:`x` when:

.. math::

    A(x) \geq \zeta.

The implementation evaluates this decision through the equivalent
log-survival threshold for numerical stability near affinity one.

Kernel ellipsoids
-----------------

Define:

.. math::

    r^2 = -2\log(\zeta).

The individual :math:`\zeta`-superlevel set of kernel :math:`i` is the
ellipsoid:

.. math::

    E_i
    =
    \left\{
        x :
        (x-x_i)^\mathsf{T}
        \Sigma_i^{-1}
        (x-x_i)
        \leq r^2
    \right\}.

OpenODD range expressions are axis-aligned, so the exporter does not attempt
to encode :math:`E_i` directly. Instead, it constructs an axis-aligned box
:math:`B_i` contained in :math:`E_i`.

Diagonal covariance
~~~~~~~~~~~~~~~~~~~

For diagonal
:math:`\Sigma_i=\operatorname{diag}(\sigma_{i1},\ldots,\sigma_{id})`, the
half-width in dimension :math:`j` is:

.. math::

    w_{ij}
    =
    \sqrt{\frac{r^2\sigma_{ij}}{d}}.

The candidate box is:

.. math::

    B_i
    =
    \left\{
        x :
        \left|x_j-x_{ij}\right|\leq w_{ij}
        \quad\text{for every }j
    \right\}.

For every point in this box:

.. math::

    \sum_{j=1}^{d}
    \frac{(x_j-x_{ij})^2}{\sigma_{ij}}
    \leq
    \sum_{j=1}^{d}\frac{r^2}{d}
    =
    r^2.

Therefore :math:`B_i\subseteq E_i`.

Full covariance
~~~~~~~~~~~~~~~

For a non-diagonal covariance, let
:math:`\lambda_{\min}(\Sigma_i)` be its smallest eigenvalue. The exporter
uses the same half-width in every dimension:

.. math::

    w_i
    =
    \sqrt{
        \frac{
            r^2\lambda_{\min}(\Sigma_i)
        }{d}
    }.

Any displacement :math:`\delta=x-x_i` in the resulting box satisfies:

.. math::

    \|\delta\|_2^2
    \leq
    r^2\lambda_{\min}(\Sigma_i),

and consequently:

.. math::

    \delta^\mathsf{T}\Sigma_i^{-1}\delta
    \leq
    \frac{\|\delta\|_2^2}{\lambda_{\min}(\Sigma_i)}
    \leq r^2.

This isotropic box is conservative for arbitrary positive-definite
covariances, although it can be smaller than a box optimized for the
ellipsoid's orientation.

Conservativeness of the union
-----------------------------

For every kernel and every point:

.. math::

    A(x)
    =
    1-\prod_{\ell=1}^{n}(1-k_\ell(x))
    \geq k_i(x).

As :math:`B_i\subseteq E_i`, membership in a generated box implies:

.. math::

    x\in B_i
    \Longrightarrow
    k_i(x)\geq\zeta
    \Longrightarrow
    A(x)\geq\zeta.

It follows that:

.. math::

    \bigcup_{i=1}^{n}B_i
    \subseteq
    \{x:A(x)\geq\zeta\}.

This proves that the generated OpenODD is an inner approximation. It can omit
points for two reasons:

* an axis-aligned box covers only part of its kernel ellipsoid; and
* several kernels can jointly make :math:`A(x)\geq\zeta` even when every
  individual kernel has :math:`k_i(x)<\zeta`.

The first effect becomes more pronounced as the encoded dimension grows.

Mapping boxes to OpenODD
------------------------

The fitted normalizer is inverted on every lower and upper bound so OpenODD
conditions use the original feature coordinates. The caller maps all encoded
dimensions exactly once using
:class:`autosafe.OpenODDNumericFeature` and
:class:`autosafe.OpenODDCategoricalFeature`.

For each candidate box:

* a floating-point numeric feature becomes an inclusive OpenODD range;
* an integer feature uses the ceiling of the lower bound and floor of the
  upper bound, and the region is discarded if no integer remains; and
* a categorical literal is included only when its complete encoded prototype
  lies inside the box. The region is discarded if a categorical feature has
  no representable literal.

Exact duplicate condition mappings are removed. Every remaining box becomes
one standard module whose feature conditions are combined with
``INCLUDE_AND``. The root ``ODD`` module references all region modules through
``INCLUDE_OR``, producing their union.

The ODD body contains only standard OpenODD taxonomy concepts, expressions,
and module references. Requested and resolved autoSAFE parameters and
derivation statistics are flattened into the root module's standard
``METADATA`` field for reproducibility; they do not become ODD conditions.

Finally, the generated mapping is checked against the derived structural
schema and semantic validation rules before it is serialized as ASAM OpenODD
1.0.0 YAML.

Time complexity
---------------

In addition to :math:`n`, :math:`d`, and :math:`R`, define:

* :math:`F` as the number of non-diagonal fitted covariances;
* :math:`L_g` as the number of literals for categorical feature group
  :math:`g`;
* :math:`d_g` as the encoded width of that categorical group;
* :math:`P=\sum_g L_gd_g` as the categorical prototype checking work per
  candidate region; and
* :math:`S` as the size of the generated YAML-decoded document.

Exporting an already fitted model has the following costs:

* Box construction is
  :math:`O((n-F)d^2+Fd^3)` in the current implementation. Diagonal detection
  compares dense :math:`d\times d` matrices, while a full covariance requires
  an eigendecomposition.
* Restoring original coordinates is :math:`O(nd)`.
* Building numeric and categorical conditions is
  :math:`O(nd+nP)`.
* Duplicate removal, validation, and serialization are expected
  :math:`O(S)` for generated documents.

The total export time is therefore:

.. math::

    O\left(
        (n-F)d^2
        +Fd^3
        +nP
        +S
    \right).

For the normal all-diagonal case this becomes
:math:`O(nd^2+nP+S)`. The mathematical box construction itself needs only
:math:`O(nd)` work for diagonal covariances. Reusing the kernel's diagonal
flag and storing its variance diagonal directly would remove the current
:math:`d^2` implementation overhead.

For numeric-only feature mappings, :math:`P=0`. If most anchors produce
distinct regions, :math:`R` is close to :math:`n` and
:math:`S=\Theta(Rd)`. In that common case, YAML generation and validation can
be practically output-bound.

If the cost of fitting from ID data is also included, exact flat
nearest-neighbour search costs :math:`O(n^2d)`. The current kernel
implementation stores and inverts each covariance as a dense matrix, giving
an :math:`O(nd^3)` upper bound for covariance inversion even though fitted
covariances are normally diagonal. Without optional OOD consistency, the
current end-to-end upper bound is consequently:

.. math::

    O\left(
        n^2d
        +nd^3
        +nP
        +S
    \right).

Optional conformal calibration and OOD consistency add costs that depend on
the number of held-out or observed OOD rows and, for OOD consistency, the
number of covariance-shrink iterations.

Space complexity
----------------

The fitted model stores:

* :math:`O(nd)` anchor coordinates; and
* :math:`O(nd^2)` covariance and inverse-covariance matrices in the current
  dense representation.

The exporter additionally stores:

* lower and upper bounds for every candidate box, :math:`O(nd)`;
* at most one temporary covariance or eigensolver workspace,
  :math:`O(d^2)`;
* caller-supplied categorical prototypes, :math:`O(P)`; and
* the generated document and serialized YAML, :math:`O(S)`.

Thus, additional export space is:

.. math::

    O(nd+d^2+P+S),

while peak space including the already fitted model is:

.. math::

    O(nd^2+P+S).

For numeric-only mappings with :math:`R` distinct regions,
:math:`S=\Theta(Rd)`. The output itself can therefore be the dominant memory
cost when a large number of anchors produces a large number of OpenODD
modules.
