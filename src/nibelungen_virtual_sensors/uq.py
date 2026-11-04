"""Uncertainty propagation of the embedded model-form parameters c_top and c_ext.

The calibrated model has two independent LogNormal convection factors, c_top
on the deck top and c_ext on all other external faces, ln c_j ~ N(ln median_j,
σ_ln,j²).  As in the calibration, each is written as a function of a standard
normal germ, c_j = median_j · exp(σ_ln,j ξ_j), and the pipeline builds an
order-p Hermite PCE in (ξ_top, ξ_ext) on the tensor (p+1)×(p+1) Gauss–Hermite
rule (same construction as ``src/simulation/pce_builder.py``).  The PCE mean
equals the weighted mean of the node outputs; the variance of the total-order
expansion differs from the weighted variance only by the truncated higher-order
terms (relative difference of order 1e-4 for the calibrated spreads), so the
weighted moments of the node outputs are used; no PCE coefficients are stored.
"""

from __future__ import annotations

import numpy as np


def lognormal_hermite_rule(median: float, sigma_ln: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Values c_i = median · exp(σ_ln ξ_i) and weights of the n-point Gauss–Hermite rule in ξ ~ N(0, 1).

    Reproduces the nodes of chaospy's ``generate_quadrature(n - 1, Normal(0, 1), rule="gaussian")``
    mapped to c; the weights sum to 1.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    xi, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return median * np.exp(sigma_ln * xi), w / w.sum()


def lognormal_hermite_tensor_rule(factors: list[tuple[float, float]], n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss–Hermite rule of independent LogNormal factors.

    ``factors`` lists ``(median, sigma_ln)`` per factor.  Returns the parameter
    values, shape ``(n_nodes ** d, d)``, and the weights (sum 1), in the order of
    chaospy's ``generate_quadrature(n - 1, J(Normal(0, 1), ...), rule="gaussian",
    sparse=False)``: the first factor varies slowest.
    """
    rules = [lognormal_hermite_rule(m, s, n_nodes) for m, s in factors]
    values = np.stack([g.ravel() for g in np.meshgrid(*[c for c, _ in rules], indexing="ij")], axis=1)
    weights = np.ones(1)
    for _, w in rules:
        weights = np.outer(weights, w).ravel()
    return values, weights


CONVECTION_FACTORS = ("top", "external")   # order of the germs, as in the pipeline config (c_top, c_ext)


def convection_rule(unc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature of the embedded convection factors from ``model_config.json["uncertainty"]``:
    values ``(n_nodes, 2)`` as ``(c_top, c_ext)`` and weights."""
    cf = unc["convection_factors"]
    return lognormal_hermite_tensor_rule(
        [(float(cf[k]["median"]), float(cf[k]["sigma_ln"])) for k in CONVECTION_FACTORS],
        int(unc["n_quadrature_nodes"]))


def weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance over axis 0 (the quadrature nodes)."""
    w = weights.reshape((-1,) + (1,) * (values.ndim - 1))
    mean = np.sum(w * values, axis=0)
    var = np.sum(w * (values - mean) ** 2, axis=0)
    return mean, var


def total_variance_moments(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Law of total variance over the model parameters c (axis 0) and the sensor position x (last axis).

    ``values`` has shape ``(n_nodes, ..., n_samples)``: the reading at every
    quadrature node and position sample.  Returns ``(mean, var_c, var_x)`` of
    shape ``(...)``:

    * ``mean  = E_c E_x Y``
    * ``var_c = Var_c[E_x Y]``  (the embedded model-form variance of the position-averaged reading)
    * ``var_x = E_c[Var_x Y]``  (the position variance, population variance over the samples)

    so that ``var_c + var_x`` is the variance of the pooled ensemble with
    weights ``w_i / n_samples``.  Both outer moments use the Gauss rule, which
    is exact for the PCE of the same order (see the module doc).
    """
    w = weights.reshape((-1,) + (1,) * (values.ndim - 2))
    mean_x = values.mean(axis=-1)
    var_x_c = values.var(axis=-1)                 # population variance, per node
    mean, var_c = weighted_moments(mean_x, weights)
    var_x = np.sum(w * var_x_c, axis=0) / np.sum(weights)
    return mean, var_c, var_x
