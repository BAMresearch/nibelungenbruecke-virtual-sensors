"""Uncertainty propagation of the embedded model-form parameter c_c.

The calibrated model treats the wind convection constant c_c as LogNormal,
ln c_c ~ N(ln median, σ_ln²).  As in the calibration, c_c is written as a
function of a standard normal germ, c_c = median · exp(σ_ln ξ), and the
pipeline builds an order-p Hermite PCE in ξ on the (p+1)-point Gauss–Hermite
rule (same construction as ``src/simulation/pce_builder.py``).  A degree-p
polynomial interpolating the p+1 Gauss nodes has its mean and variance
integrated exactly by the same rule, so the PCE moments equal the weighted
moments of the node outputs — which is all that is needed here; no PCE
coefficients are stored.
"""

from __future__ import annotations

import numpy as np


def lognormal_hermite_rule(median: float, sigma_ln: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Values c_i = median · exp(σ_ln ξ_i) and weights of the n-point Gauss–Hermite rule in ξ ~ N(0, 1).

    Reproduces the nodes of chaospy's ``generate_quadrature(n - 1, Normal(0, 1), rule="gaussian")``
    mapped to c_c; the weights sum to 1.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    xi, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return median * np.exp(sigma_ln * xi), w / w.sum()


def weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance over axis 0 (the quadrature nodes)."""
    w = weights.reshape((-1,) + (1,) * (values.ndim - 1))
    mean = np.sum(w * values, axis=0)
    var = np.sum(w * (values - mean) ** 2, axis=0)
    return mean, var


def total_variance_moments(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Law of total variance over the model parameter c (axis 0) and the sensor position x (last axis).

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
