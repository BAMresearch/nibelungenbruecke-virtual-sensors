"""Uncertainty propagation of the embedded model-form parameter c_c.

The calibrated model (previous paper, Table 4 "Final") treats the wind
convection constant c_c as LogNormal with *linear-space* mean and standard
deviation; σ_ln = √ln(1+(σ/μ)²), μ_ln = ln μ - σ_ln²/2 (same convention as
``src/simulation/pce_builder.py``).

The pipeline builds an order-p PCE on the (p+1)-point Gaussian quadrature
of that LogNormal measure.  A degree-p polynomial interpolating the p+1
Gauss nodes has its mean and variance integrated exactly by the same rule,
so the PCE moments equal the weighted moments of the node outputs — which is
all that is needed here; no PCE coefficients are stored.
"""

from __future__ import annotations

import math

import numpy as np


def lognormal_gauss_rule(mean: float, std: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the n-point Gaussian rule for LogNormal(mean, std).

    Discretised Stieltjes procedure on a fine Gauss–Hermite grid of the
    underlying normal variable, then Golub–Welsch.  Reproduces chaospy's
    ``generate_quadrature(order, LogNormal, rule="gaussian")``.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    s_ln = math.sqrt(math.log1p((std / mean) ** 2))
    m_ln = math.log(mean) - 0.5 * s_ln**2
    xi, w = np.polynomial.hermite_e.hermegauss(200)
    w = w / w.sum()
    x = np.exp(m_ln + s_ln * xi)

    # Stieltjes: monic orthogonal polynomials w.r.t. the discrete measure
    alpha = np.zeros(n_nodes)
    beta = np.zeros(n_nodes)
    p_prev = np.zeros_like(x)
    p = np.ones_like(x)
    norm_prev = 1.0
    for k in range(n_nodes):
        norm = np.sum(w * p * p)
        alpha[k] = np.sum(w * x * p * p) / norm
        beta[k] = norm / norm_prev if k > 0 else norm
        p, p_prev = (x - alpha[k]) * p - (beta[k] if k > 0 else 0.0) * p_prev, p
        norm_prev = norm

    J = np.diag(alpha) + np.diag(np.sqrt(beta[1:]), 1) + np.diag(np.sqrt(beta[1:]), -1)
    nodes, vecs = np.linalg.eigh(J)
    weights = vecs[0, :] ** 2
    return nodes, weights / weights.sum()


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
    is exact for the degree-2 PCE (see the module doc).
    """
    w = weights.reshape((-1,) + (1,) * (values.ndim - 2))
    mean_x = values.mean(axis=-1)
    var_x_c = values.var(axis=-1)                 # population variance, per node
    mean, var_c = weighted_moments(mean_x, weights)
    var_x = np.sum(w * var_x_c, axis=0) / np.sum(weights)
    return mean, var_c, var_x
