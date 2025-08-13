"""
Calculus utilities for ProbStates: gradients, sensitivities, and meta-level estimators.

This module provides:
- Analytic gradients for L2 (ProbabilisticBit) and L4 (PhaseState, mode='quant') operations
- Entropy derivatives for H2 and numerical derivative helper
- Phase sensitivity utilities
- Monte Carlo estimators for expectation under Beta priors on p

Note: For L4 modes 'norm' and 'weight', use numerical derivatives via central differences.
"""

from __future__ import annotations

from typing import Callable, Tuple, Sequence
import numpy as np


# ---------- Numerical derivatives ----------

def central_diff(f: Callable[[float], float], x: float, eps: float = 1e-6) -> float:
    """Central finite difference derivative for scalar function f."""
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


# ---------- L2 operations (independent) ----------

def l2_or(p1: float, p2: float) -> float:
    return p1 + p2 - p1 * p2


def grad_l2_or(p1: float, p2: float) -> Tuple[float, float]:
    """Gradients (∂/∂p1, ∂/∂p2) for independent OR."""
    return 1.0 - p2, 1.0 - p1


def l2_and(p1: float, p2: float) -> float:
    return p1 * p2


def grad_l2_and(p1: float, p2: float) -> Tuple[float, float]:
    return p2, p1


# ---------- Entropy (H2) ----------

def shannon_entropy(p: float) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def d_shannon_entropy_dp(p: float) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log((1 - p) / p)


# ---------- L4 'quant' operation ----------

def l4_quant_or(p1: float, phi1: float, p2: float, phi2: float) -> float:
    return p1 + p2 + 2.0 * np.sqrt(max(p1, 0.0) * max(p2, 0.0)) * np.cos(phi1 - phi2)


def grad_l4_quant_or(
    p1: float, phi1: float, p2: float, phi2: float
) -> Tuple[float, float, float, float]:
    """Gradients (∂/∂p1, ∂/∂p2, ∂/∂phi1, ∂/∂phi2) for L4 'quant' OR."""
    p1c = max(p1, 1e-18)
    p2c = max(p2, 1e-18)
    dphi = phi1 - phi2
    dp1 = 1.0 + np.sqrt(p2c / p1c) * np.cos(dphi)
    dp2 = 1.0 + np.sqrt(p1c / p2c) * np.cos(dphi)
    dphi1 = -2.0 * np.sqrt(p1c * p2c) * np.sin(dphi)
    dphi2 = +2.0 * np.sqrt(p1c * p2c) * np.sin(dphi)
    return dp1, dp2, dphi1, dphi2


def phase_sensitivity(p1: float, p2: float, dphi: float) -> float:
    """|∂p_out/∂Δφ| for L4 'quant'."""
    return 2.0 * np.sqrt(max(p1, 0.0) * max(p2, 0.0)) * abs(np.sin(dphi))


# ---------- Meta-level estimators ----------

def expected_l2_or_under_beta(
    a1: float, b1: float, a2: float, b2: float
) -> float:
    """E[p1 ∨ p2] with independent p1~Beta(a1,b1), p2~Beta(a2,b2)."""
    mu1 = a1 / (a1 + b1)
    mu2 = a2 / (a2 + b2)
    # independence: E[p1 p2] = E[p1]E[p2]
    return mu1 + mu2 - mu1 * mu2


def mc_expected_l4_quant_or_under_beta(
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    kappa: float = 1.0,
    num_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo estimate of E[p⊕] for L4 'quant' with p~Beta and E[cosΔφ]=kappa.

    If kappa is provided, we sample cos(Δφ) from a two-point mixture giving E[cosΔφ]=κ:
      cosΔφ = +1 with prob (1+κ)/2, and cosΔφ = -1 with prob (1-κ)/2.
    This ensures the desired mean coherence without coupling to p.
    """
    if rng is None:
        rng = np.random.default_rng()
    p1 = rng.beta(a1, b1, size=num_samples)
    p2 = rng.beta(a2, b2, size=num_samples)
    kappa = float(np.clip(kappa, -1.0, 1.0))
    # Sample cos(Δφ) directly from a Bernoulli mixture to match E[cosΔφ]=κ
    probs_pos = (1.0 + kappa) / 2.0
    signs = rng.random(num_samples) < probs_pos
    cos_term = np.where(signs, 1.0, -1.0)
    vals = p1 + p2 + 2.0 * np.sqrt(p1 * p2) * cos_term
    return float(np.mean(vals))


def kappa_for_classical_or(mu1: float, mu2: float, rho1: float = 1.0, rho2: float = 1.0) -> float:
    """Target κ* to match E[L4-quant OR] to classical OR in expectation.

    κ* = - sqrt(mu1*mu2) / (2 ρ1 ρ2)
    Clipped to [-1, 1].
    """
    if mu1 <= 0 or mu2 <= 0:
        return 0.0
    kappa = - np.sqrt(mu1 * mu2) / (2.0 * max(rho1, 1e-12) * max(rho2, 1e-12))
    return float(np.clip(kappa, -1.0, 1.0))


def kappa_from_phases(delta_phases: Sequence[float]) -> float:
    delta_phases = np.asarray(list(delta_phases), dtype=float)
    return float(np.mean(np.cos(delta_phases)))


def rho_from_ps(ps: Sequence[float]) -> float:
    ps = np.asarray(list(ps), dtype=float)
    mu = float(np.mean(ps))
    if mu <= 0:
        return 0.0
    return float(np.mean(np.sqrt(np.clip(ps, 0.0, 1.0))) / np.sqrt(mu))


