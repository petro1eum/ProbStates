import numpy as np

from probstates.calculus import (
    l2_or, grad_l2_or, l2_and, grad_l2_and,
    shannon_entropy, d_shannon_entropy_dp,
    l4_quant_or, grad_l4_quant_or, phase_sensitivity,
    expected_l2_or_under_beta, mc_expected_l4_quant_or_under_beta,
    kappa_for_classical_or, rho_from_ps,
)


def test_l2_gradients():
    p1, p2 = 0.3, 0.4
    # OR
    val = l2_or(p1, p2)
    dp1, dp2 = grad_l2_or(p1, p2)
    # numerical check
    eps = 1e-6
    num_dp1 = (l2_or(p1 + eps, p2) - l2_or(p1 - eps, p2)) / (2 * eps)
    num_dp2 = (l2_or(p1, p2 + eps) - l2_or(p1, p2 - eps)) / (2 * eps)
    assert np.isclose(dp1, num_dp1, atol=1e-6)
    assert np.isclose(dp2, num_dp2, atol=1e-6)

    # AND
    val_and = l2_and(p1, p2)
    dp1_and, dp2_and = grad_l2_and(p1, p2)
    num_dp1_and = (l2_and(p1 + eps, p2) - l2_and(p1 - eps, p2)) / (2 * eps)
    num_dp2_and = (l2_and(p1, p2 + eps) - l2_and(p1, p2 - eps)) / (2 * eps)
    assert np.isclose(dp1_and, num_dp1_and, atol=1e-6)
    assert np.isclose(dp2_and, num_dp2_and, atol=1e-6)


def test_entropy_derivative():
    p = 0.37
    d = d_shannon_entropy_dp(p)
    eps = 1e-6
    num = (shannon_entropy(p + eps) - shannon_entropy(p - eps)) / (2 * eps)
    assert np.isclose(d, num, atol=1e-6)


def test_l4_quant_gradients():
    p1, p2 = 0.25, 0.49
    phi1, phi2 = 0.7, 0.2
    dp1, dp2, dphi1, dphi2 = grad_l4_quant_or(p1, phi1, p2, phi2)
    eps = 1e-6
    def f_p1(x):
        return l4_quant_or(x, phi1, p2, phi2)
    def f_p2(x):
        return l4_quant_or(p1, phi1, x, phi2)
    def f_phi1(x):
        return l4_quant_or(p1, x, p2, phi2)
    def f_phi2(x):
        return l4_quant_or(p1, phi1, p2, x)
    num_dp1 = (f_p1(p1 + eps) - f_p1(p1 - eps)) / (2 * eps)
    num_dp2 = (f_p2(p2 + eps) - f_p2(p2 - eps)) / (2 * eps)
    num_dphi1 = (f_phi1(phi1 + eps) - f_phi1(phi1 - eps)) / (2 * eps)
    num_dphi2 = (f_phi2(phi2 + eps) - f_phi2(phi2 - eps)) / (2 * eps)
    assert np.isclose(dp1, num_dp1, atol=1e-5)
    assert np.isclose(dp2, num_dp2, atol=1e-5)
    assert np.isclose(dphi1, num_dphi1, atol=1e-5)
    assert np.isclose(dphi2, num_dphi2, atol=1e-5)


def test_phase_sensitivity():
    p1, p2 = 0.36, 0.49
    dphi = 0.3
    s = phase_sensitivity(p1, p2, dphi)
    # should match |∂/∂Δφ|
    eps = 1e-6
    base = l4_quant_or(p1, 0.0, p2, dphi)
    base2 = l4_quant_or(p1, 0.0, p2, dphi + eps)
    num = abs((base2 - base) / eps)
    assert np.isclose(s, num, rtol=1e-4, atol=1e-5)


def test_expected_under_beta():
    a1, b1, a2, b2 = 12, 8, 7, 9
    # L2 analytic
    val = expected_l2_or_under_beta(a1, b1, a2, b2)
    # MC sanity (treat as L2 by ignoring interference)
    rng = np.random.default_rng(0)
    p1 = rng.beta(a1, b1, 20000)
    p2 = rng.beta(a2, b2, 20000)
    mc = np.mean(p1 + p2 - p1 * p2)
    assert np.isclose(val, mc, rtol=0.02)

    # L4 MC with kappa=0 should reduce to ~mu1+mu2
    mu1 = a1 / (a1 + b1)
    mu2 = a2 / (a2 + b2)
    val_l4 = mc_expected_l4_quant_or_under_beta(a1, b1, a2, b2, kappa=0.0, num_samples=30000)
    assert np.isclose(val_l4, mu1 + mu2, rtol=0.03)


def test_programmable_fusion_matches_classical_or():
    # choose two Beta priors, compute mu and rho, solve kappa*, check match
    a1, b1, a2, b2 = 5, 7, 9, 6
    rng = np.random.default_rng(1)
    ps1 = rng.beta(a1, b1, 50000)
    ps2 = rng.beta(a2, b2, 50000)
    mu1, mu2 = float(np.mean(ps1)), float(np.mean(ps2))
    rho1, rho2 = rho_from_ps(ps1), rho_from_ps(ps2)
    kappa = kappa_for_classical_or(mu1, mu2, rho1, rho2)
    # MC estimate for L4 at kappa*
    est = mc_expected_l4_quant_or_under_beta(a1, b1, a2, b2, kappa=kappa, num_samples=40000, rng=rng)
    # expected classical OR mean
    mu_or = mu1 + mu2 - mu1 * mu2
    assert np.isclose(est, mu_or, rtol=0.05)


