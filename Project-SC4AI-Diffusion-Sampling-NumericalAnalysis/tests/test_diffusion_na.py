"""diffusion_na 的数值自检（每个解析公式都配一个可证伪的数值校验）。"""
import numpy as np
import pytest

import diffusion_na as dna


# ---------- 1. 调度与边缘 ----------
def test_int_beta_endpoints():
    assert dna.int_beta(0.0) == pytest.approx(0.0)
    assert dna.int_beta(1.0) == pytest.approx(0.1 + 9.95)        # 10.05


def test_marginal_var_range():
    s0 = 0.5
    assert dna.marginal_var(1e-6, s0) == pytest.approx(s0 ** 2, abs=1e-3)
    assert dna.marginal_var(1.0, s0) == pytest.approx(1.0, abs=1e-3)


def test_transition_kernel_moments_statistical():
    rng = np.random.default_rng(0)
    t, x0 = 0.4, 1.3
    samples = dna.alpha(t) * x0 + np.sqrt(dna.sigma2(t)) * rng.standard_normal(200000)
    assert samples.mean() == pytest.approx(dna.alpha(t) * x0, abs=5e-3)
    assert samples.var() == pytest.approx(dna.sigma2(t), abs=5e-3)


def test_exact_score_gaussian_matches_finite_difference():
    s0, t = 0.5, 0.3
    x = np.array([0.7, -1.1])
    eps = 1e-5
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        grad[i] = (dna._log_marginal_gaussian(xp, t, s0)
                   - dna._log_marginal_gaussian(xm, t, s0)) / (2 * eps)
    assert np.allclose(grad, dna.exact_score_gaussian(x, t, s0), atol=1e-4)


# ---------- 2. 高斯混合 ----------
def test_gmm_score_matches_finite_difference():
    means = np.array([[-1.5, 0.0], [1.5, 0.0]])
    weights = np.array([0.5, 0.5])
    s0, t = 0.4, 0.25
    x = np.array([0.3, -0.6])

    def lp(xx):
        return float(dna.gmm_log_marginal(xx, t, means, weights, s0)[0])

    eps = 1e-5
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        grad[i] = (lp(xp) - lp(xm)) / (2 * eps)
    assert np.allclose(grad, dna.exact_score_gmm(x, t, means, weights, s0)[0], atol=1e-4)


def test_gmm_score_batched_shape():
    means = np.array([[-1.5, 0.0], [1.5, 0.0]])
    weights = np.array([0.5, 0.5])
    x = np.random.default_rng(0).standard_normal((7, 2))
    assert dna.exact_score_gmm(x, 0.3, means, weights, 0.4).shape == (7, 2)


# ---------- 3. 反向方差 ODE（因子 2） ----------
def test_reverse_variance_exact_matches_marginal():
    s0, eps = 0.5, 1e-3
    Vend = dna.reverse_variance_exact(eps, s0=s0)
    assert Vend == pytest.approx(dna.marginal_var(eps, s0), rel=1e-3)


def test_reverse_variance_factor_two_not_one():
    # 正确(含因子2)≈0.25；漏因子2≈0.76。锁死因子 2。
    s0, eps = 0.5, 1e-3
    assert dna.reverse_variance_exact(eps, s0=s0) < 0.4


# ---------- 4. EM 递推 vs 蒙特卡洛 ----------
def test_em_recursion_matches_monte_carlo():
    s0, eps, N = 0.5, 1e-3, 200
    Vrec = dna.em_variance_recursion(N, eps=eps, s0=s0, V_start=dna.marginal_var(1.0, s0))
    Xend = dna.em_sample(N, n_samples=200000, eps=eps, s0=s0, dim=1, seed=0)
    assert Xend.var() == pytest.approx(Vrec, rel=0.05)


# ---------- 5. O(h) 偏差律与修正方程系数 ----------
def test_bias_law_slope_is_one():
    s0, eps = 0.5, 1e-3
    Ns = np.array([50, 100, 200, 400, 800, 1600])
    target = dna.reverse_variance_exact(eps, s0=s0, V_start=dna.marginal_var(1.0, s0))
    errs = np.array([abs(dna.em_variance_recursion(N, eps=eps, s0=s0,
                                                   V_start=dna.marginal_var(1.0, s0)) - target)
                     for N in Ns])
    hs = (1.0 - eps) / Ns
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 0.85 < slope < 1.15


def test_modified_eq_coefficient_consistency():
    s0, eps = 0.5, 1e-3
    c_theory = dna.bias_coeff_theory(eps=eps, s0=s0)
    c_emp = dna.bias_coeff_empirical(eps=eps, s0=s0)
    assert abs(c_theory - c_emp) / abs(c_emp) < 0.1


# ---------- 6. 改进采样器 ----------
def test_exponential_reduces_error_constant():
    s0, eps, N = 0.5, 1e-3, 64
    target = dna.reverse_variance_exact(eps, s0=s0)
    Vexp = dna.semiimplicit_variance_recursion(N, eps=eps, s0=s0, V_start=dna.marginal_var(1.0, s0))
    Vem = dna.em_variance_recursion(N, eps=eps, s0=s0, V_start=dna.marginal_var(1.0, s0))
    assert abs(Vexp - target) < abs(Vem - target)


def test_pf_ode_exact_closed_form():
    s0, eps = 0.5, 1e-3
    assert dna.pf_ode_exact(2.0, eps=eps, s0=s0) == pytest.approx(
        np.sqrt(dna.marginal_var(eps, s0) / dna.marginal_var(1.0, s0)) * 2.0, rel=1e-12)


def test_pf_ode_heun_higher_order_than_euler():
    s0, eps = 0.5, 1e-3
    ref = dna.pf_ode_exact(2.0, eps=eps, s0=s0)
    e_euler = abs(dna.pf_ode_sample(64, "euler", x_start=2.0, eps=eps, s0=s0) - ref)
    e_heun = abs(dna.pf_ode_sample(64, "heun", x_start=2.0, eps=eps, s0=s0) - ref)
    assert e_heun < e_euler


# ---------- 7. 度量 ----------
def test_bures_w2_self_zero_and_1d():
    assert dna.bures_w2(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2)) == pytest.approx(0, abs=1e-9)
    assert dna.bures_w2(np.array([0.]), np.array([[4.]]),
                        np.array([1.]), np.array([[1.]])) ** 2 == pytest.approx((0 - 1) ** 2 + (2 - 1) ** 2)


def test_sliced_w_and_energy_distance_separate():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2000, 2))
    b = rng.standard_normal((2000, 2))
    assert dna.sliced_wasserstein(a, b) < dna.sliced_wasserstein(a, b + 3.0)
    assert dna.energy_distance(a, b) < dna.energy_distance(a, b + 3.0)


# ---------- 8. 刚性与稳定域 ----------
def test_stiffness_peaks_near_t1():
    ts = np.linspace(1e-3, 1.0, 200)
    L = dna.stiffness(ts, 0.5)
    assert ts[np.argmax(L)] > 0.7


def test_critical_step_predicts_blowup():
    s0 = 0.5
    t = 1.0
    a = dna.reverse_drift_coeff(t, s0)
    dt_crit = 2.0 / abs(a)
    assert abs(1 - a * (dt_crit * 1.1)) > 1
    assert abs(1 - a * (dt_crit * 0.5)) < 1
    assert dna.em_blows_up(3, s0=s0) or dna.em_blows_up(2, s0=s0)   # 极粗网格不稳


# ---------- 9. FPE 残差 ----------
def test_fpe_residual_zero_on_exact_score():
    s0, t = 0.5, 0.3
    x = np.array([[0.5, -0.4], [1.0, 0.2]])
    res = dna.fpe_residual_gaussian(x, t, s0)
    assert np.abs(res).max() < 1e-3


def test_fpe_residual_detects_perturbation():
    s0, t = 0.5, 0.3
    x = np.array([[0.5, -0.4]])
    assert (np.abs(dna.fpe_residual_gaussian(x, t, s0, perturb=0.3)).max()
            > np.abs(dna.fpe_residual_gaussian(x, t, s0, perturb=0.0)).max())


# ---------- 10. score 网络 ----------
def test_trained_score_approximates_exact_on_gaussian():
    net = dna.train_score_net_gaussian(s0=0.5, n_iters=4000, device="cpu", seed=0)
    t = 0.3
    xs = np.linspace(-1, 1, 11).reshape(-1, 1)
    pred = dna.eval_score_net(net, xs, t)
    exact = dna.exact_score_gaussian(xs, t, 0.5)
    assert np.mean((pred - exact) ** 2) < 0.2
