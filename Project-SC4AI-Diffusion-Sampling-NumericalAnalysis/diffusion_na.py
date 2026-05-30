"""diffusion_na —— 反向扩散采样的数值分析 · 核心库（经 pytest 数值自检）。

统一时间约定（全程严格一致，已数值核验）
------------------------------------------------
- VP 前向 SDE:  dX = -1/2 beta(t) X dt + sqrt(beta(t)) dW,  t in (0,1].
- beta(t) = 0.1 + 19.9 t,  B(t) = ∫_0^t beta = 0.1 t + 9.95 t^2.
- 数据 X0 ~ N(0, s0^2 I)，默认 s0 = DATA_STD = 0.5（s0 = 1 退化，禁用）。
- 边缘 v(t) = 1 + (s0^2 - 1) e^{-B(t)}，精确 score  s*(x,t) = -x / v(t)（线性）。
- 反向漂移系数 a(t) = beta(t) (1/v(t) - 1/2) > 0。
- 反向采样网格 t_n: 1 -> eps，正步长 h = t_n - t_{n+1} > 0。
    EM 一步:        x_{n+1} = (1 - a(t_n) h) x_n + sqrt(beta(t_n) h) z
    方差递推:        V_{n+1} = (1 - a(t_n) h)^2 V_n + beta(t_n) h
- 反向时间 tau = 1 - t:  dV/dtau = -2 a(1-tau) V + beta(1-tau)  （★ 含因子 2）
- 概率流 ODE:  前向 f_pf(t,x) = 1/2 beta(t) (1/v(t) - 1) x；
    高斯闭式精确解  x(t) = sqrt(v(t)/v(1)) x(1)。
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp

BETA_MIN = 0.1
BETA_MAX = 20.0
DATA_STD = 0.5
EPS = 1e-3
SEED = 2026

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2.0 改名


# ----------------------------------------------------------------------
# 1. 调度、边缘分布与精确 score
# ----------------------------------------------------------------------
def beta(t):
    return BETA_MIN + (BETA_MAX - BETA_MIN) * np.asarray(t, dtype=float)


def int_beta(t):
    """B(t) = ∫_0^t beta(s) ds."""
    t = np.asarray(t, dtype=float)
    return BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2


def alpha(t):
    """转移核均值尺度 e^{-B/2}."""
    return np.exp(-0.5 * int_beta(t))


def sigma2(t):
    """转移核方差 1 - e^{-B}（数值稳定）."""
    return -np.expm1(-int_beta(t))


def marginal_var(t, s0=DATA_STD):
    """各向同性高斯数据的边缘方差 v(t)."""
    return 1.0 + (s0 ** 2 - 1.0) * np.exp(-int_beta(t))


def _log_marginal_gaussian(x, t, s0=DATA_STD):
    x = np.asarray(x, dtype=float)
    v = marginal_var(t, s0)
    d = x.size
    return -0.5 * np.sum(x ** 2) / v - 0.5 * d * np.log(2 * np.pi * v)


def exact_score_gaussian(x, t, s0=DATA_STD):
    """∇ log p(x,t) = -x / v(t)（线性）."""
    return -np.asarray(x, dtype=float) / marginal_var(t, s0)


# ----------------------------------------------------------------------
# 2. 高斯混合的闭式边缘与 score（批量 [N,d] -> [N,d]）
# ----------------------------------------------------------------------
def _component_var(t, s0):
    return alpha(t) ** 2 * s0 ** 2 + sigma2(t)


def gmm_log_marginal(x, t, means, weights, s0=0.4):
    x = np.atleast_2d(np.asarray(x, float))            # [N,d]
    means = np.asarray(means, float)                   # [K,d]
    w = np.asarray(weights, float)                     # [K]
    vt = _component_var(t, s0)
    d = x.shape[1]
    mu = alpha(t) * means                              # [K,d]
    sq = np.sum((x[:, None, :] - mu[None, :, :]) ** 2, axis=-1)  # [N,K]
    logc = -0.5 * d * np.log(2 * np.pi * vt)
    logp_k = logc - 0.5 * sq / vt                      # [N,K]
    return logsumexp(logp_k, b=w[None, :], axis=1)     # [N]


def exact_score_gmm(x, t, means, weights, s0=0.4):
    x = np.atleast_2d(np.asarray(x, float))            # [N,d]
    means = np.asarray(means, float)                   # [K,d]
    w = np.asarray(weights, float)                     # [K]
    vt = _component_var(t, s0)
    mu = alpha(t) * means                              # [K,d]
    diff = mu[None, :, :] - x[:, None, :]              # [N,K,d]
    sq = np.sum(diff ** 2, axis=-1)                    # [N,K]
    logp = np.log(w)[None, :] - 0.5 * sq / vt          # [N,K]
    logp = logp - logp.max(axis=1, keepdims=True)
    r = np.exp(logp)
    r = r / r.sum(axis=1, keepdims=True)               # 后验权重 [N,K]
    return np.sum(r[:, :, None] * diff / vt, axis=1)   # [N,d]


# ----------------------------------------------------------------------
# 3. 反向动力学与精确方差参照（含因子 2）
# ----------------------------------------------------------------------
def reverse_drift_coeff(t, s0=DATA_STD):
    """a(t) = beta(t)(1/v(t) - 1/2)（s0<1 时 v(t)<1 => a(t)>0 全程成立）。"""
    return beta(t) * (1.0 / marginal_var(t, s0) - 0.5)


def reverse_drift_coeff_prime(t, s0=DATA_STD):
    """a'(t) 闭式导数（供解析 V''）：a=beta(1/v-1/2)，beta'=BMAX-BMIN，v'=-(v-1)beta。"""
    v = marginal_var(t, s0)
    b = beta(t)
    bp = BETA_MAX - BETA_MIN
    return bp * (1.0 / v - 0.5) + b ** 2 * (v - 1.0) / v ** 2


def reverse_variance_exact(t_end, s0=DATA_STD, t_start=1.0, V_start=None, dense=False):
    """高精度求解反向方差 ODE dV/dtau = -2 a V + beta（含因子 2）。

    V_start=None 时取真值起点 marginal_var(t_start)（隔离离散误差）。
    dense=True 返回 OdeSolution（供取二阶导）。
    """
    if V_start is None:
        V_start = marginal_var(t_start, s0)

    def rhs(tau, V):
        t = t_start - tau
        a = reverse_drift_coeff(t, s0)
        return [-2.0 * a * V[0] + beta(t)]            # ★ 因子 2

    tau_end = t_start - t_end
    sol = solve_ivp(rhs, [0.0, tau_end], [float(V_start)],
                    rtol=1e-10, atol=1e-10, dense_output=True)
    return sol if dense else float(sol.y[0, -1])


# ----------------------------------------------------------------------
# 4. 反向 EM 方差确定性递推 + 随机采样器（互校）
# ----------------------------------------------------------------------
def em_variance_recursion(N, eps=EPS, s0=DATA_STD, t_start=1.0, V_start=1.0):
    ts = np.linspace(t_start, eps, N + 1)
    V = float(V_start)
    for i in range(N):
        t = ts[i]
        h = ts[i] - ts[i + 1]
        a = reverse_drift_coeff(t, s0)
        V = (1.0 - a * h) ** 2 * V + beta(t) * h
    return V


def em_sample(N, n_samples=10000, eps=EPS, s0=DATA_STD, dim=1, seed=0, x_start=None):
    rng = np.random.default_rng(seed)
    ts = np.linspace(1.0, eps, N + 1)
    if x_start is None:
        x = rng.standard_normal((n_samples, dim))      # 先验 N(0,I)
    else:
        x = np.array(x_start, dtype=float)
    for i in range(N):
        t = ts[i]
        h = ts[i] - ts[i + 1]
        a = reverse_drift_coeff(t, s0)
        x = (1.0 - a * h) * x + np.sqrt(beta(t) * h) * rng.standard_normal(x.shape)
    return x


# ----------------------------------------------------------------------
# 5. O(h) 偏差律与修正方程系数（后向误差分析）
# ----------------------------------------------------------------------
def bias_coeff_empirical(eps=EPS, s0=DATA_STD, Ns=(800, 1600, 3200, 6400)):
    """Richardson 外推得 (V_N - target)/h 在 h->0 的极限 c_emp."""
    target = reverse_variance_exact(eps, s0=s0, V_start=marginal_var(1.0, s0))
    Vs = np.array([em_variance_recursion(N, eps=eps, s0=s0, V_start=marginal_var(1.0, s0))
                   for N in Ns])
    hs = (1.0 - eps) / np.array(Ns, dtype=float)
    ratios = (Vs - target) / hs
    A = np.vstack([np.ones_like(hs), hs]).T
    return float(np.linalg.lstsq(A, ratios, rcond=None)[0][0])


def bias_coeff_theory(eps=EPS, s0=DATA_STD, M=40000):
    """闭式领头阶偏差系数 c = ∫ Phi(T,s)[a^2 V - 1/2 V''] ds，Phi=exp(∫ -2a)."""
    T = 1.0 - eps
    taus = np.linspace(0.0, T, M)
    t = 1.0 - taus
    a = reverse_drift_coeff(t, s0)
    sol = reverse_variance_exact(eps, s0=s0, V_start=marginal_var(1.0, s0), dense=True)
    V = sol.sol(taus)[0]
    # 解析 V''(τ)：V'=F=-2aV+β ⇒ V''=dF/dτ=2a'(1-τ)V - 2a(1-τ)F - β'(1-τ)（机器精度，与差分步长无关）
    F = -2.0 * a * V + beta(t)
    ap = reverse_drift_coeff_prime(t, s0)
    bp = BETA_MAX - BETA_MIN
    Vpp = 2.0 * ap * V - 2.0 * a * F - bp
    H = a ** 2 * V - 0.5 * Vpp                          # 修正项（含离散耦合 G=a^2 V）
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(taus))])  # ∫_0^s a
    Phi = np.exp(-2.0 * (cum[-1] - cum))                # exp(∫_s^T -2a)
    return float(_trapz(Phi * H, taus))


# ----------------------------------------------------------------------
# 6. 改进采样器：半隐式/指数处理线性漂移 + 概率流 ODE
# ----------------------------------------------------------------------
def semiimplicit_variance_recursion(N, eps=EPS, s0=DATA_STD, t_start=1.0, V_start=1.0):
    """每步对线性漂移做常系数 OU 精确步（降低误差常数，整体仍 O(h)）。"""
    ts = np.linspace(t_start, eps, N + 1)
    V = float(V_start)
    for i in range(N):
        t = ts[i]
        h = ts[i] - ts[i + 1]
        a = reverse_drift_coeff(t, s0)
        decay = np.exp(-a * h)
        if abs(a) > 1e-8:
            noise = beta(t) * (1.0 - np.exp(-2.0 * a * h)) / (2.0 * a)
        else:
            noise = beta(t) * h
        V = decay ** 2 * V + noise
    return V


def pf_ode_exact(x_start, eps=EPS, s0=DATA_STD, t_start=1.0):
    """高斯概率流 ODE 闭式精确解 x(eps) = sqrt(v(eps)/v(1)) x(1)（机器精度参照）。"""
    return float(np.sqrt(marginal_var(eps, s0) / marginal_var(t_start, s0)) * x_start)


def pf_ode_drift(t, x, s0=DATA_STD):
    """前向概率流漂移 f_pf = 1/2 beta (1/v - 1) x."""
    return 0.5 * beta(t) * (1.0 / marginal_var(t, s0) - 1.0) * np.asarray(x, float)


def pf_ode_sample(N, method, x_start, eps=EPS, s0=DATA_STD):
    """反向积分概率流 ODE（t: 1 -> eps，正步长 h）。method ∈ {'euler','heun'}."""
    ts = np.linspace(1.0, eps, N + 1)
    x = np.asarray(x_start, dtype=float)
    for i in range(N):
        t = ts[i]
        h = ts[i] - ts[i + 1]
        f1 = pf_ode_drift(t, x, s0)
        if method == "euler":
            x = x - h * f1
        elif method == "heun":
            xp = x - h * f1
            f2 = pf_ode_drift(ts[i + 1], xp, s0)
            x = x - 0.5 * h * (f1 + f2)
        else:
            raise ValueError(f"unknown method: {method}")
    return float(x) if np.ndim(x) == 0 else x


# ----------------------------------------------------------------------
# 6b. 通用 score-based 反向采样器（解析或网络 score 均可，用于多维实验）
# ----------------------------------------------------------------------
def reverse_sample(score_fn, N, method="em", x0=None, n_samples=1000, dim=2,
                   eps=EPS, seed=0, return_nfe=False):
    """反向积分（t:1->eps，正步长 h）。score_fn(x[M,d], t) -> [M,d]。

    method:
      'em'       反向 SDE Euler-Maruyama:  x += h(½β x + β s) + sqrt(βh) z
      'pf_euler' 概率流 ODE 反向 Euler:     x += h(½β x + ½β s)
      'pf_heun'  概率流 ODE 反向 Heun(2阶): 两段平均
    与解析采样器一致（s=-x/v 时 'em' 化为 (1-ah)x+√(βh)z）。
    """
    rng = np.random.default_rng(seed)
    ts = np.linspace(1.0, eps, N + 1)
    if x0 is None:
        x = rng.standard_normal((n_samples, dim))
    else:
        x = np.asarray(x0, dtype=float).copy()
    nfe = 0
    for i in range(N):
        t = ts[i]
        h = ts[i] - ts[i + 1]
        b = beta(t)
        s = score_fn(x, t); nfe += 1
        if method == "em":
            x = x + h * (0.5 * b * x + b * s) + np.sqrt(b * h) * rng.standard_normal(x.shape)
        elif method == "pf_euler":
            x = x + h * (0.5 * b * x + 0.5 * b * s)
        elif method == "pf_heun":
            x1 = x + h * (0.5 * b * x + 0.5 * b * s)
            t2 = ts[i + 1]; b2 = beta(t2)
            s2 = score_fn(x1, t2); nfe += 1
            x = x + 0.5 * h * ((0.5 * b * x + 0.5 * b * s) + (0.5 * b2 * x1 + 0.5 * b2 * s2))
        else:
            raise ValueError(f"unknown method: {method}")
    return (x, nfe) if return_nfe else x


# ----------------------------------------------------------------------
# 7. 度量：Bures-W2、矩误差、sliced-Wasserstein、energy distance
# ----------------------------------------------------------------------
def _psd_sqrt(C):
    C = 0.5 * (C + C.T)
    w, Q = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (Q * np.sqrt(w)) @ Q.T


def bures_w2(m1, C1, m2, C2):
    """两高斯间 2-Wasserstein 距离（闭式 Bures，含对称化+特征值截断）。"""
    m1, m2 = np.atleast_1d(m1).astype(float), np.atleast_1d(m2).astype(float)
    C1, C2 = np.atleast_2d(C1).astype(float), np.atleast_2d(C2).astype(float)
    s1 = _psd_sqrt(C1)
    cross = _psd_sqrt(s1 @ C2 @ s1)
    bures = np.trace(C1 + C2 - 2.0 * cross)
    return float(np.sqrt(max(np.sum((m1 - m2) ** 2) + float(bures), 0.0)))


def moment_error(samples, m_true, v_true):
    """前两阶矩误差（均值 + 方差），逐维平均。"""
    samples = np.atleast_2d(samples)
    return float(np.abs(samples.mean(0) - m_true).mean()
                 + np.abs(samples.var(0) - v_true).mean())


def sliced_wasserstein(X, Y, n_proj=200, seed=0):
    """2-D 点云主指标：sliced-Wasserstein-1（O(n log n)）。"""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    th = rng.standard_normal((d, n_proj))
    th /= np.linalg.norm(th, axis=0, keepdims=True)
    px = np.sort(X @ th, axis=0)
    py = np.sort(Y @ th, axis=0)
    if px.shape[0] != py.shape[0]:
        m = min(px.shape[0], py.shape[0])
        q = np.linspace(0.0, 1.0, m)
        px = np.stack([np.quantile(px[:, j], q) for j in range(n_proj)], axis=1)
        py = np.stack([np.quantile(py[:, j], q) for j in range(n_proj)], axis=1)
    return float(np.mean(np.abs(px - py)))


def energy_distance(X, Y, max_n=512, seed=0):
    """辅助指标：energy distance；子采样 n<=max_n 避免 O(n^2) 内存。"""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    def pd(A, B):
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) + 1e-12).mean()

    return float(2 * pd(X, Y) - pd(X, X) - pd(Y, Y))


# ----------------------------------------------------------------------
# 8. 刚性与显式 Euler 稳定域
# ----------------------------------------------------------------------
def stiffness(t, s0=DATA_STD):
    """L(t) = |a(t)|（反向漂移 Jacobian 谱）。"""
    return np.abs(reverse_drift_coeff(t, s0))


def critical_dt(t, s0=DATA_STD):
    """显式 Euler 稳定步长界 2/|a(t)|。"""
    return 2.0 / np.abs(reverse_drift_coeff(t, s0))


def em_blows_up(N, eps=EPS, s0=DATA_STD, threshold=10.0):
    """N 太小（h 太大）时 |1-ah|>1 放大，方差爆/不收敛 -> True。"""
    V = em_variance_recursion(N, eps=eps, s0=s0, V_start=marginal_var(1.0, s0))
    target = marginal_var(eps, s0)
    return (not np.isfinite(V)) or (V > threshold * target) or (V < 0.0)


# ----------------------------------------------------------------------
# 9. Fokker–Planck（log-FPE）残差算子（torch autograd, float64 CPU）
# ----------------------------------------------------------------------
def fpe_residual_gaussian(x, t, s0=DATA_STD, perturb=0.0):
    """log-FPE 残差  ∂_t log p - 1/2 beta [ d + x·s + ||s||^2 + ∇·s ]，s=∇log p。

    解析高斯 + 真值 score (perturb=0) 时应 ≈ 0；perturb≠0 残差随之增大。
    """
    import torch

    x = np.atleast_2d(np.asarray(x, float))
    N, d = x.shape
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    tt = torch.full((N, 1), float(t), dtype=torch.float64, requires_grad=True)
    Bt = BETA_MIN * tt + 0.5 * (BETA_MAX - BETA_MIN) * tt ** 2     # [N,1]
    v = 1.0 + (s0 ** 2 - 1.0) * torch.exp(-Bt)                     # [N,1]
    logp = (-0.5 * (xt ** 2).sum(1, keepdim=True) / v
            - 0.5 * d * torch.log(2 * np.pi * v))                 # [N,1]
    dlogp_dt = torch.autograd.grad(logp.sum(), tt, create_graph=True)[0].squeeze(1)  # [N]
    s = -xt / v + perturb                                         # [N,d]
    div_s = torch.zeros(N, dtype=torch.float64)
    for i in range(d):
        gi = torch.autograd.grad(s[:, i].sum(), xt, create_graph=True)[0][:, i]
        div_s = div_s + gi
    xs = (xt * s).sum(1)
    s2 = (s ** 2).sum(1)
    beta_t = BETA_MIN + (BETA_MAX - BETA_MIN) * float(t)
    res = dlogp_dt - 0.5 * beta_t * (d + xs + s2 + div_s)
    return res.detach().numpy()


# ----------------------------------------------------------------------
# 10. 小 score 网络与 DSM 训练（网络迁移实验用；支持 MPS）
# ----------------------------------------------------------------------
def _make_score_net(dim=1, hidden=64):
    import torch.nn as nn

    class ScoreNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, hidden), nn.LogSigmoid(),
                nn.Linear(hidden, hidden), nn.LogSigmoid(),
                nn.Linear(hidden, hidden), nn.LogSigmoid(),
                nn.Linear(hidden, dim),
            )

        def forward(self, x, t):
            return self.net(__import__("torch").cat([x, t], dim=1))

    return ScoreNet()


def _dsm_train(sampler, dim, n_iters, device, seed, lr, batch, hidden):
    import torch

    torch.manual_seed(seed)
    dev = torch.device(device)
    net = _make_score_net(dim=dim, hidden=hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(n_iters):
        x0 = sampler(batch, g)                                    # [batch,dim] cpu f32
        t = torch.rand(batch, 1, generator=g) * (1 - 2e-3) + 1e-3
        Bt = BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2
        a_t = torch.exp(-0.5 * Bt)
        var_t = -torch.expm1(-Bt)
        noise = torch.randn(batch, dim, generator=g)
        xt = a_t * x0 + torch.sqrt(var_t) * noise
        target = -noise / torch.sqrt(var_t)                       # (mu - xt)/var
        xt, t, target, var_t = (z.to(dev) for z in (xt, t, target, var_t))
        s = net(xt, t)
        loss = (var_t * (s - target) ** 2).mean()                 # sigma^2-加权 DSM
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    net.eval()
    net.losses_ = losses
    return net


def train_score_net_gaussian(s0=DATA_STD, n_iters=4000, device="cpu", seed=0,
                             dim=1, lr=3e-3, batch=512, hidden=64):
    import torch
    return _dsm_train(lambda b, g: s0 * torch.randn(b, dim, generator=g),
                      dim, n_iters, device, seed, lr, batch, hidden)


def make_swiss_roll_2d(n, seed=0, scale=0.1):
    from sklearn.datasets import make_swiss_roll
    X, _ = make_swiss_roll(n, noise=1.0, random_state=seed)
    X = X[:, [0, 2]] * scale
    return X.astype(np.float32)


def train_score_net_data(X, n_iters=8000, device="cpu", seed=0,
                         lr=3e-3, batch=256, hidden=64):
    import torch
    Xt = torch.tensor(np.asarray(X, np.float32))
    dim = Xt.shape[1]

    def sampler(b, g):
        idx = torch.randint(0, Xt.shape[0], (b,), generator=g)
        return Xt[idx]

    return _dsm_train(sampler, dim, n_iters, device, seed, lr, batch, hidden)


def eval_score_net(net, x, t, device="cpu"):
    import torch
    dev = torch.device(device)
    x = torch.tensor(np.atleast_2d(np.asarray(x, float)), dtype=torch.float32, device=dev)
    tt = torch.full((x.shape[0], 1), float(t), dtype=torch.float32, device=dev)
    with torch.no_grad():
        s = net(x, tt).cpu().numpy()
    return s


def pick_device():
    """优先 MPS（Apple Silicon），否则 CPU。"""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"
