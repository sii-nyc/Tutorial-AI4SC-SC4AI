# 反向扩散采样的数值分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一份内容丰富、可证伪的中文 Jupyter Notebook，把反向扩散采样当作 SDE/ODE 数值积分问题，用数值分析工具（后向误差分析/修正方程、收敛阶、刚性、误差预算、指数积分器、Fokker–Planck 残差）系统理解与改进它。

**Architecture:** 先用 TDD 构建一个经 pytest 验证的核心库 `diffusion_na.py`（调度/边缘/score/反向动力学/矩递推/采样器/度量/刚性/FPE/score 网络），每个解析公式都有数值自检；然后 notebook 导入该库，逐节写理论(markdown+LaTeX)+实验(图+定量断言)+分析。核心实验全程用解析 score、免训练；仅网络迁移实验用 MPS 训练小网络。

**Tech Stack:** Python 3.12（uv 隔离环境）、numpy、scipy、matplotlib、torch(MPS)、scikit-learn、pytest。

**关键数学常量与时间约定（全程严格统一；已数值核验，见 review）：**
- VP: `dX=-½β(t)X dt+√β(t)dW`，`β(t)=0.1+19.9t`，`B(t)=∫₀ᵗβ=0.1t+9.95t²`。
- 转移核 `p(xₜ|x₀)=N(x₀e^{-B/2},(1-e^{-B})I)`。
- 数据 `X₀~N(0,s₀²I)`，**默认 `DATA_STD=s₀=0.5`（s₀=1 退化，禁用；该解析设定仅为隔离时间离散误差，真实难点是 score 非线性 + t→0 刚性）**。边缘 `v(t)=1+(s₀²-1)e^{-B(t)}`，精确 score `s*(x,t)=-x/v(t)`。
- 反向漂移系数 `a(t)=β(t)(1/v(t)-½) > 0`。
- **统一时间约定表：**

  | 名称 | 记号 | 方向 | 步长 | 一步公式 |
  |---|---|---|---|---|
  | 前向时间 | `t` | `0→1` | `dt>0` | VP 前向 SDE |
  | 反向采样网格 | `tₙ` | `1→ε` | `h=tₙ-t_{n+1}>0` | EM: `x_{n+1}=(1-a(tₙ)h)xₙ+√(β(tₙ)h)ζ` |
  | 反向时间 | `τ=1-t` | `0→1-ε` | `dτ>0` | `dY=-a(1-τ)Y dτ+√(β(1-τ))dW̃` |

- 反向 EM 方差递推（正步长 `h`）：`V_{n+1}=(1-a(tₙ)h)²Vₙ+β(tₙ)h`，逼近 `v(ε)`。
- **反向方差 ODE（含因子 2！）：`dV/dτ=-2a(1-τ)V+β(1-τ)`**。核验：`s₀=0.5,ε=1e-3` 时末端=`v(ε)=0.250082`（因子 2 正确）；漏因子 2 错得 `0.759`。
- 概率流 ODE：前向 `f_pf(t,x)=-½βx-½β·s*=½β(1/v-1)x`；反向 Euler 步 `x_{n+1}=xₙ-h·f_pf(tₙ,xₙ)`；**高斯闭式精确解 `x(t)=√(v(t)/v(1))·x(1)`**（核验通过）。
- 稳定域：`|1-a(t)h|≤1 ⇒ 0≤h≤2/a(t)`；`t≈1` 处 `a≈10 ⇒ h≲0.2`（故最粗稳定实验在 `N≈5` 量级）。
- 采样区间 `[ε,1]`，默认 `EPS=1e-3`；从先验 `N(0,I)` 起采（误差预算实验里单列"先验失配"项；纯离散误差对照时起点取 `V_start=marginal_var(1.0,s0)`）。
- 全局 `SEED=2026`。

---

## 文件结构

```
Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/
  diffusion_na.py        # 经测试的核心库（开发期；交付时内联进 notebook）
  反向扩散采样的数值分析.ipynb   # 主交付（中文，开发期 import diffusion_na；Task23 内联为单文件）
  tests/test_diffusion_na.py     # pytest 数值自检（开发脚手架）
  pyproject.toml         # uv 依赖
  README.md              # 运行/复现说明
  figures/               # 实验图（运行生成）
  .gitignore             # 忽略 .venv/ figures/ __pycache__/
```

- `diffusion_na.py` 单文件库，按职责分区（调度/边缘/score | 反向动力学/递推 | 采样器 | 度量 | 刚性 | FPE | 网络）。
- notebook 顶部"环境与导入"cell 导入库；各节 cell 调用库函数，专注理论叙述、画图、分析。
- 交付时同目录附 `diffusion_na.py`；最后提供"可选：生成完全内联的单文件 notebook"步骤。

---

## Task 0: 项目脚手架与 uv 隔离环境

**Files:**
- Create: `Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/pyproject.toml`
- Create: `Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/.gitignore`
- Create: `Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/diffusion_na.py`（空 stub）
- Create: `Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/tests/test_diffusion_na.py`（空 stub）

- [ ] **Step 1: 建文件夹与 uv 工程**

注意中文父路径带空格，命令加引号。
```bash
cd "/Users/hariseldon/Desktop/课程作业/AI-math-advanced/Tutorial-AI4SC-SC4AI"
mkdir -p "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/tests" "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/figures"
cd "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis"
uv init --python 3.12 --no-workspace .
```

- [ ] **Step 2: 写 pyproject 依赖并安装**

把 `pyproject.toml` 的 `dependencies` 设为：
```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "torch>=2.2",
    "scikit-learn>=1.4",
    "jupyter>=1.0",
    "ipykernel>=6.29",
    "pytest>=8.0",
]
```
Run:
```bash
uv add numpy scipy matplotlib torch scikit-learn jupyter ipykernel pytest
```

- [ ] **Step 3: .gitignore**
```
.venv/
__pycache__/
*.pyc
figures/
.ipynb_checkpoints/
```

- [ ] **Step 4: 验证 torch + MPS 可用**

Run:
```bash
uv run python -c "import torch, numpy, scipy, sklearn, matplotlib; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
```
Expected: 打印 torch 版本且 `mps True`（Apple Silicon）。

- [ ] **Step 5: 注册 Jupyter kernel（供 notebook 使用）**

Run:
```bash
uv run python -m ipykernel install --user --name diffusion-na --display-name "Python (diffusion-na)"
```
Expected: `Installed kernelspec diffusion-na ...`

- [ ] **Step 6: Commit**
```bash
cd "/Users/hariseldon/Desktop/课程作业/AI-math-advanced/Tutorial-AI4SC-SC4AI"
git add "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/pyproject.toml" "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/.gitignore"
git commit -m "chore: scaffold SC4AI diffusion numerical-analysis project (uv env)"
```

---

## Task 1: 调度、边缘分布与精确 score

**Files:**
- Modify: `diffusion_na.py`
- Test: `tests/test_diffusion_na.py`

- [ ] **Step 1: 写失败测试**
```python
import numpy as np
import pytest
import diffusion_na as dna

def test_int_beta_endpoints():
    assert dna.int_beta(0.0) == pytest.approx(0.0)
    assert dna.int_beta(1.0) == pytest.approx(0.1 + 9.95)   # 10.05

def test_marginal_var_range():
    s0 = 0.5
    assert dna.marginal_var(1e-6, s0) == pytest.approx(s0**2, abs=1e-3)  # v(0)->s0^2
    assert dna.marginal_var(1.0, s0) == pytest.approx(1.0, abs=1e-3)     # v(1)->1

def test_transition_kernel_moments_statistical():
    # 直接模拟前向核，统计均值/方差匹配解析 alpha, sigma2
    rng = np.random.default_rng(0)
    t, x0 = 0.4, 1.3
    samples = dna.alpha(t)*x0 + np.sqrt(dna.sigma2(t))*rng.standard_normal(200000)
    assert samples.mean() == pytest.approx(dna.alpha(t)*x0, abs=5e-3)
    assert samples.var() == pytest.approx(dna.sigma2(t), abs=5e-3)

def test_exact_score_gaussian_matches_finite_difference():
    s0 = 0.5
    t = 0.3
    x = np.array([0.7, -1.1])
    def logp(x): return -0.5*np.sum(x**2)/dna.marginal_var(t,s0) - np.log(2*np.pi*dna.marginal_var(t,s0))/... # 见实现
    # 用数值梯度校验 exact_score_gaussian
    eps = 1e-5
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i]+=eps
        xm = x.copy(); xm[i]-=eps
        grad[i] = (dna._log_marginal_gaussian(xp,t,s0)-dna._log_marginal_gaussian(xm,t,s0))/(2*eps)
    assert np.allclose(grad, dna.exact_score_gaussian(x,t,s0), atol=1e-4)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_diffusion_na.py -q`
Expected: FAIL（`AttributeError: module 'diffusion_na' has no attribute ...`）。

- [ ] **Step 3: 实现**
```python
import numpy as np

BETA_MIN = 0.1
BETA_MAX = 20.0
SEED = 2026

def beta(t):
    return BETA_MIN + (BETA_MAX - BETA_MIN) * np.asarray(t, dtype=float)

def int_beta(t):  # B(t) = ∫_0^t beta
    t = np.asarray(t, dtype=float)
    return BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t**2

def alpha(t):  # 均值尺度 e^{-B/2}
    return np.exp(-0.5 * int_beta(t))

def sigma2(t):  # 转移核方差 1 - e^{-B}（数值稳定）
    return -np.expm1(-int_beta(t))

def marginal_var(t, s0=0.5):  # v(t)，X0~N(0,s0^2 I)
    return 1.0 + (s0**2 - 1.0) * np.exp(-int_beta(t))

def _log_marginal_gaussian(x, t, s0=0.5):  # 各向同性高斯边缘的 log 密度（含常数）
    x = np.asarray(x, dtype=float); v = marginal_var(t, s0); d = x.size
    return -0.5*np.sum(x**2)/v - 0.5*d*np.log(2*np.pi*v)

def exact_score_gaussian(x, t, s0=0.5):  # ∇log p = -x/v
    return -np.asarray(x, dtype=float) / marginal_var(t, s0)
```
（修正 Step1 测试里 `logp` 写法，统一用 `_log_marginal_gaussian`。）

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_diffusion_na.py -q`
Expected: PASS（4 passed）。

- [ ] **Step 5: Commit**
```bash
cd "/Users/hariseldon/Desktop/课程作业/AI-math-advanced/Tutorial-AI4SC-SC4AI"
git add "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/diffusion_na.py" "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/tests/test_diffusion_na.py"
git commit -m "feat: VP schedule, marginals and exact Gaussian score with numerical checks"
```

---

## Task 2: 高斯混合精确 score（用于误差三分解与 FPE 诊断）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_gmm_score_matches_finite_difference():
    # 两分量各向同性高斯混合 X0 ~ Σ w_k N(m_k, s0^2 I)
    means = np.array([[-1.5, 0.0],[1.5, 0.0]]); weights = np.array([0.5,0.5]); s0=0.4
    t = 0.25; x = np.array([0.3,-0.6])
    eps=1e-5; grad=np.zeros_like(x)
    for i in range(len(x)):
        xp=x.copy();xp[i]+=eps; xm=x.copy();xm[i]-=eps
        grad[i]=(dna.gmm_log_marginal(xp,t,means,weights,s0)-dna.gmm_log_marginal(xm,t,means,weights,s0))/(2*eps)
    assert np.allclose(grad, dna.exact_score_gmm(x,t,means,weights,s0), atol=1e-4)
```

- [ ] **Step 2: 运行确认失败** — Run `uv run pytest -q -k gmm`，Expected FAIL。

- [ ] **Step 3: 实现**

边缘：每个分量经 VP 演化为 `N(alpha(t) m_k, (alpha(t)^2 s0^2 + sigma2(t)) I)`。
```python
def _component_var(t, s0):
    return alpha(t)**2 * s0**2 + sigma2(t)

def gmm_log_marginal(x, t, means, weights, s0=0.4):
    x=np.asarray(x,float); means=np.asarray(means,float); w=np.asarray(weights,float)
    vt=_component_var(t,s0); d=x.shape[-1]
    mu=alpha(t)*means                                  # [K,d]
    sq=np.sum((x-mu)**2, axis=-1)                      # [K]
    logc=-0.5*d*np.log(2*np.pi*vt)
    logp_k=logc-0.5*sq/vt
    from scipy.special import logsumexp
    return logsumexp(logp_k, b=w)

def exact_score_gmm(x, t, means, weights, s0=0.4):   # 批量版 [N,d]->[N,d]（review#5）
    x=np.atleast_2d(np.asarray(x,float))               # [N,d]
    means=np.asarray(means,float); w=np.asarray(weights,float)  # [K,d],[K]
    vt=_component_var(t,s0); mu=alpha(t)*means          # [K,d]
    diff=mu[None,:,:]-x[:,None,:]                       # [N,K,d]
    sq=np.sum(diff**2,axis=-1)                          # [N,K]
    logp=np.log(w)[None,:]-0.5*sq/vt                    # [N,K]
    logp=logp-logp.max(axis=1,keepdims=True)
    r=np.exp(logp); r=r/r.sum(axis=1,keepdims=True)     # 后验权重 [N,K]
    return np.sum(r[:,:,None]*diff/vt, axis=1)          # [N,d]
```
（`gmm_log_marginal` 同样支持 `[N,d]` 批量：`sq` 用 `np.sum((x[:,None,:]-mu[None,:,:])**2,axis=-1)` 得 `[N,K]`，再 `logsumexp(.,axis=1,b=w)`。）

- [ ] **Step 4: 运行确认通过** — Run `uv run pytest -q -k gmm`，Expected PASS。
- [ ] **Step 5: Commit** — `git commit -m "feat: closed-form Gaussian-mixture marginal score with FD check"`

---

## Task 3: 反向动力学与精确方差（解析参照）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_reverse_variance_exact_matches_marginal():
    # 反向方差 ODE（含因子2）从 t=1 精解到 t=eps，应≈ marginal_var(eps)=0.250082
    s0=0.5; eps=1e-3
    Vend = dna.reverse_variance_exact(eps, s0=s0)         # 默认 V_start=marginal_var(1,s0)
    assert Vend == pytest.approx(dna.marginal_var(eps,s0), rel=1e-3)
def test_reverse_variance_factor_two_not_one():
    # 漏因子2会得到≈0.759（错），明确锁死因子2
    s0=0.5; eps=1e-3
    assert dna.reverse_variance_exact(eps, s0=s0) < 0.4   # 正确≈0.25，错误≈0.76
```

- [ ] **Step 2: 运行确认失败**

- [ ] **Step 3: 实现**

反向时间 τ=1-t，**方差 ODE 含因子 2**：`dV/dτ=-2a(1-τ)V+β(1-τ)`（由 Itô：线性 SDE `dY=κYdτ+gdW` 有 `V'=2κV+g²`，此处 `κ=-a`）。用 `scipy.integrate.solve_ivp`（rtol=atol=1e-10）从 τ=0 解到 τ=1-eps。
```python
from scipy.integrate import solve_ivp

def reverse_drift_coeff(t, s0=0.5):  # a(t) > 0
    return beta(t)*(1.0/marginal_var(t,s0) - 0.5)

def reverse_variance_exact(t_end, s0=0.5, t_start=1.0, V_start=None, dense=False):
    if V_start is None:
        V_start = marginal_var(t_start, s0)          # 纯离散误差对照用真值起点
    def rhs(tau, V):
        t = t_start - tau
        a = reverse_drift_coeff(t, s0)
        return [-2.0*a*V[0] + beta(t)]               # ★ 含因子 2
    tau_end = t_start - t_end
    sol = solve_ivp(rhs, [0.0, tau_end], [V_start], rtol=1e-10, atol=1e-10, dense_output=True)
    return sol if dense else float(sol.y[0,-1])
```
（注：`V_start=marginal_var(t_start)` 为纯离散误差对照；从先验 `V_start=1` 起则含"先验失配"，在实验⑧单列。`dense=True` 供 Task 5 取二阶导。）

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: Commit** — `git commit -m "feat: exact reverse-time variance ODE reference"`

---

## Task 4: 反向 EM 方差确定性递推 + 随机采样器（互校）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_em_recursion_matches_monte_carlo():
    s0=0.5; eps=1e-3; N=200
    Vrec = dna.em_variance_recursion(N, eps=eps, s0=s0, V_start=dna.marginal_var(1.0,s0))
    # 蒙特卡洛：跑很多条 1D 反向 EM 轨迹，末端方差应≈递推
    Xend = dna.em_sample(N, n_samples=200000, eps=eps, s0=s0, dim=1, seed=0,
                         x_start=None)  # x_start=None -> 从 N(0,1) 起采
    assert Xend.var() == pytest.approx(Vrec, rel=0.03)
```

- [ ] **Step 2: 运行确认失败**

- [ ] **Step 3: 实现**
```python
def em_variance_recursion(N, eps=1e-3, s0=0.5, t_start=1.0, V_start=1.0):
    ts = np.linspace(t_start, eps, N+1)   # 递减
    V = float(V_start)
    for i in range(N):
        t = ts[i]; dt = ts[i]-ts[i+1]
        a = reverse_drift_coeff(t, s0)
        V = (1.0 - a*dt)**2 * V + beta(t)*dt
    return V

def em_sample(N, n_samples=10000, eps=1e-3, s0=0.5, dim=1, seed=0, x_start=None):
    rng = np.random.default_rng(seed)
    ts = np.linspace(1.0, eps, N+1)
    x = rng.standard_normal((n_samples, dim)) if x_start is None else np.array(x_start, float)
    for i in range(N):
        t=ts[i]; dt=ts[i]-ts[i+1]; a=reverse_drift_coeff(t,s0)
        x = (1.0 - a*dt)*x + np.sqrt(beta(t)*dt)*rng.standard_normal(x.shape)
    return x
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: Commit** — `git commit -m "feat: deterministic EM variance recursion + stochastic sampler cross-check"`

---

## Task 5: O(Δt) 偏差律与修正方程系数（后向误差分析核心）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_bias_law_slope_is_one():
    s0=0.5; eps=1e-3
    Ns=np.array([50,100,200,400,800,1600])
    target=dna.reverse_variance_exact(eps,s0=s0,V_start=dna.marginal_var(1.0,s0))
    errs=np.array([abs(dna.em_variance_recursion(N,eps=eps,s0=s0,V_start=dna.marginal_var(1.0,s0))-target) for N in Ns])
    dts=(1.0-eps)/Ns
    slope=np.polyfit(np.log(dts),np.log(errs),1)[0]
    assert 0.85 < slope < 1.15           # 弱阶≈1

def test_modified_eq_coefficient_consistency():
    s0=0.5; eps=1e-3
    c_theory=dna.bias_coeff_theory(eps=eps,s0=s0)
    c_emp=dna.bias_coeff_empirical(eps=eps,s0=s0)   # Richardson 外推得 err/h 极限
    assert abs(c_theory-c_emp)/abs(c_emp) < 0.1     # 理论闭式 vs 经验交叉验证
```

- [ ] **Step 2: 运行确认失败**

- [ ] **Step 3: 实现（修正方程含离散耦合项 G 与因子 2，见 review#3）**

EM 方差递推写成 `V_{n+1}=Vₙ+hF(τ,V)+h²G(τ,V)`，其中
`F=-2a(1-τ)V+β(1-τ)`（注意因子 2），`G=a(1-τ)²V`（来自 `(1-ah)²` 展开的 `a²h²V`）。
修正方程 `V'=F+hH+O(h²)`，一步 Taylor 匹配得 **`H=G-½(F_τ+F_V·F)=a²V-½V''`**（沿精确解 `F_τ+F_V F=dF/dτ=V''`）。
领头阶全局偏差 `V_N-v(ε)=c·h+O(h²)`，**`c=∫₀^T Φ(T,s)[a(1-s)²V(s)-½V''(s)]ds`**，
传播子 **`Φ(T,s)=exp(∫_s^T F_V dr)=exp(∫_s^T -2a(1-r)dr)`**。
```python
def bias_coeff_empirical(eps=1e-3, s0=0.5, Ns=(800,1600,3200,6400)):
    target=reverse_variance_exact(eps,s0=s0,V_start=marginal_var(1.0,s0))
    Vs=np.array([em_variance_recursion(N,eps=eps,s0=s0,V_start=marginal_var(1.0,s0)) for N in Ns])
    hs=(1.0-eps)/np.array(Ns); ratios=(Vs-target)/hs        # err/h
    A=np.vstack([np.ones_like(hs),hs]).T                    # Richardson: 线性外推到 h=0
    return float(np.linalg.lstsq(A,ratios,rcond=None)[0][0])

def bias_coeff_theory(eps=1e-3, s0=0.5, M=40000):
    T=1.0-eps; taus=np.linspace(0,T,M); t=1.0-taus
    a=reverse_drift_coeff(t,s0)
    sol=reverse_variance_exact(eps,s0=s0,V_start=marginal_var(1.0,s0),dense=True)  # 高精度 V(τ)
    hh=1e-4
    V  =sol.sol(taus)[0]
    Vpp=(sol.sol(np.clip(taus+hh,0,T))[0]-2*V+sol.sol(np.clip(taus-hh,0,T))[0])/hh**2  # V''(τ)
    H=a**2*V-0.5*Vpp                                        # 修正项（含 G=a²V）
    # 传播子 Φ(T,s)=exp(∫_s^T -2a dτ')
    cum=np.concatenate([[0],np.cumsum(0.5*(a[1:]+a[:-1])*np.diff(taus))])  # ∫_0^s a
    Phi=np.exp(-2.0*(cum[-1]-cum))                          # 含因子 2
    return float(np.trapz(Phi*H,taus))
```
（符号关键处：F 含因子 2、必须保留 G=a²V、传播子用 `-2a`。验收口径：`test_bias_law_slope_is_one` 与 `test_modified_eq_coefficient_consistency` 必须同时通过；若 <10% 不达标，notebook 诚实标注并以 Richardson 经验系数为主、闭式为辅。）

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: Commit** — `git commit -m "feat: O(dt) bias law + modified-equation coefficient (theory vs Richardson)"`

---

## Task 6: 改进采样器（半隐式/指数处理线性漂移 + 概率流 ODE + 指数积分器）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_exponential_drift_is_exact_on_linear_part():
    # 冻结系数指数法降低误差常数（不声称机器精度）：N=64 时比 EM 更准
    s0=0.5; eps=1e-3; N=64
    target=dna.reverse_variance_exact(eps,s0=s0)        # 默认真值起点
    Vexp=dna.semiimplicit_variance_recursion(N,eps=eps,s0=s0,V_start=dna.marginal_var(1.0,s0))
    Vem =dna.em_variance_recursion(N,eps=eps,s0=s0,V_start=dna.marginal_var(1.0,s0))
    assert abs(Vexp-target) < abs(Vem-target)           # 指数法误差常数更小
def test_pf_ode_exact_closed_form():
    # 高斯 PF ODE 闭式 x(eps)=sqrt(v(eps)/v(1)) x(1) 必须等于解析公式（机器精度参照）
    s0=0.5; eps=1e-3
    assert dna.pf_ode_exact(2.0, eps=eps, s0=s0) == pytest.approx(
        np.sqrt(dna.marginal_var(eps,s0)/dna.marginal_var(1.0,s0))*2.0, rel=1e-12)
def test_pf_ode_heun_higher_order_than_euler():
    # 概率流 ODE 末态(1D) Heun 比 Euler 更接近闭式精确解（阶提升仅 ODE 成立）
    s0=0.5; eps=1e-3; ref=dna.pf_ode_exact(2.0,eps=eps,s0=s0)
    e_euler=abs(dna.pf_ode_sample(64,'euler',x_start=2.0,eps=eps,s0=s0)-ref)
    e_heun =abs(dna.pf_ode_sample(64,'heun', x_start=2.0,eps=eps,s0=s0)-ref)
    assert e_heun < e_euler
```

- [ ] **Step 2: 运行确认失败**

- [ ] **Step 3: 实现**

- `semiimplicit_variance_recursion`：每步对线性漂移用指数因子 `e^{-a h}`、噪声项用常系数 OU 精确方差 `β(1-e^{-2ah})/(2a)`。**它只在每步常系数意义下精确；因 a,β 时变，整体仍为一阶，只是误差常数小于 EM**（不声称机器精度）。
- `pf_ode_exact`：高斯闭式 `x(t)=√(v(t)/v(1))·x(1)`（机器精度参照 = "理想指数积分器"）。
- `pf_ode_sample`：前向漂移 `f_pf(t,x)=½β(1/v-1)x`；反向 Euler 步 `x_{n+1}=xₙ-h·f_pf(tₙ,xₙ)`，Heun 用两段平均。
```python
def semiimplicit_variance_recursion(N, eps=1e-3, s0=0.5, t_start=1.0, V_start=1.0):
    ts=np.linspace(t_start,eps,N+1); V=float(V_start)
    for i in range(N):
        t=ts[i]; h=ts[i]-ts[i+1]; a=reverse_drift_coeff(t,s0)
        decay=np.exp(-a*h)                                   # 指数化线性漂移
        noise=beta(t)*(1-np.exp(-2*a*h))/(2*a) if abs(a)>1e-8 else beta(t)*h
        V=decay**2*V+noise
    return V

def pf_ode_exact(x_start, eps=1e-3, s0=0.5, t_start=1.0):     # 闭式精确解
    return float(np.sqrt(marginal_var(eps,s0)/marginal_var(t_start,s0))*x_start)

def pf_ode_drift(t, x, s0):                                  # 前向 f_pf=½β(1/v-1)x
    return 0.5*beta(t)*(1.0/marginal_var(t,s0)-1.0)*np.asarray(x,float)

def pf_ode_sample(N, method, x_start, eps=1e-3, s0=0.5):
    ts=np.linspace(1.0,eps,N+1); x=np.asarray(x_start,float)
    for i in range(N):
        t=ts[i]; h=ts[i]-ts[i+1]                              # 正步长 h>0
        f1=pf_ode_drift(t,x,s0)
        if method=='euler':
            x=x - h*f1                                        # 反向 Euler: x - h f_pf
        elif method=='heun':
            xp=x - h*f1; f2=pf_ode_drift(ts[i+1],xp,s0); x=x-0.5*h*(f1+f2)
    return float(x) if np.ndim(x)==0 else x
```
（`exp_integrator` 采样器版即 `semiimplicit` 的轨迹形式，列入实验⑥；所有反向步进符号已与统一约定表一致。）

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: Commit** — `git commit -m "feat: semi-implicit/exponential variance recursion + probability-flow ODE samplers"`

---

## Task 7: 度量（Bures-W2、矩误差、能量距离/sliced-W）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_bures_w2_self_zero_and_1d():
    assert dna.bures_w2(np.zeros(2),np.eye(2),np.zeros(2),np.eye(2))==pytest.approx(0,abs=1e-9)
    # 1D: W2^2 = (m1-m2)^2+(s1-s2)^2
    assert dna.bures_w2(np.array([0.]),np.array([[4.]]),np.array([1.]),np.array([[1.]]))**2 \
        == pytest.approx((0-1)**2+(2-1)**2)
def test_sliced_w_and_energy_distance_separate():
    rng=np.random.default_rng(0)
    a=rng.standard_normal((2000,2)); b=rng.standard_normal((2000,2))
    assert dna.sliced_wasserstein(a,b) < dna.sliced_wasserstein(a, b+3.0)
    assert dna.energy_distance(a,b) < dna.energy_distance(a, b+3.0)   # 子采样版
```

- [ ] **Step 2: 运行确认失败**

- [ ] **Step 3: 实现**

`bures_w2`：对协方差对称化 + 特征值非负截断（review#6 稳定性）。2-D 点云**主用 `sliced_wasserstein`**；`energy_distance` 仅辅助且**子采样 n≤512**（review#4，避免 O(n²) 内存）。
```python
from scipy.linalg import sqrtm
def _psd_sqrt(C):
    C=0.5*(C+C.T); w,Q=np.linalg.eigh(C); w=np.clip(w,0,None)
    return (Q*np.sqrt(w))@Q.T
def bures_w2(m1,C1,m2,C2):
    m1,m2=np.atleast_1d(m1),np.atleast_1d(m2); C1,C2=np.atleast_2d(C1),np.atleast_2d(C2)
    s1=_psd_sqrt(C1); cross=_psd_sqrt(s1@C2@s1)
    bures=np.trace(C1+C2-2*cross)
    return float(np.sqrt(max(np.sum((m1-m2)**2)+float(bures),0.0)))
def moment_error(samples, m_true, v_true):                  # 前两阶矩误差
    return float(abs(samples.mean(0)-m_true).mean()+abs(samples.var(0)-v_true).mean())
def sliced_wasserstein(X,Y,n_proj=200,seed=0):              # 2-D 主指标，O(n log n)
    rng=np.random.default_rng(seed); d=X.shape[1]
    th=rng.standard_normal((d,n_proj)); th/=np.linalg.norm(th,axis=0,keepdims=True)
    px=np.sort(X@th,axis=0); py=np.sort(Y@th,axis=0)         # 等样本数下排序匹配
    return float(np.mean(np.abs(px-py)))
def energy_distance(X,Y,max_n=512,seed=0):                 # 辅助；子采样避免 O(n^2)
    rng=np.random.default_rng(seed)
    X=X[rng.choice(len(X),min(len(X),max_n),replace=False)]
    Y=Y[rng.choice(len(Y),min(len(Y),max_n),replace=False)]
    def pd(A,B): return np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1)+1e-12).mean()
    return float(2*pd(X,Y)-pd(X,X)-pd(Y,Y))
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: Commit** — `git commit -m "feat: Bures-W2, moment error, energy distance metrics"`

---

## Task 8: 刚性与显式 Euler 稳定域

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_stiffness_peaks_near_t1():
    ts=np.linspace(1e-3,1.0,200); s0=0.5
    L=np.abs(dna.reverse_drift_coeff(ts,s0))
    assert ts[np.argmax(L)] > 0.7         # 最刚处靠近 t=1
def test_critical_step_predicts_blowup():
    s0=0.5
    # 在最刚 t 处，dt 略超 2/|a| 时单步放大>1 -> 不稳
    t=1.0; a=dna.reverse_drift_coeff(t,s0); dt_crit=2.0/abs(a)
    assert abs(1-a*(dt_crit*1.1))>1       # 越界放大
    assert abs(1-a*(dt_crit*0.5))<1       # 域内收缩
```

- [ ] **Step 2–4:** 实现 `stiffness(t,s0)=abs(reverse_drift_coeff)`、`critical_dt(t,s0)=2/abs(a)`、`nan_rate(N,...)`（跑 em_sample 统计 NaN/溢出比例）；运行确认通过。
- [ ] **Step 5: Commit** — `git commit -m "feat: stiffness L(t) and explicit-Euler stability bound"`

---

## Task 9: Fokker–Planck（score-FPE）残差算子（torch autograd）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_fpe_residual_zero_on_exact_score():
    s0=0.5; t=0.3
    x=np.array([[0.5,-0.4],[1.0,0.2]])
    res=dna.fpe_residual_gaussian(x,t,s0)        # 解析 score 上残差≈0
    assert np.abs(res).max() < 1e-3
def test_fpe_residual_detects_perturbation():
    s0=0.5; t=0.3; x=np.array([[0.5,-0.4]])
    assert np.abs(dna.fpe_residual_gaussian(x,t,s0,perturb=0.3)).max() \
         > np.abs(dna.fpe_residual_gaussian(x,t,s0,perturb=0.0)).max()
```

- [ ] **Step 2: 运行确认失败**

- [ ] **Step 3: 实现**

**完整 log-FPE 恒等式**（review#6，已推导核验）：对 VP 前向过程，`s=∇log p` 满足
`∂ₜ log p - ½β[ d + x·s + ‖s‖² + ∇·s ] = 0`。
对解析高斯：`∂ₜ log p = ‖x‖²v'/(2v²) - (d/2)v'/v`（`v'=dv/dt`），其余项由 `s=-x/v` 闭式给出，残差恒等于 0；**硬断言只对解析/扰动 score**。网络 score 无解析 `∂ₜ log p` ⇒ **只作定性热力图，不做"残差≈误差"硬断言**（见 Task 20）。用 float64 CPU、逐分量 autograd 求 `∇·s`（2-D 极小规模）。
```python
import torch
def fpe_residual_gaussian(x, t, s0=0.5, perturb=0.0):
    x=np.atleast_2d(np.asarray(x,float))                 # [N,d]
    xt=torch.tensor(x,dtype=torch.float64,requires_grad=True)
    tt=torch.tensor(float(t),dtype=torch.float64,requires_grad=True)
    v=1.0+(s0**2-1.0)*torch.exp(-(0.1*tt+9.95*tt**2))    # v(t)
    s=-xt/v + perturb                                    # 解析 score(+扰动)
    d=x.shape[1]
    # ∂_t log p（对解析高斯）：用 autograd 对 t 求 logp 的导
    logp=(-0.5*(xt**2).sum(1)/v - 0.5*d*torch.log(2*np.pi*v))      # [N]
    dlogp_dt=torch.autograd.grad(logp.sum(), tt, create_graph=True)[0]
    # ∇·s：逐分量
    div_s=sum(torch.autograd.grad(s[:,i].sum(), xt, create_graph=True)[0][:,i] for i in range(d))
    xs=(xt*s).sum(1); s2=(s**2).sum(1)
    res = dlogp_dt - 0.5*beta(float(t))*(d + xs + s2 + div_s)      # log-FPE 残差
    return res.detach().numpy()
```
（先在解析 score 上调到残差<1e-3 再用于诊断；`perturb≠0` 时残差应随之增大。）

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: Commit** — `git commit -m "feat: score-FPE residual operator with exact-score sanity check"`

---

## Task 10: 小 score 网络与 DSM 训练（网络迁移实验用）

**Files:** Modify `diffusion_na.py`; Test `tests/test_diffusion_na.py`

- [ ] **Step 1: 失败测试**
```python
def test_trained_score_approximates_exact_on_gaussian():
    # 在 1D 高斯数据上短训练，学到的 score 在 t-切片上≈ -x/v
    net=dna.train_score_net_gaussian(s0=0.5, n_iters=3000, device='cpu', seed=0)
    t=0.3; xs=np.linspace(-1,1,11)
    pred=dna.eval_score_net(net, xs.reshape(-1,1), t)
    exact=dna.exact_score_gaussian(xs.reshape(-1,1),t,0.5)
    assert np.mean((pred-exact)**2) < 0.2        # 量级一致
```

- [ ] **Step 2: 运行确认失败**

- [ ] **Step 3: 实现**

复刻 Lect6 的 MLP score 网络（输入 (x,t)，LogSigmoid，3×64）+ DSM 损失（解析转移核目标 `(mu-x_t)/var`，权重 `sigma2`）；`train_score_net_gaussian`/`train_score_net_data` + `eval_score_net`；device 支持 'mps'/'cpu'。训练≤几千步。
- [ ] **Step 4: 运行确认通过**（CPU 上跑测试；notebook 里可切 MPS）
- [ ] **Step 5: Commit** — `git commit -m "feat: small DSM score network (MPS-capable) + exact-score recovery test"`

---

## Task 11: 全测试绿 + 库收尾

- [ ] **Step 1:** Run `uv run pytest -q`，Expected: all PASS。修复任何残留。
- [ ] **Step 2:** Run `uv run python -c "import diffusion_na"` 确认无副作用导入。
- [ ] **Step 3: Commit** — `git commit -m "test: full numerical self-check suite green"`

---

## Notebook 组装（Task 12–22）

每个 notebook 实验 cell 都：导入 `diffusion_na`、设 `np.random.seed(SEED)`、调用已测函数、画图存 `figures/`、并在 cell 末尾 `assert` 关键定量结论（自校验），随后一段中文分析 markdown。所有 cell 用注册的 `diffusion-na` kernel；最终"运行全部"必须无错。

### Task 12: 笔记本骨架 + Part 0 引言
- Create `反向扩散采样的数值分析.ipynb`。首 cell：标题/作者(牛昱琛 253108120111)/日期/单人说明/目录。
- 环境 cell：导入、`device` 选择、`SEED`、matplotlib 中文字体设置（`plt.rcParams['axes.unicode_minus']=False`，尝试 'Heiti TC'/'Arial Unicode MS' 等 mac 自带中文字体并兜底）。
- Part 0 markdown：背景、扩散即 SDE、采样即数值积分、SC4AI 主题、与 Lect5/Lect6 呼应、**贡献清单**、阅读导航。
- [ ] 运行至此 cell 无错。 Commit `docs: notebook skeleton + intro`。

### Task 13: Part I 理论（§1 SDE/score/DSM 回顾，§2 可解析设定）
- §1 markdown：VP 前向/反向(Anderson) SDE 推导、score、DSM 目标（含转移核闭式）。
- §2 markdown + cell：高斯线性 score 推导；画 `v(t)`、`a(t)`、`β(t)`、`alpha/sigma2` 随 t 曲线（调用库）；高斯混合 score 推导与一张 score 向量场/热力图。assert：`marginal_var` 端点值符合预期。
- [ ] 运行无错。 Commit `docs: Part I theory + analytic setup figures`。

### Task 14: §3–§4 矩递推与后向误差分析（理论 markdown）
- §3 markdown：推导 EM 一步映射 `x_{n+1}=(1-a(tₙ)h)xₙ+√(β(tₙ)h)ζ` 与确定性方差递推 `V_{n+1}=(1-a(tₙ)h)²Vₙ+β(tₙ)h`；由 Itô 给出连续极限 `dV/dτ=-2aV+β`（强调因子 2）。
- §4 markdown：后向误差分析/修正方程推导，写出 `F=-2aV+β`、离散耦合项 `G=a²V`、修正项 `H=a²V-½V''`、闭式系数 `c=∫Φ[a²V-½V'']`，传播子 `Φ=exp(∫-2a)`；说明漂移项与噪声注入项各自的 O(h) 贡献（"只指数化漂移不足以消偏"）。
- [ ] Commit `docs: moment recursion + backward-error analysis derivation`。

### Task 15: 实验① O(Δt) 偏差律
- cell：`Ns=[50,100,200,400,800,1600]`，画 `|V_N-v(ε)|` vs `Δt` log-log；标斜率。assert `0.85<slope<1.15`。中文分析。
- [ ] 运行无错、断言通过。 Commit `exp: O(dt) bias law`。

### Task 16: 实验② 修正方程系数对照 + 实验③ 弱阶
- 实验②：`c_theory` vs `c_emp`（Richardson）对照表/图，assert 相对误差<10%（若失败则诚实切换为"以经验系数为准"叙述，见 Task5 注）。
- 实验③：蒙特卡洛(多种子)估 `E‖Y₀‖²` 弱误差 vs Δt，叠加确定性递推预测；assert 斜率≈1。
- [ ] 运行无错。 Commit `exp: modified-eq coefficient + weak-order`。

### Task 17: Part III 实验④ 刚性与稳定步长
- §8 markdown（刚性/稳定域推导）+ cell：画 `L(t)`；画"临界步长理论 vs 实测 NaN 率"；均匀 t / 均匀 log-SNR / 自适应网格在固定 NFE 下的矩误差对比；时间裁剪 `t_end` 扫描。assert：最刚处靠近 t=1；理论临界步长与实测发散阈值同量级。
- [ ] 运行无错。 Commit `exp: stiffness & stability step-size criterion`。

### Task 18: Part IV 实验⑤ 修正采样器 + 实验⑥ 指数积分器/少步采样
- §10 markdown + 实验⑤：半隐式/指数 vs EM 的 Bures-W2 vs h（含 Richardson 外推线），assert 指数法误差常数<EM。
- §12 markdown + 实验⑥：概率流 ODE Euler/Heun/指数 收敛阶 log-log，**以闭式 PF 映射 `x∝√v` 为机器精度参照**；Swiss-Roll 上 NFE-质量 Pareto（**主指标 sliced-Wasserstein**，energy distance 辅助、子采样）；**诚实**呈现"朴素 frozen-score 指数法在含 score 项时不一定胜 Euler"，并明确"Heun 阶提升仅 ODE 成立"。assert：Heun 末态误差<Euler（对闭式参照）；冻结系数指数法误差常数<EM。
- [ ] 运行无错。 Commit `exp: corrected sampler + exponential integrator (honest)`。

### Task 19: 实验⑦ 偏差-方差（SDE vs ODE）—— 探索性，不做硬断言
- §13 markdown + cell：无扰动 SDE(EM) vs ODE(Heun) 的 W2/矩 vs NFE；受控分数扰动 `s+ε·noise` 的误差曲线；偏差-方差分量图；并排相图。**定位为探索性研究**：报告"在哪些扰动方向/步长/强度下出现 SDE 自校正、哪些没有"，**不**对"SDE 必更鲁棒"做硬 assert（review#7）。唯一硬校验：所有曲线对解析矩闭式真值口径一致、多种子平均、无 NaN。
- [ ] 运行无错。 Commit `exp: bias-variance SDE vs ODE (exploratory)`。

### Task 20: Part V 实验⑧ 误差三分解 + 实验⑨ FPE 诊断
- §14 markdown + 实验⑧：高斯混合 toy，A=精确score+极小步 / B=精确score+大步 / C=网络score+大步，做 矩误差（主）/sliced-W（辅）误差瀑布图(分数/离散/先验三段)，先验项用闭式 KL。assert：三段和≈总误差。
- §15 markdown + 实验⑨：**硬断言只在解析/扰动 score**——解析 score 残差<1e-3（单元校验）、`perturb` 增大时残差单调增大；**网络 score 仅定性**——score 误差 (x,t) 热力图 + 残差热力图并排观察，文字讨论其定性关联，**不**做"残差≈误差"硬 assert（review#6，网络无解析 ∂ₜlogp）。
- [ ] 运行无错。 Commit `exp: error-budget decomposition + FPE residual diagnostic`。

### Task 21: 实验⑩ 网络 score 迁移
- §16 markdown + cell：用 `train_score_net_data`（Swiss-Roll，MPS，1–3min）或高斯数据训练；复现弱误差 vs Δt 曲线，叠加解析离散误差曲线，标出**离散误差 vs 网络逼近误差交叉点**。assert：曲线先随 Δt 减小下降、后被网络误差地板截断（存在交叉/拐点）。
- [ ] 运行无错。 Commit `exp: transfer to trained network score (MPS)`。

### Task 22: Part VI 总结 + 附录 + 全文跑通
- §17 markdown：与 DPM-Solver/指数积分器联系（诚实）、与 Lect5 修正方程统一视角、局限与展望。
- 附录 markdown：完整推导补全、超参/种子表、环境清单。
- [ ] **Step:** `uv run jupyter nbconvert --to notebook --execute --inplace "反向扩散采样的数值分析.ipynb"`（或 papermill）端到端执行，Expected: 全部 cell 无错、所有 assert 通过、figures/ 生成。
- [ ] 修复任何执行错误后重跑。 Commit `docs: conclusion + appendix; full notebook executes clean`。

---

## Task 23: README、复现说明与最终审阅

**Files:** Create `README.md`

- [ ] **Step 1:** 写 README：项目简介、`uv sync` / 选 kernel / `jupyter nbconvert --execute` 运行步骤、文件说明、依赖、预计运行时间、作者信息。
- [ ] **Step 2:** 通读 notebook：中文表达、公式渲染、图清晰、每节有分析、无残留调试输出。
- [ ] **Step 3（必做，默认交付形态）：** 把 `diffusion_na.py` 内联进 notebook 顶部"工具区"cell（替换 `import diffusion_na as dna` 为内联定义 + `dna=sys.modules[__name__]` 兼容写法或直接命名空间），使 **notebook 单文件即可跑通**（评卷老师无需附带 .py）。内联后重跑 `jupyter nbconvert --execute` 确认仍全绿、所有 assert 通过。`diffusion_na.py`+`tests/` 仍随仓库保留为验证脚手架。
- [ ] **Step 4: Commit** — `git commit -m "docs: README + reproducibility; self-contained notebook"`。

---

## Self-Review（写完计划后自查）

**1. Spec coverage（逐条对照 spec §4/§5）：**
- §0 引言→Task12；§1-2 理论→Task1,2,13；§3-4→Task14;
- 实验①→Task15；②③→Task16；④→Task17；⑤⑥→Task18；⑦→Task19；⑧⑨→Task20；⑩→Task21；
- §17+附录→Task22；环境/复现/代码质量→Task0,11,23。10 个实验全部有任务覆盖。✓

**2. Placeholder scan：** 已消除所有 `...` 占位——`fpe_residual_gaussian`（完整 log-FPE）、`pf_ode_sample`/`pf_ode_exact`、`exact_score_gmm`（批量）、`bias_coeff_theory`（含 G 项）、`bures_w2`/`sliced_wasserstein`/`energy_distance` 均给出完整可运行代码。其余步骤均为精确命令。✓

**3. Type/命名一致性：** 库函数命名跨任务一致（`marginal_var, reverse_drift_coeff, reverse_variance_exact, em_variance_recursion, em_sample, semiimplicit_variance_recursion, pf_ode_sample, pf_ode_exact, pf_ode_drift, bures_w2, sliced_wasserstein, energy_distance, moment_error, stiffness, critical_dt, fpe_residual_gaussian, exact_score_gaussian, exact_score_gmm, gmm_log_marginal, train_score_net_*, eval_score_net, bias_coeff_theory, bias_coeff_empirical`）。`s0`/`DATA_STD=0.5`、`EPS=1e-3`、`SEED=2026`、正步长 `h` 全程统一。✓

**4. 外部审查闭环（2026-05-29 review，已逐条核验并修订）：** ① 反向方差 ODE 含**因子 2**（数值核验 0.2501 vs 错误 0.759）；② 修正系数按 `H=a²V-½V''`、`Φ=exp(∫-2a)` 正确推导 + Richardson 交叉验证；③ EM/稳定域统一正步长 `(1-ah)`；④ PF ODE 反向步进符号修正 + 闭式 `x∝√v` 作机器精度参照；⑤ 指数法"近机器精度"降级为"降低误差常数"；⑥ FPE 完整 log-FPE，网络只定性；⑦ SDE 自校正改探索性；⑧ GMM 批量、energy_distance 子采样、Bures-W2 对称化+特征值截断、sliced-W 主指标。

**5. 范围决策：** 审查建议砍至主线（适合 solo 人类）；但用户明确要求"内容丰富撑 50%/团队规模"，且实现方由 Claude + 并行编排——故**保持丰富体量**，仅把易翻车的硬断言（⑦自校正、⑨网络FPE）降级为探索/定性，使广度不依赖脆弱断言；解析高斯硬主线（①②③④⑤⑥⑧解析部分）为铁底。
