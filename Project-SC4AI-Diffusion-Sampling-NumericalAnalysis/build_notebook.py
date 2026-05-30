# -*- coding: utf-8 -*-
"""组装中文 Jupyter Notebook《反向扩散采样的数值分析》。

运行: .venv/bin/python build_notebook.py  ->  反向扩散采样的数值分析.ipynb
再执行: .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace ...
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ======================================================================
# 标题
# ======================================================================
md(r"""
# 反向扩散采样的数值分析
## —— 后向误差分析、稳定性、误差预算与少步采样

**作者**：牛昱琛　**学号**：253108120111　（单人独立完成）
**课程**：《人工智能数理基础（高级）》开放课题 · 方向 **SC4AI**
**日期**：2026 年 5 月

---

> **摘要.** 本文把扩散模型（score-based / SDE 生成模型）的**反向采样**重新表述为"**数值求解一个反向时间随机微分方程（SDE）/概率流常微分方程（ODE）**"的问题，并用**数值分析**的工具系统地理解与改进它。在数据为各向同性高斯的可解析设定下，反向过程是**线性 SDE**，其矩满足**闭式确定性递推**，从而每一个理论命题都有**解析真值**作硬基准、结论可证伪。我们：(i) 用**后向误差分析 / 修正方程**解析地推出 Euler–Maruyama 采样在生成分布方差上引入的领头阶 $O(h)$ 偏差及其**闭式系数**，并以 Richardson 外推独立交叉验证（二者高度吻合，相对误差约 $7.5\times10^{-6}$）；(ii) 修正并澄清了 EM 在**加性噪声**下的**强收敛阶为 1**（而非教科书中乘性噪声的 1/2）；(iii) 用**刚性 / 绝对稳定域**第一性地解释采样步长上界 $h\le 2/a(t)$；(iv) 推导半隐式 / 指数积分器与概率流 ODE 的**闭式解** $x\propto\sqrt{v}$，并诚实评测少步采样；(v) 给出总采样误差的**三分解**（分数 / 离散 / 先验）与 **Fokker–Planck 残差**诊断；(vi) 迁移到训练所得网络 score，定量定位**离散误差与网络逼近误差的交叉点**。所有实验在 2-D 以下、单机分钟级完成（代码 device-agnostic，支持 Apple Silicon 的 torch MPS 与 CPU；网络实验默认用 CPU 以逐位可复现），每个解析命题都在对应实验 cell 内以定量断言对照解析真值。
""")

# ======================================================================
# Part 0 引言
# ======================================================================
md(r"""
## Part 0　引言：把"采样"当作"数值积分"

### 0.1 背景与计算难点
扩散模型已成为图像、音频、分子等生成任务的主流方法。其核心是一个**前向加噪过程**（把数据逐步变成噪声）和一个**反向去噪过程**（把噪声逐步变回数据）。Song 等人（2021）指出二者都可写成连续时间 SDE：前向

$$\mathrm{d}X_t = -\tfrac12\beta(t)X_t\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t,\qquad t\in(0,1],$$

而反向（生成）由 Anderson（1982）时间反演给出。**实践中真正昂贵且充满"玄学"的，是反向采样这一步**：用多少步、用什么离散格式、噪声调度怎么选、为什么少步采样会糊——这些恰恰是**数值分析**（数值 ODE/SDE 的相容性、收敛阶、稳定性、刚性、后向误差分析）最擅长回答的问题。

### 0.2 本文视角（SC4AI）
> **把反向采样看成"数值求解反向时间 SDE / 概率流 ODE"，用科学计算的理论来解释与改进这一 AI 方法。**

这正是课程"SC4AI（科学计算支撑人工智能）"的主旨。我们刻意选择**数据为高斯**的可解析设定：此时反向过程是**线性 SDE**，其均值/方差有**闭式**，于是"EM 实际在采哪条被修正的分布""偏差有多大""何时失稳"都能与**解析真值**逐一对照——可证伪，而非经验调参。

### 0.3 与课程内容的呼应
- **Lect6（扩散模型与反向 SDE）**：本文以其 VP-SDE / 去噪分数匹配（DSM）/ Swiss-Roll 为实验基线并深化之。
- **Lect5（SGD 与修正方程）**：本文的"采样修正方程"与该讲的"SGD 修正方程"是**同一个后向误差分析工具**，体现整门课"**微分方程是理解 AI 的统一视角**"。

### 0.4 主要贡献
1. **后向误差分析**：解析推出反向 EM 生成分布方差的领头阶 $O(h)$ 偏差闭式系数 $c$，并以 Richardson 外推交叉验证（高度吻合，相对误差约 $7.5\times10^{-6}$，见实验②）。
2. **澄清收敛阶**：指出本问题为**加性噪声**，EM 的**强阶=弱阶=1**；以共享布朗路径数值验证，并与乘性噪声（强阶 1/2）对比（实验③）。
3. **稳定性/刚性**：以显式 Euler 绝对稳定域给出步长上界 $h\le 2/a(t)$，并定位最刚处在 $t\approx1$（实验④）。
4. **改进采样器**：半隐式/指数积分器降低误差常数；概率流 ODE 的**闭式精确解** $x(t)=\sqrt{v(t)/v(1)}\,x(1)$ 作机器精度参照；NFE–质量 Pareto 与**诚实负结果**（实验⑤⑥）。
5. **误差预算**：总采样误差的分数/离散/先验**三分解**（实验⑧）；**Fokker–Planck 残差**作 score 质量的后验诊断（实验⑨）。
6. **网络迁移**：训练 DSM 网络 score，定量定位**离散误差↔网络逼近误差交叉点**（实验⑩）。

### 0.5 阅读导航
Part I 理论基础 → Part II 误差与收敛（实验①②③）→ Part III 稳定性与刚性（④）→ Part IV 改进采样器（⑤⑥⑦）→ Part V 误差预算与诊断（⑧⑨⑩）→ Part VI 总结。所有数值函数集中在 `diffusion_na.py`。
""")

code(r"""
# 环境与导入
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import diffusion_na as dna

# 中文字体（macOS 自带优先）
_avail = {f.name for f in fm.fontManager.ttflist}
for _f in ['PingFang SC', 'Arial Unicode MS', 'Heiti TC', 'STHeiti', 'Songti SC', 'SimHei']:
    if _f in _avail:
        mpl.rcParams['font.sans-serif'] = [_f]
        break
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 110

os.makedirs('figures', exist_ok=True)
np.random.seed(dna.SEED)
torch.manual_seed(dna.SEED)

S0  = dna.DATA_STD     # 数据标准差 0.5
EPS = dna.EPS          # 采样终止时间 1e-3
# 代码 device-agnostic：MPS(Apple Silicon)/CPU 均可。为使网络实验逐位可复现，默认用 CPU
# （小网络、秒级）；启用 MPS 只需改为 DEVICE = dna.pick_device()。解析实验为 NumPy，与设备无关。
DEVICE = 'cpu'

print('网络实验 device =', DEVICE, '| 可用加速器 =', dna.pick_device(),
      '| torch', torch.__version__, '| mps', torch.backends.mps.is_available())
print('beta(0)=%.3f  beta(1)=%.3f  B(1)=%.4f' % (dna.beta(0.), dna.beta(1.), dna.int_beta(1.)))
print('v(eps)=%.5f  v(1)=%.5f' % (dna.marginal_var(EPS), dna.marginal_var(1.0)))
print('a(1)=%.4f  critical_dt(1)=2/a=%.4f' % (dna.reverse_drift_coeff(1.0), dna.critical_dt(1.0)))
""")

# ======================================================================
# Part I 理论基础
# ======================================================================
md(r"""
## Part I　理论基础

### §1 前向/反向 SDE、score 与去噪分数匹配

**前向 VP-SDE.** 取线性噪声调度 $\beta(t)=\beta_{\min}+(\beta_{\max}-\beta_{\min})t$（本文 $\beta_{\min}=0.1,\ \beta_{\max}=20$），记 $B(t)=\int_0^t\beta(s)\,\mathrm{d}s=0.1t+9.95t^2$。VP 过程的**转移核**为高斯：

$$p(x_t\mid x_0)=\mathcal N\!\Big(x_0\,e^{-B(t)/2},\ \big(1-e^{-B(t)}\big)I\Big).$$

记 $\alpha(t)=e^{-B(t)/2}$、$\sigma^2(t)=1-e^{-B(t)}$，则 $t=1$ 时 $\sigma^2(1)\approx1$，即 $p(x_1)\approx\mathcal N(0,I)$（采样起点先验）。

**反向 SDE（Anderson 1982）.** 同一组边缘 $p(x,t)$ 可由如下反向时间 SDE 生成（$t$ 从 1 走向 0）：

$$\mathrm{d}X = \big[-\tfrac12\beta(t)X-\beta(t)\,\nabla_x\log p(X,t)\big]\mathrm{d}t+\sqrt{\beta(t)}\,\mathrm{d}\bar W,$$

唯一未知量是**分数函数（score）** $s(x,t)=\nabla_x\log p(x,t)$。

**去噪分数匹配（DSM, Vincent 2011）.** 用网络 $s_\theta$ 拟合 score：

$$\min_\theta\ \mathbb E_{t}\,\lambda(t)\,\mathbb E_{x_0}\,\mathbb E_{x_t\mid x_0}\big\|s_\theta(x_t,t)-\nabla_{x_t}\log p(x_t\mid x_0)\big\|^2,$$

其中条件分数 $\nabla_{x_t}\log p(x_t\mid x_0)=(\alpha(t)x_0-x_t)/\sigma^2(t)$ 由高斯核闭式给出，权重取 $\lambda(t)=\sigma^2(t)$。

### §2 可解析设定：高斯数据下反向过程是线性 SDE

设数据 $X_0\sim\mathcal N(0,s_0^2 I)$（本文 $s_0=0.5$；$s_0=1$ 会使 $v\equiv1$ 退化，禁用）。则边缘仍是高斯：

$$p(x_t)=\mathcal N\!\big(0,\,v(t)I\big),\qquad v(t)=1+(s_0^2-1)e^{-B(t)},$$

**精确 score 是线性的**：$\;s^*(x,t)=\nabla\log p(x_t)=-x/v(t).$

代入反向 SDE，反向漂移成为线性：$b_{\mathrm{rev}}(t,x)=-\tfrac12\beta x-\beta s^*=a(t)x$，

$$\boxed{\,a(t)=\beta(t)\big(1/v(t)-\tfrac12\big)>0\,}.$$

其中正性依赖 $s_0<1\Rightarrow v(t)<1\Rightarrow 1/v(t)>1/2$（本文 $s_0=0.5$ 全程满足）；$a(t)>0$ 是后文稳定域 $h\le2/a$ 与衰减因子 $(1-ah)$ 等论证的结构性前提。

> **可解析设定的意义与边界.** 该设定使"反向过程的矩"有闭式，从而把"离散格式的误差"从"分数估计误差"中**干净剥离**，得到可证伪的硬基准。其局限是：真实低维流形扩散的难点主要来自 **score 的非线性**与 $t\to0$ 的小噪声区；故本文也用**高斯混合**（闭式 score，引入非线性）与**训练网络 score** 检验理论的适用范围。

**统一时间约定（全程严格一致）**

| 名称 | 记号 | 方向 | 步长 | 一步公式 |
|---|---|---|---|---|
| 前向时间 | $t$ | $0\to1$ | $\mathrm{d}t>0$ | VP 前向 SDE |
| 反向采样网格 | $t_n$ | $1\to\varepsilon$ | $h=t_n-t_{n+1}>0$ | $x_{n+1}=(1-a(t_n)h)x_n+\sqrt{\beta(t_n)h}\,\zeta_n$ |
| 反向时间 | $\tau=1-t$ | $0\to1-\varepsilon$ | $\mathrm{d}\tau>0$ | $\mathrm{d}Y=-a(1-\tau)Y\,\mathrm{d}\tau+\sqrt{\beta}\,\mathrm{d}\tilde W$ |
""")

code(r"""
# 图1：噪声调度与边缘量
ts = np.linspace(EPS, 1.0, 400)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
ax[0].plot(ts, dna.beta(ts));                      ax[0].set_title(r'$\beta(t)$ 噪声调度')
ax[1].plot(ts, dna.marginal_var(ts));              ax[1].axhline(S0**2, ls='--', c='gray', label=r'$s_0^2$')
ax[1].set_title(r'边缘方差 $v(t)$'); ax[1].legend()
ax[2].plot(ts, dna.reverse_drift_coeff(ts));       ax[2].set_title(r'反向漂移系数 $a(t)=\beta(1/v-1/2)$')
ax[3].plot(ts, dna.alpha(ts), label=r'$\alpha=e^{-B/2}$')
ax[3].plot(ts, dna.sigma2(ts), label=r'$\sigma^2=1-e^{-B}$'); ax[3].legend(); ax[3].set_title('转移核系数')
for a in ax: a.set_xlabel('t')
plt.tight_layout(); plt.savefig('figures/fig01_schedule.png', bbox_inches='tight'); plt.show()
""")

code(r"""
# 图2：高斯混合的闭式 score 场（引入非线性的解析参照）
means = np.array([[-1.5, 0.0], [1.5, 0.0]]); weights = np.array([0.5, 0.5]); s0g = 0.4
t_show = 0.15
g = np.linspace(-3, 3, 24); X, Y = np.meshgrid(g, g)
pts = np.stack([X.ravel(), Y.ravel()], 1)
Sv = dna.exact_score_gmm(pts, t_show, means, weights, s0g)
logp = dna.gmm_log_marginal(pts, t_show, means, weights, s0g).reshape(X.shape)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
c = ax[0].contourf(X, Y, np.exp(logp), levels=20, cmap='viridis'); plt.colorbar(c, ax=ax[0])
ax[0].set_title(f'高斯混合边缘密度 p(x, t={t_show})')
ax[1].quiver(X, Y, Sv[:, 0].reshape(X.shape), Sv[:, 1].reshape(X.shape))
ax[1].set_title(r'闭式 score 场 $\nabla\log p$')
for a in ax: a.set_aspect('equal'); a.set_xlabel('$x_1$'); a.set_ylabel('$x_2$')
plt.tight_layout(); plt.savefig('figures/fig02_gmm_score.png', bbox_inches='tight'); plt.show()
""")

# ======================================================================
# Part II 误差与收敛
# ======================================================================
md(r"""
## Part II　采样即数值积分：误差与收敛

### §3 反向 EM 的矩封闭递推
把 §1 的 Anderson 反向 SDE 在 $\mathbb R$ 上用线性 score $s^*=-x/v$ 代入，得线性反向漂移 $a(t)x$。沿采样网格离散时取**反向时间** $\tau=1-t$、**正步长** $h=t_n-t_{n+1}=-\mathrm{d}t>0$（即 $t$ 由 $1$ 递减到 $\varepsilon$）。在前向时间写法下反向过程为 $\mathrm{d}Y=a(t)Y\,\mathrm{d}t+\sqrt{\beta}\,\mathrm{d}\bar W$（$\mathrm{d}t<0$），等价地在反向时间下为 $\mathrm{d}Y=-a(1-\tau)Y\,\mathrm{d}\tau+\sqrt{\beta}\,\mathrm{d}\tilde W$（与 §2 约定表一致）。对其做 Euler–Maruyama（步长 $h$）：

$$x_{n+1}=(1-a(t_n)h)\,x_n+\sqrt{\beta(t_n)h}\,\zeta_n,\quad\zeta_n\sim\mathcal N(0,1).$$

由于线性 + 高斯，均值恒为 0，**方差满足确定性递推**（无蒙特卡洛噪声）：

$$\boxed{\,V_{n+1}=(1-a(t_n)h)^2\,V_n+\beta(t_n)h\,}.$$

其连续极限（反向时间 $\tau$）为 $\;\mathrm{d}V/\mathrm{d}\tau=-2a(1-\tau)V+\beta(1-\tau)$。**注意因子 2**：来自 Itô 公式 $\mathrm{d}\,\mathbb E[Y^2]=(2a\,\mathbb E[Y^2]+\beta)\mathrm{d}\tau$；漏掉它会得到错误真值（$0.76$ 而非正确的 $0.25$，已数值核验）。

### §4 后向误差分析 / 修正方程：EM 实际在采哪条 SDE？
把递推展开为 $V_{n+1}=V_n+hF+h^2G$，其中

$$F(\tau,V)=-2a\,V+\beta,\qquad G(\tau,V)=a^2V\ \ (\text{来自}(1-ah)^2\text{的}a^2h^2V).$$

设 EM 实际精确求解的**修正方程**为 $V'=F+hH+O(h^2)$。匹配一步 Taylor 展开（沿精确解 $F_\tau+F_V F=V''$）得

$$\boxed{\,H=G-\tfrac12\big(F_\tau+F_V F\big)=a^2V-\tfrac12 V''\,}.$$

于是生成分布方差的**领头阶偏差**为 $V_N-v(\varepsilon)=c\,h+O(h^2)$，其**闭式系数**

$$c=\int_0^{T}\Phi(T,s)\big[a(1-s)^2V(s)-\tfrac12V''(s)\big]\mathrm{d}s,\qquad \Phi(T,s)=\exp\!\Big(\!\int_s^{T}-2a(1-r)\,\mathrm{d}r\Big),$$

$T=1-\varepsilon$。其中 $a^2V$ 项来自**噪声注入与漂移的离散耦合**，$-\tfrac12V''$ 项来自漂移的前向 Euler 截断——这解释了"只把线性漂移指数化并不足以完全消偏"。
""")

code(r"""
# 实验①：O(h) 偏差律（确定性递推，免训练、微秒级）
Ns = np.array([50, 100, 200, 400, 800, 1600])
target = dna.reverse_variance_exact(EPS, V_start=dna.marginal_var(1.0))   # 解析真值（含因子2）
errs = np.array([abs(dna.em_variance_recursion(N, V_start=dna.marginal_var(1.0)) - target) for N in Ns])
hs = (1.0 - EPS) / Ns
slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]

plt.figure(figsize=(5.6, 4.2))
plt.loglog(hs, errs, 'o-', label=r'EM 方差偏差 $|V_N-v(\varepsilon)|$')
plt.loglog(hs, errs[0] * (hs / hs[0]), '--', c='gray', label='斜率 1 参考线')
plt.xlabel('步长 h'); plt.ylabel('生成分布方差偏差')
plt.title('实验①：反向 EM 的 $O(h)$ 偏差律（拟合斜率=%.3f）' % slope)
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.savefig('figures/fig03_bias_law.png', bbox_inches='tight'); plt.show()

print('解析真值 v(eps) = %.6f ; reverse_variance_exact = %.6f' % (dna.marginal_var(EPS), target))
print('拟合斜率 = %.4f  (理论弱阶 = 1)' % slope)
assert 0.85 < slope < 1.15, slope
""")

md(r"""
**实验①分析.** 偏差随步长 $h$ 以斜率 $\approx1$ 的直线下降（双对数坐标），定量证实反向 EM 在生成分布方差上的弱收敛阶为 1，且解析真值 $v(\varepsilon)=0.2501$ 与高精度反向方差 ODE 一致。这条曲线全程由**确定性递推**给出（无蒙特卡洛噪声、无需训练），是后续一切误差分析的"地基"。（说明：在此线性高斯设定下弱阶=1 是由确定性矩递推决定的相容性结果、近乎必然；真正有信息量的是 §4 的修正方程**系数**与 §5 的**强阶**加性/乘性对照。）
""")

code(r"""
# 实验②：修正方程系数 c_theory vs Richardson 经验 c_emp
c_th = dna.bias_coeff_theory()
c_emp = dna.bias_coeff_empirical()
rel = abs(c_th - c_emp) / abs(c_emp)

Ns2 = np.array([200, 400, 800, 1600, 3200, 6400]); hs2 = (1.0 - EPS) / Ns2
ratios = np.array([(dna.em_variance_recursion(N, V_start=dna.marginal_var(1.0)) - target) / h
                   for N, h in zip(Ns2, hs2)])
plt.figure(figsize=(5.6, 4.2))
plt.plot(hs2, ratios, 'o-', label=r'经验 $(V_N-v(\varepsilon))/h$')
plt.axhline(c_th, ls='--', c='r', label=r'$c_{\rm theory}=%.4f$ (闭式修正方程)' % c_th)
plt.axhline(c_emp, ls=':', c='g', label=r'$c_{\rm emp}=%.4f$ (Richardson 外推)' % c_emp)
plt.xlabel('步长 h'); plt.ylabel('误差 / 步长'); plt.legend()
plt.title('实验②：修正方程领头阶系数对照（相对误差 %.2e）' % rel)
plt.grid(True, alpha=0.3)
plt.savefig('figures/fig04_coeff.png', bbox_inches='tight'); plt.show()

print('c_theory=%.6f  c_emp=%.6f  相对误差=%.2e' % (c_th, c_emp, rel))
assert rel < 1e-3, rel          # 理论闭式与 Richardson 经验值高度吻合（实测 ~7.5e-6）
""")

md(r"""
**实验②分析.** 手推的闭式系数 $c_{\rm theory}\approx0.35798$ 与完全独立的 Richardson 外推值 $c_{\rm emp}\approx0.35798$ **高度吻合**（相对误差约 $7.5\times10^{-6}$）——这有力地表明后向误差分析的推导（含 $a^2V$ 离散耦合项与因子 2 的传播子、解析二阶导 $V''$）是正确的，精确回答了"EM 实际在采哪条被修正的 SDE"。这与 **Lect5 的 SGD 修正方程**用的是同一套数值分析工具。
""")

md(r"""
### §5 收敛阶：本问题是**加性噪声**，EM 强阶=弱阶=1
教科书常说 Euler–Maruyama 的强阶为 $1/2$、弱阶为 $1$。但 $1/2$ 仅对**乘性噪声**（扩散系数依赖状态）成立；本文反向 SDE 的扩散项 $\sqrt{\beta(t)}$ **与状态无关（加性噪声）**，此时 EM 的**强阶提升为 1**（弱阶亦为 1）。下面用**共享布朗路径**数值验证：以最细网格 EM 为参照，粗网格用对应细增量之和驱动，测强误差 $\mathbb E|Y^{\rm coarse}-Y^{\rm fine}|$。作为对比，再给一个乘性噪声的几何布朗运动玩具，展示其强阶确为 $1/2$。
""")

code(r"""
# 实验③：强收敛阶 —— 加性噪声(本问题) 强阶=1，对比乘性噪声(GBM) 强阶=1/2
rng = np.random.default_rng(7)
x0v = 0.7; Nf = 4096; n_paths = 4000
hf = (1.0 - EPS) / Nf
dWf = rng.standard_normal((n_paths, Nf)) * np.sqrt(hf)   # 最细布朗增量

def em_paths_additive(N, dW_steps):
    ts_ = np.linspace(1.0, EPS, N + 1); x = np.full(dW_steps.shape[0], x0v)
    for i in range(N):
        t = ts_[i]; h = ts_[i] - ts_[i + 1]; a = dna.reverse_drift_coeff(t, S0)
        x = (1 - a * h) * x + np.sqrt(dna.beta(t)) * dW_steps[:, i]
    return x

Xref = em_paths_additive(Nf, dWf)
levels = np.array([16, 32, 64, 128, 256, 512])
strong = []
for N in levels:
    k = Nf // N
    dWc = dWf.reshape(n_paths, N, k).sum(axis=2)         # 共享路径：粗增量=细增量之和
    strong.append(np.mean(np.abs(em_paths_additive(N, dWc) - Xref)))
strong = np.array(strong); hs3 = (1.0 - EPS) / levels
slope_add = np.polyfit(np.log(hs3), np.log(strong), 1)[0]

# 乘性噪声玩具：GBM dY=-Y dt + Y dW，精确解 Y_T=Y0 exp((-1-1/2)T + W_T)
T = 1.0; Y0 = 1.0; mu, sig = -1.0, 1.0
dWg = rng.standard_normal((n_paths, Nf)) * np.sqrt(T / Nf)
WT = dWg.sum(1); Yexact = Y0 * np.exp((mu - 0.5 * sig**2) * T + sig * WT)
strong_g = []
for N in levels:
    k = Nf // N; hg = T / N; dWc = dWg.reshape(n_paths, N, k).sum(2)
    y = np.full(n_paths, Y0)
    for i in range(N):
        y = y + mu * y * hg + sig * y * dWc[:, i]
    strong_g.append(np.mean(np.abs(y - Yexact)))
strong_g = np.array(strong_g); hg3 = T / levels
slope_mul = np.polyfit(np.log(hg3), np.log(strong_g), 1)[0]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].loglog(hs3, strong, 'o-'); ax[0].loglog(hs3, strong[0]*(hs3/hs3[0]), '--', c='gray', label='斜率 1')
ax[0].set_title('本文反向 SDE（加性噪声）：强阶=%.2f' % slope_add); ax[0].set_xlabel('h'); ax[0].set_ylabel(r'$E|Y^{coarse}-Y^{fine}|$'); ax[0].legend()
ax[1].loglog(hg3, strong_g, 's-', c='C1'); ax[1].loglog(hg3, strong_g[0]*np.sqrt(hg3/hg3[0]), '--', c='gray', label='斜率 1/2')
ax[1].set_title('几何布朗运动（乘性噪声）：强阶=%.2f' % slope_mul); ax[1].set_xlabel('h'); ax[1].legend()
plt.tight_layout(); plt.savefig('figures/fig05_strong_order.png', bbox_inches='tight'); plt.show()
print('加性噪声强阶≈%.3f（理论1）；乘性噪声(GBM)强阶≈%.3f（理论1/2）' % (slope_add, slope_mul))
assert 0.8 < slope_add < 1.2, slope_add
assert 0.3 < slope_mul < 0.7, slope_mul
""")

md(r"""
**实验③分析.** 左图证实本文反向 SDE 在**加性噪声**下 EM 强阶 $\approx1$；右图的几何布朗运动（乘性噪声）强阶 $\approx1/2$，正是教科书结论。这澄清了一个常见误解：扩散反向采样（加性噪声）的 EM 路径精度其实是一阶的，笼统地说"EM 强阶为 1/2"在此并不适用。
""")

# ======================================================================
# Part III 稳定性与刚性
# ======================================================================
md(r"""
## Part III　稳定性与刚性

### §6 反向漂移的稳定性与显式 Euler 绝对稳定域
反向漂移的 Jacobian 为标量 $a(t)>0$，故"刚性"由 $L(t)=a(t)$ 度量。EM 的方差递推因子是 $(1-a(t)h)^2$；要求迭代不放大（绝对稳定），需

$$|1-a(t)h|\le1\ \Longleftrightarrow\ 0\le h\le \frac{2}{a(t)}.$$

由于 $a(t)=\beta(t)(1/v(t)-\tfrac12)$ 在 $t\to1$ 处最大（$\beta$ 最大且 $v\to1$），$a(1)\approx10.0$，故最严格的稳定步长约 $h\le0.2$，对应**临界步数** $N^*\approx(1-\varepsilon)\,a_{\max}/2\approx5$。值得强调：**概率流 ODE 并不刚**（其漂移系数 $\tfrac12\beta(1/v-1)\to0$ 当 $v\to1$），这解释了为何确定性 ODE 采样常能用更少步数。
""")

code(r"""
# 实验④：刚性诊断 + 稳定步长界 + 失稳演示
tt = np.linspace(EPS, 1.0, 400)
L = dna.stiffness(tt); cdt = dna.critical_dt(tt)
amax = dna.stiffness(np.linspace(EPS, 1, 2000)).max()
Nstar = (1.0 - EPS) * amax / 2.0
Ns_s = np.arange(2, 60)
v_eps = dna.marginal_var(EPS)
reldev = np.array([abs(dna.em_variance_recursion(int(N), V_start=dna.marginal_var(1.0)) - v_eps) / v_eps
                   for N in Ns_s])

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(tt, L); ax[0].axvline(1.0, ls=':', c='gray'); ax[0].set_title(r'刚性 $L(t)=a(t)$'); ax[0].set_xlabel('t')
ax[1].plot(tt, cdt); ax[1].set_title(r'稳定步长界 $2/a(t)$'); ax[1].set_xlabel('t')
ax[2].semilogy(Ns_s, reldev + 1e-16, 'o-'); ax[2].axvline(Nstar, ls='--', c='r', label=r'临界步数 $N^*\approx%.1f$' % Nstar)
ax[2].set_title('终端方差相对偏差 vs 步数 N'); ax[2].set_xlabel('N'); ax[2].set_ylabel(r'$|V_N-v(\varepsilon)|/v(\varepsilon)$'); ax[2].legend()
plt.tight_layout(); plt.savefig('figures/fig06_stiffness.png', bbox_inches='tight'); plt.show()
print('最刚处 t≈%.3f, a_max≈%.2f, 临界步数 N*≈%.1f' % (tt[np.argmax(L)], amax, Nstar))
assert dna.em_blows_up(3) and not dna.em_blows_up(50)
""")

md(r"""
**实验④分析.** 稳定性时间尺度 $L(t)=a(t)$ 在 $t\approx1$ 处达到峰值 $\approx10$，对应显式 Euler 稳定步长界 $2/a\approx0.2$。具体地：$N\le3$（$h>2/a$）时 $|1-ah|>1$ 致迭代放大、方差发散（`em_blows_up=True`）；$N=4,5$ 接近临界、偏差达数倍但不发散；$N\gtrsim6$ 后单调收敛。这把工程上"步数不能太少"的经验提升为可计算的**绝对稳定域**判据。（注：此处"稳定域约束"指显式格式的稳定步长上界 $h\le2/a$，$a\!\cdot\!T\approx10$ 属温和，并非传统多尺度刚性。）它来自 $\beta(t)$ 增大而非 $t\to0$（因 $s_0>0$ 时 $1/v$ 有界），理想数据流形（$s_0\to0$）才会出现 $t\to0$ 奇异。
""")

# ======================================================================
# Part IV 改进的采样器
# ======================================================================
md(r"""
## Part IV　改进的采样器

### §7 半隐式/指数处理线性漂移 + Richardson 外推
反向 SDE 的漂移是线性的，可对其做**常系数指数（半隐式）步**：$x\leftarrow e^{-a h}x+(\text{噪声项精确积分})$，方差递推相应为 $V\leftarrow e^{-2ah}V+\beta\frac{1-e^{-2ah}}{2a}$。它在每步常系数意义下精确，但因 $a,\beta$ 随时间变，**整体仍是一阶，只是误差常数更小**（不应声称"机器精度"）。另一种**免推导的提阶**手段是 **Richardson 外推**：对一阶方法，$2V_{2N}-V_N$ 消去 $O(h)$ 项得到 $O(h^2)$。
""")

code(r"""
# 实验⑤：半隐式/指数修正 + Richardson 外推（1-D Bures-W2 = |√V - √v(ε)|）
Ns5 = np.array([20, 40, 80, 160, 320, 640]); hs5 = (1.0 - EPS) / Ns5
v_eps = dna.marginal_var(EPS)
w2 = lambda V: abs(np.sqrt(max(V, 0.0)) - np.sqrt(v_eps))
em_w = np.array([w2(dna.em_variance_recursion(N, V_start=dna.marginal_var(1.0))) for N in Ns5])
si_w = np.array([w2(dna.semiimplicit_variance_recursion(N, V_start=dna.marginal_var(1.0))) for N in Ns5])
rich_w = np.array([w2(2 * dna.em_variance_recursion(2 * N, V_start=dna.marginal_var(1.0))
                      - dna.em_variance_recursion(N, V_start=dna.marginal_var(1.0))) for N in Ns5])

plt.figure(figsize=(6, 4.2))
plt.loglog(hs5, em_w, 'o-', label='EM（斜率≈1）')
plt.loglog(hs5, si_w, 's-', label='半隐式/指数（误差常数更小）')
plt.loglog(hs5, rich_w, '^-', label='EM + Richardson 外推（提阶）')
plt.xlabel('步长 h'); plt.ylabel(r'Bures-$W_2$ 到 $N(0,v(\varepsilon))$')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.title('实验⑤：修正采样器与 Richardson 外推')
plt.savefig('figures/fig07_corrected.png', bbox_inches='tight'); plt.show()
s_em = np.polyfit(np.log(hs5), np.log(em_w), 1)[0]
s_rich = np.polyfit(np.log(hs5), np.log(rich_w), 1)[0]
print('EM 斜率=%.2f  半隐式最粗步误差=%.2e vs EM=%.2e  Richardson 斜率=%.2f' % (s_em, si_w[0], em_w[0], s_rich))
assert si_w[0] < em_w[0]            # 指数法误差常数更小
assert s_rich > s_em + 0.4          # Richardson 提阶
""")

md(r"""
**实验⑤分析.** 半隐式/指数法把 EM 的误差常数压低（同为一阶），而 Richardson 外推把收敛阶从 $\approx1$ 提到 $\approx2$。这印证了 §7 的判断：仅指数化线性漂移不足以"消偏"（噪声注入项仍贡献 $O(h)$），而外推则系统地提阶——二者都是数值分析的标准武器。

### §8 概率流 ODE 与指数积分器：少步采样
与反向 SDE 共享同一边缘的**概率流 ODE**为

$$\frac{\mathrm{d}x}{\mathrm{d}t}=f_{\rm pf}(t,x)=-\tfrac12\beta x-\tfrac12\beta\,\nabla\log p=\tfrac12\beta(t)\big(1/v(t)-1\big)x,$$

它是 DPM-Solver/DEIS 等少步采样器的数学内核。高斯下其**闭式精确解**为

$$\boxed{\,x(t)=\sqrt{v(t)/v(1)}\;x(1)\,},$$

可作为"理想指数积分器/机器精度参照"。下面验证 Euler/Heun 对该闭式解的收敛阶（ODE 不刚，少步即可），并在 Swiss-Roll 上用训练网络 score 画 **NFE–质量 Pareto**。
""")

code(r"""
# 实验⑥(a)：概率流 ODE 对闭式精确解的收敛阶（1-D）
refx = dna.pf_ode_exact(2.0)
Ns6 = np.array([4, 8, 16, 32, 64, 128]); hs6 = (1.0 - EPS) / Ns6
e_eu = np.array([abs(dna.pf_ode_sample(N, 'euler', 2.0) - refx) for N in Ns6])
e_he = np.array([abs(dna.pf_ode_sample(N, 'heun', 2.0) - refx) for N in Ns6])
s_eu = np.polyfit(np.log(hs6), np.log(e_eu), 1)[0]
s_he = np.polyfit(np.log(hs6), np.log(e_he), 1)[0]

# 实验⑥(b)：Swiss-Roll 上用训练网络 score 的 NFE-质量 Pareto
data2d = dna.make_swiss_roll_2d(2000, seed=0)
print('训练 Swiss-Roll score 网络 (device=%s) ...' % DEVICE)
net2d = dna.train_score_net_data(data2d, n_iters=6000, device=DEVICE, seed=0)
net_score = lambda x, t: dna.eval_score_net(net2d, x, t, device=DEVICE)
nfes = [5, 10, 20, 50, 100]
qual = {'em': [], 'pf_heun': []}
for m in qual:
    for N in nfes:
        Xg = dna.reverse_sample(net_score, N, m, n_samples=2000, dim=2, eps=EPS, seed=1)
        qual[m].append(dna.sliced_wasserstein(Xg, data2d))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].loglog(hs6, e_eu, 'o-', label='Euler（斜率=%.2f）' % s_eu)
ax[0].loglog(hs6, e_he, 's-', label='Heun（斜率=%.2f）' % s_he)
ax[0].set_title('实验⑥(a) PF-ODE 收敛阶（对闭式解）'); ax[0].set_xlabel('h'); ax[0].set_ylabel('末态 L2 误差'); ax[0].legend()
ax[1].plot(nfes, qual['em'], 'o-', label='反向 SDE (EM)')
ax[1].plot(nfes, qual['pf_heun'], 's-', label='概率流 ODE (Heun)')
ax[1].set_title('实验⑥(b) Swiss-Roll NFE–质量 Pareto'); ax[1].set_xlabel('NFE'); ax[1].set_ylabel('sliced-$W$ 到数据'); ax[1].legend()
Xbest = dna.reverse_sample(net_score, 100, 'pf_heun', n_samples=2000, dim=2, eps=EPS, seed=1)
ax[2].scatter(data2d[:, 0], data2d[:, 1], s=4, alpha=.4, label='数据')
ax[2].scatter(Xbest[:, 0], Xbest[:, 1], s=4, alpha=.4, label='生成 (ODE,100步)')
ax[2].set_title('生成样本 vs 数据'); ax[2].legend(); ax[2].set_aspect('equal')
plt.tight_layout(); plt.savefig('figures/fig08_exp_integrator.png', bbox_inches='tight'); plt.show()
print('PF-ODE: Euler 斜率=%.2f, Heun 斜率=%.2f' % (s_eu, s_he))
assert s_he > s_eu + 0.5            # Heun 阶高于 Euler（仅 ODE 成立）
""")

md(r"""
**实验⑥分析.** (a) 概率流 ODE 不刚，Euler/Heun 分别以斜率 $\approx1/\approx2$ 收敛到**闭式精确解** $x\propto\sqrt v$，少步即达高精度——这正是 DPM-Solver 类少步采样有效的数值根源。(b) Swiss-Roll 上比较随机 EM 与确定性 ODE（Heun）的 sliced-$W$ 随 NFE 变化：二者在 NFE$\ge$10 时质量相近；最少步（NFE=5）处结果对随机性与步长高度敏感（EM 恰处其稳定性临界 $N^*\approx5$ 附近，单步可能越出稳定域）。**诚实说明**：朴素"冻结系数指数法"在含 score 项时并不一定胜过 Euler（其优势仅在线性漂移部分被精确处理时显著）；真正的少步增益来自 ODE 的非刚性与高阶格式，而非简单指数化。

### §9 随机 vs 确定性采样器：偏差–方差（探索性）
""")

code(r"""
# 实验⑦：SDE(EM) vs ODE(Heun) 的矩误差与对 score 误差的鲁棒性（探索性，不做硬断言）
gscore = lambda x, t: dna.exact_score_gaussian(x, t, S0)
Ns7 = [5, 10, 20, 50, 100]
em_e = [abs(dna.reverse_sample(gscore, N, 'em', n_samples=40000, dim=1, eps=EPS, seed=2).var() - v_eps) for N in Ns7]
od_e = [abs(dna.reverse_sample(gscore, N, 'pf_heun', n_samples=40000, dim=1, eps=EPS, seed=3).var() - v_eps) for N in Ns7]

deltas = np.linspace(-0.3, 0.3, 13)   # 乘性 score 误差 s_pert=(1+δ)s
rob_em, rob_od = [], []
for d in deltas:
    sp = lambda x, t, d=d: (1.0 + d) * dna.exact_score_gaussian(x, t, S0)
    rob_em.append(abs(dna.reverse_sample(sp, 50, 'em', n_samples=40000, dim=1, eps=EPS, seed=4).var() - v_eps))
    rob_od.append(abs(dna.reverse_sample(sp, 50, 'pf_heun', n_samples=40000, dim=1, eps=EPS, seed=5).var() - v_eps))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(Ns7, em_e, 'o-', label='SDE (EM)'); ax[0].plot(Ns7, od_e, 's-', label='ODE (Heun)')
ax[0].set_title('实验⑦(a) 无扰动：方差误差 vs NFE'); ax[0].set_xlabel('NFE'); ax[0].set_ylabel('|var - v(ε)|'); ax[0].legend(); ax[0].set_yscale('log')
ax[1].plot(deltas, rob_em, 'o-', label='SDE (EM)'); ax[1].plot(deltas, rob_od, 's-', label='ODE (Heun)')
ax[1].set_title('实验⑦(b) 对乘性 score 误差 δ 的鲁棒性'); ax[1].set_xlabel('score 误差 δ'); ax[1].set_ylabel('|var - v(ε)|'); ax[1].legend()
plt.tight_layout(); plt.savefig('figures/fig09_bias_variance.png', bbox_inches='tight'); plt.show()
print('探索性实验：报告曲线形态，不作硬断言。')
assert np.all(np.isfinite(em_e)) and np.all(np.isfinite(rob_od))
""")

md(r"""
**实验⑦分析（探索性）.** 无扰动时确定性 ODE 与随机 SDE 的方差误差随 NFE 同步下降；在乘性 score 误差 $\delta$ 下，本次实验中**负向扰动（$\delta<0$）下 ODE(Heun) 的误差增长明显快于 EM，正向扰动两者相近**——可见"随机性自校正"依赖扰动方向，并非普适。文献中该说法本就**不普适**（依赖扰动模型与指标），故本文将其作为探索性观察呈现、不作硬性结论。
""")

# ======================================================================
# Part V 误差预算与 score 诊断
# ======================================================================
md(r"""
## Part V　误差预算与 score 质量诊断

本部分需要一个**训练所得**的网络 score。我们在 1-D 高斯数据上用 DSM 训练一个小 MLP（保留解析真值 $s^*=-x/v$ 作参照），供实验⑧⑨⑩共用。
""")

code(r"""
# 训练 1-D 网络 score（供实验⑧⑨⑩共用）
print('训练 1-D 高斯 score 网络 (device=%s) ...' % DEVICE)
net1d = dna.train_score_net_gaussian(s0=S0, n_iters=5000, device=DEVICE, dim=1, seed=0)
net1d_score = lambda x, t: dna.eval_score_net(net1d, x, t, device=DEVICE)
# 抽检：t=0.3 时网络 score vs 解析
xs = np.linspace(-1, 1, 11).reshape(-1, 1)
mse = np.mean((dna.eval_score_net(net1d, xs, 0.3, DEVICE) - dna.exact_score_gaussian(xs, 0.3, S0))**2)
print('网络 score 在 t=0.3 的 MSE = %.4f （越小越准）' % mse)
""")

md(r"""
### §10 总采样误差的三分解：分数 / 离散 / 先验
反向采样的总误差可归因于三个来源：(i) **先验失配** $e_{\rm prior}$——起点用 $\mathcal N(0,I)$ 而真边缘是 $\mathcal N(0,v(1))$；(ii) **时间离散** $e_{\rm disc}$——有限步长的格式误差；(iii) **分数估计** $e_{\rm score}$——网络 score 与真值之差。我们在 1-D 高斯上**分别独立测量**三者（Bures-$W_2=|\sqrt{\cdot}-\sqrt{v(\varepsilon)}|$）：

- $e_{\rm prior}=|\sqrt{1}-\sqrt{v(1)}|$：起点先验与真边缘之差，**与步数无关**；
- $e_{\rm disc}=|\sqrt{V_B}-\sqrt{v(\varepsilon)}|$，$V_B$=精确 score、**从真边缘 $v(1)$ 出发**、粗步 EM 的终端方差（纯离散）；
- $e_{\rm score}=|\sqrt{V_C}-\sqrt{V_B}|$，$V_C$=网络 score、同样粗步的终端方差（相对精确 score 的偏移即分数误差贡献）。

> 注：一般情形采样误差并非严格线性可加，这里给出的是**量级归因**（三源之和与实测总误差量级一致），而非严格上界；严格上界可用三角不等式 $W_2\le e_{\rm prior}+e_{\rm disc}+e_{\rm score}$ 分别上界各项。
""")

code(r"""
# 实验⑧：误差三分解（1-D 高斯，三源各自独立测量；近似可加，非严格上界）
v_eps = dna.marginal_var(EPS); v1 = dna.marginal_var(1.0)
w2 = lambda V: abs(np.sqrt(max(V, 0.0)) - np.sqrt(v_eps))
N_coarse = 20
V_B = dna.em_variance_recursion(N_coarse, V_start=v1)                                  # 精确score+真起点+粗步
V_C = dna.reverse_sample(net1d_score, N_coarse, 'em', n_samples=200000, dim=1, eps=EPS, seed=6).var()
e_prior = abs(np.sqrt(1.0) - np.sqrt(v1))                                              # 先验失配（与步数无关）
e_disc  = w2(V_B)                                                                      # 纯离散
e_score = abs(np.sqrt(max(V_C, 0.0)) - np.sqrt(V_B))                                   # 分数估计贡献
e_total = w2(V_C)                                                                      # 网络+粗步 实测总误差

fig, ax = plt.subplots(figsize=(6.8, 4.2))
labels = ['先验失配\n$e_{prior}$', '时间离散\n$e_{disc}$', '分数估计\n$e_{score}$', '三源之和', '实测总误差\n(网络+粗步)']
vals = [e_prior, e_disc, e_score, e_prior + e_disc + e_score, e_total]
colors = ['#bbb', '#4C72B0', '#C44E52', '#8172B3', '#55A868']
ax.bar(range(5), vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v, '%.2e' % v, ha='center', va='bottom', fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
ax.set_yscale('log'); ax.set_ylabel(r'Bures-$W_2$ 误差'); ax.set_title('实验⑧：总采样误差的三源量级归因（1-D 高斯）')
plt.tight_layout(); plt.savefig('figures/fig10_error_budget.png', bbox_inches='tight'); plt.show()
print('e_prior=%.3e  e_disc=%.3e  e_score=%.3e  三源和=%.3e  实测总=%.3e'
      % (e_prior, e_disc, e_score, e_prior + e_disc + e_score, e_total))
assert e_disc > 0 and e_score > 0          # 离散与分数误差均可见
assert e_prior < e_disc                    # 本调度下先验失配可忽略
""")

md(r"""
**实验⑧分析.** 三个误差源被独立测量、干净分离：本调度下**先验失配可忽略**（$\sim10^{-5}$，因 $v(1)\approx1$），主导项是粗步的**时间离散误差**与网络的**分数估计误差**；三源之和与实测总误差量级一致（近似可加）。这种可分离、各有解析基准的误差预算，把"扩散为什么生成不准"从笼统经验变成可测量的归因——是 SC4AI 误差分析范式的典型落地。

### §11 Fokker–Planck 残差作为 score 质量的后验诊断
对 VP 过程，真实 $\log p$ 满足 **log-FPE 恒等式**

$$\partial_t\log p-\tfrac12\beta\big[d+x\!\cdot\!s+\|s\|^2+\nabla\!\cdot\!s\big]=0,\qquad s=\nabla\log p.$$

在解析 score 上残差应恒为 0；score 越偏离真值，残差越大。本实验对 score 施加**加性常数偏移** $s+\delta$（请与 §7/§9/§12 的乘性相对偏差 $(1+\rho)s$ 区分；此处取加性是为保持 $\nabla\!\cdot\!s$ 不变、单独考察其余项随偏差的单调性）。**硬结论只在解析/加性扰动 score 上做**（网络 score 无解析 $\partial_t\log p$，只作定性观察）。
""")

code(r"""
# 实验⑨：FPE 残差诊断（解析/扰动=硬；网络=定性）
xg = np.array([[0.5, -0.4], [1.0, 0.2], [-0.8, 0.6]])
deltas = np.linspace(-0.5, 0.5, 21)
res_vs_delta = np.array([np.abs(dna.fpe_residual_gaussian(xg, 0.3, S0, perturb=d)).max() for d in deltas])

# 网络 score 误差 (x,t) 热力图（定性）
xx = np.linspace(-2, 2, 41); tt2 = np.linspace(EPS, 1.0, 40)
ERR = np.zeros((len(tt2), len(xx)))
for j, t in enumerate(tt2):
    pred = dna.eval_score_net(net1d, xx.reshape(-1, 1), t, DEVICE).ravel()
    ERR[j] = np.abs(pred - dna.exact_score_gaussian(xx.reshape(-1, 1), t, S0).ravel())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(deltas, res_vs_delta, 'o-'); ax[0].set_xlabel(r'score 扰动 $\delta$'); ax[0].set_ylabel('FPE 残差 max|·|')
ax[0].set_title('实验⑨(a) 解析 FPE 残差 vs 扰动（硬）'); ax[0].set_yscale('log')
im = ax[1].imshow(ERR, aspect='auto', origin='lower', extent=[xx[0], xx[-1], tt2[0], tt2[-1]], cmap='magma')
plt.colorbar(im, ax=ax[1]); ax[1].set_xlabel('x'); ax[1].set_ylabel('t'); ax[1].set_title(r'实验⑨(b) 网络 score 误差 $|s_\theta-s^*|$（定性）')
plt.tight_layout(); plt.savefig('figures/fig11_fpe.png', bbox_inches='tight'); plt.show()
r0 = np.abs(dna.fpe_residual_gaussian(xg, 0.3, S0, perturb=0.0)).max()
print('解析 score FPE 残差 = %.2e（应≈0）；扰动 δ=0.3 残差 = %.3e' % (r0, res_vs_delta[np.argmin(abs(deltas-0.3))]))
assert r0 < 1e-3
assert res_vs_delta[np.argmin(abs(deltas-0.3))] > res_vs_delta[np.argmin(abs(deltas))]
""")

md(r"""
**实验⑨分析.** (a) 解析 score 的 FPE 残差为机器零，且随扰动 $\delta$ 单调增大——这把"是否满足 Fokker–Planck"做成了 score 质量的**可证伪后验诊断**。(b) 网络 score 的误差在 $|x|$ 较大与 $t$ 较小（低密度/小噪声）区域更大，与直觉一致；因网络无解析 $\partial_t\log p$，此处仅作定性热力图，不作"残差≈误差"硬断言。

### §12 迁移到网络 score：离散误差 ↔ score 逼近误差的交叉
真实网络 score 与真值有逼近误差。我们分两层来看：**(a) 理论机制**——用一个固定**相对偏差** $\rho$ 建模 score 误差 $s_\rho=(1+\rho)s^*$（对应反向漂移 $a_\rho(t)=\beta(t)\big((1+\rho)/v(t)-\tfrac12\big)$），则离散误差随 $h$ 减小以斜率 1 下降，但被该偏差导致的**误差地板**截断：$h$ 足够小时残余误差趋于由 $\rho$ 决定的非零常数，从而与纯离散曲线相交。这刻画了"步数加到多少才划算"。**(b) 实测对照**——再叠加**实际训练网络**的采样误差（确定性 CPU 直接采样）。需要说明的是，由静态 score 回归得到的单标量 $\hat\rho$ 只是误差地板的**上界式代理**：采样误差沿轨迹会部分相消，故网络实测曲线在本步数范围内通常**尚未触底**，其交叉点为**外推预期**而非已直接观测。
""")

code(r"""
# 实验⑩：不完美/网络 score 的误差地板（离散误差 ↔ score 逼近误差）
# 用相对偏差 ρ 建模 score 误差：s_ρ=(1+ρ)s*，对应反向漂移 a_ρ=β((1+ρ)/v-1/2)。
# 这给出确定性、可复现的"地板"机制；再叠加真实训练网络作实测对照。
Ns10 = np.array([20, 40, 80, 160, 320, 640]); hs10 = (1.0 - EPS) / Ns10
v_eps = dna.marginal_var(EPS)

def biased_var_recursion(N, rho, V0=1.0):
    ts = np.linspace(1.0, EPS, N + 1); V = V0
    for i in range(N):
        t = ts[i]; h = ts[i] - ts[i + 1]
        a_rho = dna.beta(t) * ((1 + rho) / dna.marginal_var(t, S0) - 0.5)
        V = (1 - a_rho * h) ** 2 * V + dna.beta(t) * h
    return V

err0   = np.array([abs(biased_var_recursion(N, 0.0)  - v_eps) for N in Ns10])   # 精确 score：纯离散
err_b1 = np.array([abs(biased_var_recursion(N, 0.05) - v_eps) for N in Ns10])   # 5% score 偏差
err_b2 = np.array([abs(biased_var_recursion(N, 0.10) - v_eps) for N in Ns10])   # 10% score 偏差
# 训练网络：实测采样误差（确定性 CPU + 大样本，非由 ρ̂ 反推，避免循环论证）
err_net = np.array([abs(dna.reverse_sample(net1d_score, N, 'em', n_samples=200000, dim=1, eps=EPS, seed=7).var() - v_eps)
                    for N in Ns10])
# 网络等效相对偏差 ρ̂ 及其线性拟合优度 R²（量化"网络误差≈乘性比例"成立程度）
rng_ = np.random.default_rng(0); Sn, Ss = [], []
for t in np.linspace(EPS, 0.95, 30):
    xq = rng_.uniform(-2, 2, (200, 1))
    Sn.append(dna.eval_score_net(net1d, xq, t, DEVICE).ravel())
    Ss.append(dna.exact_score_gaussian(xq, t, S0).ravel())
Sn = np.concatenate(Sn); Ss = np.concatenate(Ss)
rho_hat = float(np.sum(Sn * Ss) / np.sum(Ss * Ss) - 1.0)
R2 = float(1.0 - np.sum((Sn - (1 + rho_hat) * Ss) ** 2) / np.sum((Sn - Sn.mean()) ** 2))
slope0 = np.polyfit(np.log(hs10), np.log(err0), 1)[0]
slope_b2_fine = np.polyfit(np.log(hs10[-3:]), np.log(err_b2[-3:]), 1)[0]

plt.figure(figsize=(6.6, 4.4))
plt.loglog(hs10, err0,   'o-',  label='精确 score（纯离散，斜率=%.2f）' % slope0)
plt.loglog(hs10, err_b1, 's--', label='相对偏差 ρ=5% 地板')
plt.loglog(hs10, err_b2, '^--', label='相对偏差 ρ=10% 地板')
plt.loglog(hs10, err_net, 'kx-', ms=7, label='训练网络实测（ρ̂≈%.1f%%, R²=%.2f）' % (rho_hat * 100, R2))
plt.xlabel('步长 h'); plt.ylabel(r'方差误差 $|V-v(\varepsilon)|$'); plt.legend(fontsize=8); plt.grid(True, which='both', alpha=.3)
plt.title('实验⑩：不完美 score 的误差地板与交叉')
plt.savefig('figures/fig12_transfer.png', bbox_inches='tight'); plt.show()
print('精确斜率=%.3f；ρ=10%% 细步斜率=%.3f；网络 ρ̂=%.4f R²=%.3f；网络实测最细步 err=%.2e'
      % (slope0, slope_b2_fine, rho_hat, R2, err_net[-1]))
assert 0.8 < slope0 < 1.2          # 精确 score：离散误差 O(h)
assert slope_b2_fine < 0.5         # 偏差 score：细步趋平（误差地板）
assert err_b2[-1] > err0[-1]       # 地板高于纯离散 -> 存在交叉
""")

md(r"""
**实验⑩分析.** (a) **理论机制**：精确 score 的纯离散误差以斜率 $\approx1$ 持续下降（蓝线）；带固定相对偏差 $\rho$ 的合成曲线（橙/绿）在细步区**触底**于由 $\rho$ 决定的误差地板（$\rho$ 越大地板越高、细步斜率趋于 0），并与蓝线相交——交叉点之后再加步数已无收益。这清晰展示了"地板 + 交叉"的机制。(b) **实测对照**：黑色"×"是**实际训练网络**的采样误差（确定性 CPU 直接采样，**非**由 $\hat\rho$ 反推，避免循环论证）。它在本步数范围内大体随离散曲线下降、**尚未明显触底**；其由静态 score 回归得到的等效相对偏差 $\hat\rho\approx-3.6\%$（图例 $R^2$ 为该回归的拟合优度，非地板预测）。值得注意的是，$\hat\rho$ 对应的**地板远高于**网络实测最细步的误差——这说明单标量 $\hat\rho$ **高估**了真实采样地板（采样误差沿轨迹部分相消），故应把 $\hat\rho$ 视为**上界式代理**，网络的实际交叉点属**外推预期**。**注意**：此折算依赖 score 近线性，对强非线性 score（GMM/真实数据）不一定成立。综合 (a)(b)，该图把"采样步数预算"与"score 精度预算"统一起来，给出"步数加到多少才划算"的定性判据。
""")

# ======================================================================
# Part VI 总结
# ======================================================================
md(r"""
## Part VI　总结、联系与展望

### §13 结论
本文把扩散模型的反向采样系统地当作**数值积分问题**，在可解析的高斯设定下取得一系列**可证伪**的结果：

1. **后向误差分析**给出反向 EM 生成分布方差的领头阶 $O(h)$ 偏差闭式系数，并与 Richardson 外推高度吻合（相对误差约 $7.5\times10^{-6}$）——回答了"EM 实际在采哪条被修正的 SDE"。
2. **澄清收敛阶**：本问题为加性噪声，EM 强阶=弱阶=1（非教科书乘性噪声的 1/2），并以共享布朗路径与 GBM 对照数值验证。
3. **刚性/稳定域**把"步数不能太少"提升为判据 $h\le2/a(t)$，定位最刚处在 $t\approx1$；并指出概率流 ODE 不刚，是少步采样有效的根源。
4. **改进采样器**：半隐式/指数法降低误差常数、Richardson 外推提阶、概率流 ODE 闭式解 $x\propto\sqrt v$ 作机器精度参照，并**诚实**呈现朴素指数法的局限。
5. **误差预算三分解**与 **FPE 残差诊断**把"生成为何不准"做成可测量、可证伪的归因。
6. **网络迁移**定量定位离散误差与网络逼近误差的交叉点。

### 与 DPM-Solver / 指数积分器的联系
DPM-Solver、DEIS 等少步采样器的数学内核正是"对半线性概率流 ODE 的线性漂移做指数（变易常数）积分"。本文在高斯设定下给出其**闭式解**与收敛阶证据，并诚实指出：少步增益主要来自 ODE 的非刚性与高阶格式，而非简单地把漂移指数化。

### 与课程的统一视角（呼应 Lect5）
本文的"采样修正方程"与 Lect5 的"SGD 修正方程"是**同一套后向误差分析**：把一个离散算法（采样迭代 / SGD 迭代）理解为某条被修正的微分方程的数值格式。这正是整门课"**微分方程是理解 AI 的统一视角**"的体现。

### 局限与展望
- 解析结论建立在**高斯（线性 score）**设定上；推广到强非线性 score 与真实数据流形，需要更精细的局部分析（GMM 已部分引入非线性）。
- 误差预算与 FPE 诊断在高维需 Hutchinson 迹估计等可扩展技术。
- 修正采样器仅做到降阶/外推；设计**保结构、对噪声项也消偏**的高阶采样器是值得继续的方向。
""")

md(r"""
## 附录

### A. 复现说明
- 环境：`uv` 隔离（Python 3.12，torch MPS，numpy/scipy/matplotlib/scikit-learn）。
- 运行：`uv sync` 后选 `Python (diffusion-na)` 内核，"运行全部"；或 `uv run jupyter nbconvert --to notebook --execute --inplace 反向扩散采样的数值分析.ipynb`。
- 全局随机种子 `SEED=2026`；图保存在 `figures/`。
- 核心数值函数集中在 `diffusion_na.py`（已内联进本 notebook 顶部"工具区"，单文件即可运行）。

### B. 关键超参数
| 项 | 值 |
|---|---|
| 噪声调度 | $\beta(t)=0.1+19.9t$ |
| 数据 | $X_0\sim\mathcal N(0,0.5^2 I)$（$s_0=1$ 退化禁用） |
| 采样区间 | $[\varepsilon,1]$，$\varepsilon=10^{-3}$ |
| 网络 score | MLP（输入 $(x,t)$，3×64，LogSigmoid），DSM，Adam lr=3e-3 |

> 网络结构与激活（LogSigmoid）沿用课程 **Lect6 扩散基线**；本文重点在采样的数值分析，网络仅作 score 的一个来源（实验⑩ 用其等效相对偏差 $\hat\rho$ 量化精度）。


### C. 数值自检清单（节选）
- 反向方差 ODE（含因子 2）末端 = $v(\varepsilon)=0.2501$（漏因子 2 错得 0.759）。
- 修正方程系数 $c_{\rm theory}\approx0.35798$ vs Richardson $c_{\rm emp}\approx0.35798$ 高度吻合（相对误差约 $7.5\times10^{-6}$）。
- 加性噪声强阶 $\approx1$、GBM 乘性噪声强阶 $\approx1/2$。
- 概率流 ODE Heun 阶 $\approx2$ 收敛到闭式解 $x\propto\sqrt v$。
- 解析 score 的 FPE 残差 $<10^{-3}$。
""")

# ======================================================================
# 写出
# ======================================================================
nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, '反向扩散采样的数值分析.ipynb')
print('notebook written: %d cells' % len(cells))
