# 反向扩散采样数值分析设计审查

- 审查对象：
  - `docs/superpowers/specs/2026-05-29-reverse-diffusion-sampling-numerical-analysis-design.md`
  - `docs/superpowers/plans/2026-05-29-reverse-diffusion-sampling-numerical-analysis.md`
- 参考基线：`Lect6-SC4AI-扩散模型及其随机微分方程/1. JupyterNotebook-Diffusion-Model/扩散模型与反向随机微分方程.ipynb`
- 审查日期：2026-05-29
- 结论摘要：选题价值高，和 SC4AI/数值分析课程主题匹配；但当前两份文档存在若干会导致实现和实验结论失败的核心数学错误，尤其是反向时间方差 ODE、EM 符号、修正方程系数和概率流/FPE 残差口径。必须先统一时间符号，再开展实现。

## 严重错误

### 1. 反向时间方差 ODE 少了因子 2，符号口径混乱

位置：
- spec 第 38 行：反向方差写为 `dot V = 2aV + beta`
- plan 第 277-288 行：文字写 `dV/dtau=2 a_tilde V + beta`，代码实现为 `return [-a*V[0] + beta(t)]`

问题：
plan 的代码少了因子 `2`，且 spec 的 `+2aV` 与反向时间正步长不一致。

正确推导：

前向 VP SDE 为
```text
dX_t = -1/2 beta(t) X_t dt + sqrt(beta(t)) dW_t.
```

对 `X_0 ~ N(0, s0^2 I)`，
```text
v(t) = 1 + (s0^2 - 1) exp(-B(t)),
score(x,t) = -x / v(t),
B(t) = 0.1 t + 9.95 t^2.
```

Anderson 反向 SDE 在“正向时间变量 t 但从 1 走向 0”的写法为
```text
dX = [ -1/2 beta(t) X - beta(t) score(X,t) ] dt + sqrt(beta(t)) dWbar,
```
其中 `dt < 0`。代入 score：
```text
a(t) = beta(t) (1/v(t) - 1/2).
```

若改用反向时间 `tau = 1 - t`，并令 `Y_tau = X_{1-tau}`，则
```text
dY_tau = -a(1-tau) Y_tau dtau + sqrt(beta(1-tau)) dW_tau.
```

因此方差满足
```text
dV/dtau = -2 a(1-tau) V + beta(1-tau).
```

应将 plan 中实现改为：
```python
def reverse_variance_exact(t_end, s0=0.5, t_start=1.0, V_start=None):
    if V_start is None:
        V_start = marginal_var(t_start, s0)

    def rhs(tau, V):
        t = t_start - tau
        a = reverse_drift_coeff(t, s0)
        return [-2.0 * a * V[0] + beta(t)]

    tau_end = t_start - t_end
    sol = solve_ivp(rhs, [0.0, tau_end], [V_start],
                    rtol=1e-10, atol=1e-10)
    return float(sol.y[0, -1])
```

用错误公式会得到错误真值：在 `s0=0.5, eps=1e-3` 下，正确末端约为 `v(eps)=0.25008`；错误的 `-aV+beta` 会得到约 `0.759`，后续收敛阶和偏差系数实验都会失效。

### 2. EM 一步和方差递推在 spec 中符号错误

位置：
- spec 第 39 行：`Y_{n+1}=(1+a_n Delta t)Y_n+sqrt(beta_n Delta t) zeta_n`
- spec 第 40 行：使用 `(1+a Delta t)^2`
- plan 第 15-16 行、319-335 行：使用 `(1-a Delta t)`，这是正确方向

问题：
spec 在同一套公式中把漂移按 `+a Delta t` 写，噪声方差又按正步长 `beta Delta t` 写。这不是一个一致的时间离散。

正确写法：
统一定义反向采样正步长
```text
h_n = t_n - t_{n+1} > 0,  t_{n+1} < t_n.
```

则 EM 一步应为
```text
X_{n+1} = (1 - a(t_n) h_n) X_n + sqrt(beta(t_n) h_n) zeta_n,
```
方差递推为
```text
V_{n+1} = (1 - a(t_n) h_n)^2 V_n + beta(t_n) h_n.
```

若要沿用 Lect6 基线的写法，也可以令 `dt = t_{n+1}-t_n < 0`，但此时噪声项必须写成 `sqrt(beta(t_n) |dt|)`，不能写 `sqrt(beta dt)`。

### 3. 修正方程/后向误差分析系数 `c_theory` 推导口径错误

位置：
- spec 第 40 行：声称给出闭式系数并分离漂移与噪声贡献
- plan 第 369-398 行：`bias_coeff_theory` 的推导和代码

问题：
当前 plan 把方差 ODE 写成 `V'=-aV+beta`，这是错误的；同时把局部截断误差密度写为 `+1/2 V''`，并遗漏了 EM 方差递推中 `a^2 h^2 V` 这个离散噪声-漂移耦合项。因此 `c_theory` 即使数值上做 Richardson 校验，也不是正确理论系数。

正确推导口径：

用反向时间 `tau` 和正步长 `h`，EM 方差递推为
```text
V_{n+1}
= V_n + h F(tau_n, V_n) + h^2 G(tau_n, V_n),
```
其中
```text
F(tau,V) = -2 a(1-tau) V + beta(1-tau),
G(tau,V) = a(1-tau)^2 V.
```

设修正方程为
```text
V' = F(tau,V) + h H(tau,V) + O(h^2).
```

匹配一步 Taylor 展开：
```text
H = G - 1/2 (F_tau + F_V F).
```

沿精确解 `V(tau)`，`F_tau + F_V F = V''(tau)`，所以领头阶全局误差系数为
```text
c = int_0^T Phi(T,s) [ a(1-s)^2 V(s) - 1/2 V''(s) ] ds,
```
其中
```text
Phi(T,s) = exp( int_s^T F_V(r) dr )
         = exp( int_s^T -2 a(1-r) dr ).
```

实现建议：
- 先用正确 `reverse_variance_exact` 取 dense solution。
- 用解析或高精度差分得到 `V''`。
- 同时保留 Richardson 经验系数，但它只能作为验证，不能替代错误理论推导。

### 4. 稳定域公式符号错误

位置：
- spec 第 42 行：`|1+a(t)Delta t| <= 1`
- plan 第 508-519 行：测试使用 `|1-a dt|`

问题：
spec 符号与 plan 不一致。若 `Delta t` 是反向正步长，应为 `1-a Delta t`。

正确写法：
```text
|1 - a(t) h| <= 1,  h > 0.
```
由于本设定 `s0=0.5` 下 `a(t)>0`，稳定步长界为
```text
0 <= h <= 2/a(t).
```

在 `t=1` 附近，`a(1) approx 10.00065`，临界步长约 `0.19999`。因此最粗网格的稳定性实验可围绕 `N≈5` 的量级做，不应只用很大 `N`。

### 5. 概率流 ODE 的反向时间公式需要重写

位置：
- spec 第 43 行
- plan 第 431-455 行

问题：
plan 中 `dx/dtau = ...` 的文字公式没有写对，代码注释也写着“需取负”。这会让后续 Euler/Heun/指数积分器实现非常容易错。

正确写法：

forward-time probability-flow ODE 为
```text
dx/dt = f_pf(t,x)
      = -1/2 beta(t) x - 1/2 beta(t) score(x,t).
```

对高斯 score：
```text
f_pf(t,x) = 1/2 beta(t) (1/v(t)-1) x.
```

如果仍从 `t=1` 走到 `t=eps`，正步长 `h=t_n-t_{n+1}>0`，则
```text
x_{n+1} = x_n - h f_pf(t_n, x_n)
```
是 Euler 反向步。

若使用 `tau=1-t`，则
```text
dx/dtau = -f_pf(1-tau,x)
        = 1/2 beta(1-tau) (1 - 1/v(1-tau)) x.
```

Heun 也必须在同一变量下写清楚，避免 `t` 和 `tau` 混用。

### 6. Fokker-Planck 残差公式不完整，网络实验口径不成立

位置：
- spec 第 45 行
- plan 第 527-560 行

问题：
当前 plan 只写了 `div s + ||s||^2` 和“drift 项”，没有完整公式；并且对网络 score，一般没有 `log p` 的时间导数，不能直接计算 log-FPE 残差。

对解析高斯可用的 log-FPE 形式为：
```text
partial_t log p
- beta/2 [ d + x dot s + div s + ||s||^2 ] = 0,
```
其中 `s=grad log p`。

如果要在网络 score 上做诊断，有两种更稳妥的口径：
1. 只在 Gaussian/GMM 解析 score 上验证 FPE 残差为 0，把它作为理论诊断示例。
2. 若坚持网络 score，改为 score PDE 残差或使用已知解析分布生成训练数据，并用解析 `partial_t log p` 做校验。Swiss-roll 网络没有解析 `p_t`，不适合硬断言“残差约等于误差”。

## 风险

### 1. “指数法在纯线性高斯上近机器精度”的断言过强

位置：
- spec 第 43 行
- plan 第 411-417、431-440、635-636 行

当前 `semiimplicit_variance_recursion` 是冻结系数 OU 精确步。它只精确处理每一步常系数问题；由于 `a(t)` 和 `beta(t)` 仍随时间变，它整体通常仍是一阶，只是误差常数可能小于 EM。

若要声称“近机器精度”，必须使用：
- 高斯 PF ODE 的闭式映射，例如 `x(eps)=sqrt(v(eps)/v(1)) x(1)`；
- 或对线性时变 ODE/SDE 做高精度变易常数积分；
- 或明确限定为“常系数线性测试问题”。

建议改为：
```text
冻结系数指数法在解析线性高斯上显著降低误差常数，但不应声称对时间变系数问题达到机器精度；闭式线性高斯积分器可作为近机器精度上界。
```

### 2. `s0=0.5` 可以避免退化，但不能完全代表扩散模型难点

位置：
- plan 第 14 行
- spec 第 42 行

`s0=1` 会使 `v(t)=1`，score 与 prior 完全一致，反向过程退化，确实应禁用。`s0=0.5` 能避免退化，也让刚性在 `t≈1` 附近更明显。

但应补充一句：真实低维流形数据的困难主要来自 score 非线性与近 `t=0` 的小噪声区域，而 Gaussian `s0=0.5` 只是用于隔离时间离散误差的解析基准。

### 3. Heun 阶高于 Euler 的实验应限于 ODE，不应混到 SDE 强弱阶

位置：
- spec 第 41、69 行
- plan 第 418-424、636 行

Heun 对概率流 ODE 通常二阶；对 SDE 的随机 Heun/Milstein 需要额外条件和不同噪声处理。文档应明确：
```text
Heun 阶提升只在 probability-flow ODE 实验中验证。
EM 的 SDE 弱阶约 1，强阶约 1/2。
```

### 4. `energy_distance` 当前实现有 O(n^2) 内存风险

位置：
- plan 第 493-496 行

当前实现：
```python
((A[:,None,:]-B[None,:,:])**2).sum(-1)
```
会构造 `n x n x d` 数组。`n=2000,d=2` 约数千万浮点操作，多次实验会拖慢 notebook；更大样本会直接爆内存。

建议：
- 默认对子样本 `n<=512` 计算 energy distance；
- 或实现分块 pairwise distance；
- Swiss-roll Pareto 图主指标改为 sliced-Wasserstein，energy distance 只作辅助。

### 5. GMM score 实现只支持单点，向量场/批量实验会出错

位置：
- plan 第 245-252 行

`exact_score_gmm` 当前假设 `x` 是一维向量；若 `x` 是 `(N,d)`，`sq=np.sum((x-mu)**2, axis=-1)` 无法按 `(N,K)` 正确广播。

建议把实现改为批量版：
```python
def exact_score_gmm(x, t, means, weights, s0=0.4):
    x = np.atleast_2d(np.asarray(x, float))      # [N,d]
    means = np.asarray(means, float)             # [K,d]
    w = np.asarray(weights, float)               # [K]
    vt = _component_var(t, s0)
    mu = alpha(t) * means                        # [K,d]
    diff = mu[None, :, :] - x[:, None, :]        # [N,K,d]
    sq = np.sum(diff**2, axis=-1)                # [N,K]
    logp = np.log(w)[None, :] - 0.5 * sq / vt
    logp = logp - logp.max(axis=1, keepdims=True)
    r = np.exp(logp)
    r = r / r.sum(axis=1, keepdims=True)
    score = np.sum(r[:, :, None] * diff / vt, axis=1)
    return score
```

### 6. Bures-W2 的 `sqrtm(...).real` 稳定性一般

位置：
- plan 第 485-490 行

低维实验可以跑通，但若样本协方差接近奇异，`sqrtm` 可能给出小虚部或非 PSD 误差。建议：
- 对协方差先 `0.5*(C+C.T)` 对称化；
- 对特征值做 `clip(lambda, 0, inf)`；
- 或保留 `sqrtm` 但加入 `np.real_if_close` 和非负截断。

### 7. “SDE 噪声自校正退化更慢”不是稳健结论

位置：
- spec 第 70、91 行
- plan 第 639-641 行

受控 score 扰动下，SDE 是否更鲁棒取决于扰动方向、步长、噪声强度和指标。这个断言不应作为硬 assert。建议改为探索性实验：
```text
比较 SDE 与 ODE 在若干扰动模型下的误差曲线，并报告哪些情形出现自校正，哪些没有。
```

### 8. 两天单人范围过载

位置：
- spec 第 117-123 行
- plan Task 0-23

10 个实验、pytest 核心库、FPE、GMM、Swiss-roll Pareto、网络训练、完整中文 notebook，2 天单人风险很高。最大风险不是代码量，而是公式和实验断言无法全部可靠收敛。

建议保留评分贡献最高的主线：
1. VP 高斯解析基准。
2. EM 方差递推、O(h) 偏差律、正确 `c_theory`。
3. 稳定性/刚性。
4. PF ODE Euler vs Heun。
5. 一个小型网络 score 迁移实验。

把 GMM 误差预算、FPE、SDE vs ODE 扰动鲁棒性作为可选加分，不要放在必成路径。

## 改进建议

### 1. 在两份文档开头加入统一时间约定表

建议加入如下表格：

| 名称 | 记号 | 方向 | 步长 | 公式 |
|---|---|---|---|---|
| forward time | `t` | `0 -> 1` | `dt>0` | VP forward SDE |
| reverse sampling grid | `t_n` | `1 -> eps` | `h=t_n-t_{n+1}>0` | `x_{n+1}=x_n-h b_rev(t_n,x_n)+sqrt(beta h)z` |
| reverse time | `tau=1-t` | `0 -> 1-eps` | `dtau>0` | `dY=-b_rev(1-tau,Y)dtau+sqrt(beta)dW` |

并明确：
```text
b_rev(t,x) = [-1/2 beta(t)x - beta(t) score(x,t)] = a(t)x.
```

### 2. 把“解析高斯”作为硬主线，先修通所有数学

硬主线应包括：
- `v(t)` 和 score 的解析推导；
- 正确 Anderson reverse SDE；
- 正确 EM 方差递推；
- 正确反向时间方差 ODE；
- `O(h)` 偏差律；
- 正确修正方程系数；
- 稳定域；
- PF ODE Euler/Heun。

这些内容足以体现数值分析主题，并能保证可证伪。

### 3. 指数积分器表述降级，避免不成立断言

建议将原断言：
```text
指数法在纯线性高斯上近机器精度
```
改为：
```text
冻结系数指数/半隐式方法显著降低线性漂移导致的误差常数；对时间变系数线性高斯，闭式或高精度变易常数积分可作为近机器精度参考。
```

### 4. FPE 诊断只在解析分布上做硬结论

建议实验 9 改为：
- Gaussian/GMM 解析 score：FPE 残差数值接近 0；
- 人工扰动 score：残差随扰动增大；
- 网络 score：只作为定性热力图或探索，不做强相关硬 assert。

### 5. Notebook 最终应默认自包含

位置：
- spec 第 6、100、109 行
- plan 第 37、666 行

作业要求最终产出一个中文 Jupyter Notebook。plan 把“完全内联单文件 notebook”作为可选不够稳。建议最终交付默认内联关键库代码，`diffusion_na.py` 只作为开发阶段辅助文件。

### 6. 建议的砍单顺序

若时间不足，按如下顺序砍：
1. 先砍 SDE vs ODE 扰动鲁棒性硬断言。
2. 再砍 FPE 网络残差，只保留解析 FPE。
3. 再砍 GMM 误差预算瀑布图。
4. 保留高斯解析主线、刚性、PF Heun、网络 score 迁移。

## 必须修改的验收清单

- [ ] spec 中所有 `Delta t` 改成统一的反向正步长 `h>0`，EM 漂移因子使用 `1-a h`。
- [ ] plan 中 `reverse_variance_exact` 改为 `-2*a*V + beta`。
- [ ] `bias_coeff_theory` 按 `G - 1/2 V''` 重写，其中 `G=a^2V`，传播子用 `exp(int -2a)`。
- [ ] probability-flow ODE 分清 forward drift 和 reverse step。
- [ ] FPE 残差写出完整 log-FPE 公式；网络实验不做无法验证的硬断言。
- [ ] 指数法“近机器精度”断言改成有条件说法。
- [ ] `energy_distance` 改成分块/子采样，或主用 sliced-Wasserstein。
- [ ] GMM score 改为批量实现。
- [ ] 最终 notebook 默认自包含，而不是可选内联。

