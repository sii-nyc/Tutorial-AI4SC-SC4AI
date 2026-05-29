# 设计文档：反向扩散采样的数值分析（SC4AI 开放课题）

- **作者**：牛昱琛（学号 253108120111），单人完成
- **课程**：《人工智能数理基础（高级）》开放课题，方向 SC4AI
- **日期**：2026-05-29（截止 2026-05-31）
- **交付**：一个自包含中文 Jupyter Notebook（理论 + 代码 + 实验 + 分析）
- **定位**：占比高（用户告知 50%）、按 1–3 人协作规模要求 ⇒ 内容须**非常丰富、系统、严谨**

---

## 1. 目标与核心主张（thesis）

把扩散模型的**反向采样**当作"**数值求解一个反向时间 SDE/概率流 ODE**"的问题，用**数值分析**的工具箱系统地回答三个问题：

1. **采样器实际在采哪条分布？**（后向误差分析 / 修正方程：离散格式引入的领头阶偏差）
2. **采样为什么会不准、不稳？**（收敛阶、刚性/稳定性、误差预算三分解）
3. **如何可验证地改进？**（降阶修正采样器、Richardson 外推、指数积分器/少步采样、Fokker–Planck 残差诊断）

**关键可解析设定**：当数据为高斯/各向同性 $X_0\sim\mathcal N(\mu_0,\Sigma_0)$ 时，VP 前向过程的边缘仍是高斯、**精确 score 线性**，反向过程是**线性 SDE**，其矩满足**闭式确定性递推**。于是每一个理论命题都有**解析真值**作硬基准，结论**可证伪**（falsifiable）。再推广到高斯混合（闭式 score）与训练得到的网络 score，说明理论的适用边界。

**SC4AI 立意**：用科学计算（数值分析）理解并改进 AI 方法（扩散模型）。并与课程内部呼应——**Lect5 的 SGD 修正方程**与本文的**采样 EM 修正方程**是同一个"后向误差分析"工具；**Lect6 的扩散 notebook**是本文的实验基线。体现整门课"**微分方程是理解 AI 的统一视角**"的主题。

## 2. 评分轴映射

| 评分轴 | 权重 | 本方案如何拿分 |
|---|---|---|
| 选题价值 | 20% | 把热门扩散模型的采样问题落到数值分析第一性原理；贯通 Lect5/Lect6；问题真实（少步采样、稳定性是工程痛点） |
| 方法创新 | 20% | 解析推出 EM 生成分布偏差的闭式系数并设计降阶修正；误差三分解；指数积分器的诚实评测（含负结果）；FPE 残差作后验诊断 |
| 报告完整 | 40% | 多 Part 结构、每个命题配解析真值的可证伪实验、瀑布图/收敛阶图/Pareto/相图、局限与展望、附录完整推导 |
| 代码实现 | 20% | uv 隔离环境、MPS 适配、固定种子、自包含可复跑、模块化函数 + 单元式数值校验 |

## 3. 理论内容（会写成带 LaTeX 推导的 markdown）

**记号与时间约定（全程严格统一；经数值核验）**：VP 前向 SDE $dX=-\tfrac12\beta(t)X\,dt+\sqrt{\beta(t)}\,dW$，$\beta(t)=\beta_{\min}+(\beta_{\max}-\beta_{\min})t$（$\beta_{\min}=0.1,\beta_{\max}=20$），$B(t)=\int_0^t\beta=0.1t+9.95t^2$。反向漂移 $b_{\mathrm{rev}}(t,x)=-\tfrac12\beta(t)x-\beta(t)\nabla\log p(x,t)=a(t)x$，其中 $a(t)=\beta(t)\!\left(1/v(t)-\tfrac12\right)>0$。

| 名称 | 记号 | 方向 | 步长 | 一步公式 |
|---|---|---|---|---|
| 前向时间 | $t$ | $0\!\to\!1$ | $dt>0$ | VP 前向 SDE |
| 反向采样网格 | $t_n$ | $1\!\to\!\varepsilon$ | $h=t_n-t_{n+1}>0$ | EM: $x_{n+1}=(1-a(t_n)h)x_n+\sqrt{\beta(t_n)h}\,\zeta_n$ |
| 反向时间 | $\tau=1-t$ | $0\!\to\!1-\varepsilon$ | $d\tau>0$ | $dY=-a(1-\tau)Y\,d\tau+\sqrt{\beta(1-\tau)}\,d\tilde W$ |

1. **转移核与边缘**：$p(x_t\mid x_0)=\mathcal N\!\big(x_0e^{-B(t)/2},(1-e^{-B(t)})I\big)$。各向同性数据 $X_0\sim\mathcal N(0,s_0^2I)$ ⇒ $p(x_t)=\mathcal N(0,v(t)I)$，$v(t)=1+(s_0^2-1)e^{-B(t)}$，**精确 score** $\nabla\log p(x_t)=-x_t/v(t)$（线性）。高斯混合数据：score 为高斯分量加权和（闭式）。
2. **Anderson 反向 SDE**：$dY=\big[-\tfrac12\beta Y-\beta\nabla\log p\big]dt+\sqrt{\beta}\,d\bar W$（$t:1\!\to\!0$）。代入线性 score ⇒ 反向漂移 $=a(t)Y$ ⇒ **反向过程为线性 SDE**。
3. **矩的精确演化（含因子 2，已数值核验）**：由 Itô，对线性 SDE $dY=\kappa Y\,d\tau+g\,dW$ 有 $\dot V=2\kappa V+g^2$。本处 $\kappa=-a(1-\tau),g^2=\beta(1-\tau)$ ⇒ **$dV/d\tau=-2a(1-\tau)V+\beta(1-\tau)$**，闭式可解；$s_0=0.5,\varepsilon=10^{-3}$ 时末端 $\to v(\varepsilon)=0.25008$（核验通过；漏因子 2 会错得到 $0.759$）。
4. **EM 的矩封闭递推（正步长 $h$）**：$Y_{n+1}=(1-a(t_n)h)Y_n+\sqrt{\beta(t_n)h}\,\zeta_n$ ⇒ **确定性递推** $V_{n+1}=(1-a(t_n)h)^2V_n+\beta(t_n)h$（无 MC 噪声，微秒级）。展开 $(1-ah)^2=1-2ah+a^2h^2$，领头阶即上面的 $-2aV+\beta$，离散多出的 $a^2h^2V$ 是漂移-噪声耦合项。
5. **后向误差分析 / 修正方程**：把递推写成 $V_{n+1}=V_n+hF+h^2G$，$F=-2aV+\beta,\,G=a^2V$。修正方程 $V'=F+hH+O(h^2)$，匹配一步 Taylor 得 $H=G-\tfrac12(F_\tau+F_V F)=a^2V-\tfrac12V''$。领头阶全局偏差 $V_N-v(\varepsilon)=c\,h+O(h^2)$，**闭式系数** $c=\int_0^T\Phi(T,s)\big[a(1-s)^2V(s)-\tfrac12V''(s)\big]ds$，$\Phi(T,s)=\exp\!\big(\int_s^T-2a(1-r)\,dr\big)$。Richardson 外推得 $c_{\mathrm{emp}}$ 作**独立交叉验证**（非替代）。
6. **收敛阶**：EM 的 SDE 弱阶（分布/矩）≈1、强阶（轨迹）≈1/2；**Heun 的高阶仅在概率流 ODE 实验中验证**（SDE 的随机 Heun/Milstein 需额外条件，不在此声称）。
7. **刚性/稳定性**：反向漂移 Jacobian $=a(t)I$，$a>0$。显式 Euler 绝对稳定域 $|1-a(t)h|\le1$ ⇒ 步长界 $0\le h\le 2/a(t)$。$t\!\to\!1$ 处 $a(1)\approx10.0$ ⇒ $h\lesssim0.2$（故最粗稳定性实验在 $N\approx5$ 量级，而非只取大 $N$）。诚实说明 $s_0^2>0$ 时 $1/v(t)$ 在 $t\to0$ **有界**，刚性主要来自 $\beta(t)$ 增大；理想数据流形（$s_0\to0$）才出现 $t\to0$ 奇异。
8. **概率流 ODE 与指数积分器**：前向 $f_{\mathrm{pf}}(t,x)=-\tfrac12\beta x-\tfrac12\beta\nabla\log p=\tfrac12\beta(1/v-1)x$；反向 Euler 步 $x_{n+1}=x_n-h\,f_{\mathrm{pf}}(t_n,x_n)$。**高斯下有闭式精确解** $x(t)=\sqrt{v(t)/v(1)}\,x(1)$（已核验，作为机器精度参照与"理想指数积分器"）。冻结系数指数/半隐式法**显著降低线性漂移引入的误差常数，但对时变 $a,\beta$ 整体仍为一阶**——不声称"近机器精度"；闭式映射才是近机器精度上界。诚实呈现：朴素 frozen-score 指数法在含 score 项时不一定胜过 Euler。
9. **误差预算三分解**：总采样误差 = 分数估计误差 + 时间离散误差 + 先验失配误差（$\mathrm{KL}(p(\cdot,1)\,\|\,\mathcal N(0,I))$ 闭式），算子分裂式归因。
10. **Fokker–Planck 残差**：对解析高斯/GMM 的 log-FPE 恒等式 $\partial_t\log p-\tfrac12\beta\big[d+x\!\cdot\!s+\|s\|^2+\nabla\!\cdot\!s\big]=0$（$s=\nabla\log p$，已推导核验）。**硬结论仅在解析 score（残差≈0）与人为扰动 score（残差随扰动增大）上做**；网络 score 没有解析 $\partial_t\log p$，故**只作定性热力图/探索**，不做"残差≈误差"硬断言。

> **可解析设定的边界说明**：高斯 $s_0=0.5$ 仅用于**隔离时间离散误差**并提供解析真值（$s_0=1$ 退化，禁用）；真实低维流形扩散的难点主要来自 **score 非线性**与 **$t\to0$ 的小噪声刚性区**，GMM 实验部分引入非线性以贴近真实。

## 4. Notebook 结构（中文；markdown 理论 + 可运行代码 + 图 + 分析）

**Part 0 引言**：背景、扩散即 SDE、采样即数值积分、SC4AI 主题、与 Lect5/Lect6 呼应、**贡献清单**、阅读导航。

**Part I 理论基础**
- §1 扩散模型与前向/反向 SDE、score、DSM（自洽回顾 + 推导）
- §2 可解析设定：高斯/各向同性闭式边缘与线性 score；高斯混合闭式 score；"用解析 score 隔离误差"的方法论

**Part II 采样即数值积分：误差与收敛**
- §3 反向 EM 的矩封闭递推
- §4 后向误差分析与修正方程（领头阶偏差闭式系数）
- §5 **实验①** O(Δt) 偏差律（log-log 斜率≈1）
- §6 **实验②** 修正方程系数对照 $c_{\text{theory}}$ vs $c_{\text{emp}}$（<5%）
- §7 **实验③** 弱阶≈1 蒙特卡洛验证（叠加确定性递推预测）+ 强/弱阶讨论

**Part III 稳定性与刚性**
- §8 反向漂移 Jacobian 与刚性 $L(t)$、显式 Euler 稳定域、步长界
- §9 **实验④** 刚性诊断 + 临界步长/NaN 率；均匀 $t$ / 均匀 log-SNR / 自适应网格对比；时间裁剪 $t_{\text{end}}$ 扫描

**Part IV 改进的采样器**
- §10 降阶修正/半隐式采样器（漂移指数化 + 噪声项一阶修正）；Richardson 外推
- §11 **实验⑤** 修正采样器 Bures-W2 降阶验证
- §12 概率流 ODE 与指数积分器（推导）；**实验⑥** 收敛阶 + NFE-质量 Pareto（Swiss-Roll，energy distance/sliced-W）+ 诚实负结果
- §13 **实验⑦** 随机 vs 确定性采样器偏差-方差权衡（受控分数扰动，噪声自校正曲线）

**Part V 误差预算与 score 诊断**
- §14 **实验⑧** 总采样误差三分解瀑布图（高斯混合 toy）
- §15 **实验⑨** Fokker–Planck 残差作后验诊断（解析 score 标定残差=误差代理；逐点散点 + 相关系数）
- §16 **实验⑩** 迁移到网络 score：用基线 DSM 训练小 score 网络（MPS，1–3 min），复现弱误差曲线，标出**离散误差 vs 网络逼近误差交叉点**

**Part VI 总结**
- §17 与 DPM-Solver/指数积分器文献的联系（诚实）、与 Lect5 修正方程的统一视角、局限与展望
- **附录**：完整公式推导、复现说明、超参与种子表、环境清单

## 5. 实验—指标—解析参照（每个实验都有硬基准，可证伪）

| 实验 | 输出 | 指标 | 解析参照 | 断言强度 |
|---|---|---|---|---|
| ① O(Δt) 偏差律 | log-log 图 | 方差误差 vs h（斜率≈1） | VP 闭式矩 $v(\varepsilon)$ | 硬 |
| ② 系数对照 | 表/图 | $c_{\text{theory}}$ vs $c_{\text{emp}}$（<10%） | 高精度反向方差 ODE + Richardson 外推 | 硬 |
| ③ 弱阶 | log-log 图 | $E\|Y_0\|^2$ 弱误差 vs h（斜率≈1） | 确定性递推预测 | 硬 |
| ④ 刚性/稳定 | 曲线+热力 | $L(t)=a(t)$；临界步长 $2/a$、NaN 率 | 闭式 Jacobian/稳定域 | 硬 |
| ⑤ 修正采样器 | log-log 图 | Bures-W2 vs h（误差常数↓） | 高斯间闭式 Bures-W2 | 硬 |
| ⑥ 指数积分器/PF ODE | 收敛阶图+Pareto | 末态 L2；NFE-质量 | **闭式 PF 映射 $x\propto\sqrt v$**（机器精度参照）/ Swiss-Roll 真值 | 硬(线性高斯) |
| ⑦ 偏差-方差 SDE↔ODE | 曲线+相图 | W2/矩 bias²/var；鲁棒性 vs 扰动 ε | 解析末态矩 | **探索性**(无硬断言) |
| ⑧ 误差三分解 | 瀑布图 | W2/矩分段 | 高斯混合闭式矩 | 硬(三段和≈总) |
| ⑨ FPE 残差 | 散点+热力 | 解析残差≈0、扰动单调；网络相关系数 | 闭式 $s^\*$ | 硬(解析/扰动) + **定性**(网络) |
| ⑩ 网络 score 迁移 | 交叉点图 | 矩误差 vs h | 闭式递推 + 训练网络 | 硬(存在拐点) |

## 6. 代码架构与环境

- **环境（uv 隔离）**：新文件夹内 `uv` 建 venv（Python 3.12）；依赖 `torch, numpy, scipy, matplotlib, scikit-learn, jupyter, ipykernel`；高斯 W2 用闭式 Bures（核心**不**依赖 POT）。
- **设备**：`device = 'mps' if torch.backends.mps.is_available() else 'cpu'`；核心解析实验是 NumPy 标量/向量，秒级；仅 §16 训练上 MPS；FPE 二阶导用 float64 CPU（小规模）。
- **度量**：高斯用闭式 Bures-W2；2-D 点云主用 **sliced-Wasserstein**，energy distance 仅作辅助且**子采样 $n\le512$ 或分块**（避免 $O(n^2)$ 内存）。
- **开发结构 vs 交付结构**：开发期核心函数集中在经 pytest 验证的 `diffusion_na.py`（`beta,B,v,exact_score_gaussian,exact_score_gmm,reverse_drift_coeff,reverse_variance_exact,em_variance_recursion,em_sample,semiimplicit_variance_recursion,pf_ode_sample,pf_ode_exact,bures_w2,sliced_wasserstein,energy_distance,moment_error,stiffness,critical_dt,fpe_residual_gaussian,train_score_net_*,eval_score_net,bias_coeff_theory,bias_coeff_empirical`）。**最终交付默认把该库内联进 notebook 顶部一个"工具区"cell**，使评卷老师单文件即可跑通；`diffusion_na.py` 与 `tests/` 作为开发/验证脚手架附带。
- **可复现**：全局 `SEED=2026`；每个随机实验固定/多种子平均；图保存到 `figures/`；`README.md` 写运行步骤。
- **正确性自检**：每个解析公式都配数值校验（递推 vs MC 模拟、`reverse_variance_exact`(含因子2) vs 递推 vs $v(\varepsilon)$、$c_{\text{theory}}$ vs Richardson、解析 score 上 FPE 残差≈0、闭式 PF 映射 vs 高分辨 Heun、稳定域阈值 vs 实测 NaN）。

## 7. 文件与交付

新文件夹（仓库根，**英文名**，不动课程原始文件）：
```
Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/
  反向扩散采样的数值分析.ipynb      # 主交付（中文，默认内联 diffusion_na 库，单文件可跑）
  diffusion_na.py                    # 开发期核心库（经 pytest 验证；交付时其代码已内联进 notebook）
  tests/test_diffusion_na.py         # 数值自检（开发脚手架）
  pyproject.toml                     # uv 依赖
  README.md                          # 运行/复现说明（含 uv 步骤）
  figures/                           # 运行自动生成的图
  .venv/                             # uv 环境（git 忽略）
```
Notebook 抬头含：标题、作者 牛昱琛 / 253108120111、单人完成说明、日期、目录。

## 8. 范围分层与时间/风险计划（2 天单人；Claude 实现，用户审阅）

- **必成核心（Tier-0，第 1 天）**：§0–§7（理论 + 实验①②③）。全程解析、免训练、有铁基准——保证一份完整可交付报告。
- **重点加分（Tier-1，第 2 天上午）**：§8–§13（刚性、修正采样器、指数积分器、偏差-方差）。
- **丰富加分（Tier-2，第 2 天下午）**：§14–§16（误差三分解、FPE 诊断、网络 score 迁移）+ §17 + 附录 + 全文润色。
- **砍单原则**：Tier 顺序砍；任何延伸被砍都不破坏报告完整性。**保持丰富体量（用户要求，撑 50%/团队规模），但把易翻车的硬断言降级为探索性**——具体：实验⑦"随机自校正"改为探索性（报告哪些扰动下出现/不出现，不硬断言）、实验⑨网络 score 的 FPE 仅定性（解析/扰动 score 才硬断言残差）。
- **风险与缓解（已据外部审查修订）**：① 时间符号——全程用统一约定表，反向方差 ODE 含**因子 2**（已数值核验）；② 修正系数——按 $H=a^2V-\tfrac12V''$、$\Phi=\exp(\int-2a)$ 正确推导 + Richardson 独立交叉验证；③ 指数积分器——"近机器精度"仅指**闭式 PF 映射** $x\propto\sqrt v$，冻结系数法只声称降低误差常数并诚实呈现负结果；④ FPE——网络 score 无解析 $\partial_t\log p$，只定性。
- **实现方式**：用多代理 workflow 并行实现相互独立的章节并对每个数值/理论命题做对抗式验证（build → verify），再合并进单一 notebook。

## 9. 待确认/默认决定

- 文件夹英文名：`Project-SC4AI-Diffusion-Sampling-NumericalAnalysis`（默认采用）。
- 保留 §16 网络 score 迁移（训练小网络）——默认**保留**（更完整、更丰富）。
- 度量：高斯用闭式 Bures-W2；混合分布用矩误差为主、经验 W2 为辅（默认，避免 Sinkhorn 噪声掩盖趋势）。
