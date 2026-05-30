# 反向扩散采样的数值分析（SC4AI 开放课题）

**作者**：牛昱琛　**学号**：253108120111　（单人独立完成）
**课程**：《人工智能数理基础（高级）》开放课题 · 方向 SC4AI

把扩散模型（score-based / SDE 生成模型）的反向采样看成数值求解反向时间 SDE / 概率流 ODE，
用数值分析的方法（后向误差分析 / 修正方程、收敛阶、刚性 / 稳定性、误差预算、指数积分器、Fokker–Planck 残差）
考察其误差与稳定性。在高斯可解析设定下，反向过程的各阶矩有闭式表达，可与每个结论逐一对照。

## 主交付
- `反向扩散采样的数值分析.ipynb` —— 中文报告（理论 + 代码 + 11 个实验 + 分析），单文件即可运行（核心库已内联）。

## 文件结构
```
反向扩散采样的数值分析.ipynb   主交付 notebook（自包含：核心库已内联进顶部“工具区”cell）
diffusion_na.py               核心数值库（唯一真源；开发期被内联进 notebook）
tests/test_diffusion_na.py    数值自检（23 项）
build_notebook.py             由脚本组装 notebook（生成 import 版本，开发用）
inline_library.py             把 diffusion_na.py 内联进 notebook（生成自包含交付版本）
pyproject.toml                uv 依赖
figures/                      运行生成的图
```

> **交付说明**：助教只需 `反向扩散采样的数值分析.ipynb` 单文件即可运行（已内联核心库，并以“隐藏 diffusion_na.py 后执行”验证过自包含）。
> **开发者构建流程**：`build_notebook.py`（组装 import 版）→ `jupyter nbconvert --execute`（执行）→ `inline_library.py`（内联为自包含版）→ 再 `nbconvert --execute` 确认。`diffusion_na.py` 为唯一真源。

## 运行方式（uv 隔离环境，Apple Silicon / torch MPS）

```bash
# 1) 创建隔离环境并安装依赖
uv sync

# 2A) 交互式：注册并选择内核 "Python (diffusion-na)" 后“运行全部”
uv run python -m ipykernel install --user --name diffusion-na --display-name "Python (diffusion-na)"
uv run jupyter lab   # 或 jupyter notebook

# 2B) 命令行一键执行（约数分钟，含若干次小网络训练）
uv run jupyter nbconvert --to notebook --execute --inplace "反向扩散采样的数值分析.ipynb" \
    --ExecutePreprocessor.timeout=1200

# 3) 运行数值自检
uv run pytest -q
```

## 复现要点
- 全局随机种子 `SEED=2026`；图保存在 `figures/`。
- 解析 / 确定性实验为 NumPy 运算，与设备无关、可逐位复现；网络 score 实验默认在 CPU 上训练（便于逐位复现，代码亦支持 Apple Silicon 的 MPS）。
- 无需 CUDA。整本 notebook 端到端执行约数分钟。

## 主要结论（详见 notebook）
1. 反向 EM 的生成分布方差有闭式 $O(h)$ 偏差律；修正方程系数 $c_\text{theory}$ 与 Richardson 外推 $c_\text{emp}$ 高度吻合（相对误差约 $7.5\times10^{-6}$）。
2. 本问题为加性噪声，EM 强阶 = 弱阶 = 1（不是乘性噪声的 1/2），已用共享布朗路径与几何布朗运动对照验证。
3. 显式 Euler 的绝对稳定域给出步长上界 $h\le 2/a(t)$，最不稳定时刻在 $t\approx1$。
4. 概率流 ODE 有闭式解 $x\propto\sqrt{v}$；半隐式 / 指数法降低误差常数，Richardson 外推提阶，并比较了少步采样。
5. 总采样误差分解为分数 / 离散 / 先验三项，并用 Fokker–Planck 残差诊断 score 质量；网络 score 迁移定位离散误差与逼近误差的交叉。
6. 在二维高斯混合（非线性 score）上重做误差三分解，验证该框架不限于可解析的高斯情形。
