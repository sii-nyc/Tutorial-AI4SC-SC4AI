# 反向扩散采样的数值分析（SC4AI 开放课题）

**作者**：牛昱琛　**学号**：253108120111　（单人独立完成）
**课程**：《人工智能数理基础（高级）》开放课题 · 方向 SC4AI

把扩散模型（score-based / SDE 生成模型）的**反向采样**重新表述为"数值求解反向时间 SDE / 概率流 ODE"，
用数值分析工具（后向误差分析 / 修正方程、收敛阶、刚性 / 稳定性、误差预算、指数积分器、Fokker–Planck 残差）
系统地理解与改进它。在高斯可解析设定下每个命题都有**解析真值**作硬基准，结论可证伪。

## 主交付
- **`反向扩散采样的数值分析.ipynb`** —— 中文报告（理论 + 代码 + 10 个实验 + 分析）。**单文件可跑通**（核心库已内联）。

## 文件结构
```
反向扩散采样的数值分析.ipynb   主交付 notebook（自包含）
diffusion_na.py               核心数值库（开发期；其代码已内联进 notebook）
tests/test_diffusion_na.py    23 项 pytest 数值自检
build_notebook.py             由脚本组装 notebook（开发用）
pyproject.toml                uv 依赖
figures/                      运行生成的图
```

## 运行方式（uv 隔离环境，Apple Silicon / torch MPS）
```bash
# 1) 创建隔离环境并安装依赖
uv sync

# 2A) 交互式：注册并选择内核 "Python (diffusion-na)" 后“运行全部”
uv run python -m ipykernel install --user --name diffusion-na --display-name "Python (diffusion-na)"
uv run jupyter lab   # 或 jupyter notebook

# 2B) 命令行一键执行（约数分钟，含两次小网络训练）
uv run jupyter nbconvert --to notebook --execute --inplace "反向扩散采样的数值分析.ipynb" \
    --ExecutePreprocessor.timeout=900

# 3) 运行数值自检
uv run pytest -q
```

## 复现要点
- 全局随机种子 `SEED=2026`；图保存在 `figures/`。
- 设备自动选择：`mps`（Apple Silicon）否则 `cpu`；核心解析实验为 NumPy 标量/向量运算，秒级；
  仅"网络 score"实验做小网络 DSM 训练（分钟级）。
- 无需 CUDA。预计整本 notebook 端到端执行数分钟内完成。

## 主要结论（详见 notebook）
1. 反向 EM 的生成分布方差有闭式 $O(h)$ 偏差律；修正方程系数 $c_\text{theory}$ 与 Richardson 外推 $c_\text{emp}$ 相对误差约 6%。
2. 本问题为**加性噪声**，EM 强阶 = 弱阶 = 1（非乘性噪声的 1/2），已用共享布朗路径与几何布朗运动对照验证。
3. 显式 Euler 绝对稳定域给出步长上界 $h\le 2/a(t)$，最刚处在 $t\approx1$。
4. 概率流 ODE 闭式解 $x\propto\sqrt{v}$；半隐式 / 指数法降低误差常数、Richardson 提阶；少步采样的诚实评测。
5. 总采样误差三分解（分数 / 离散 / 先验）+ Fokker–Planck 残差诊断；网络 score 迁移定位离散↔逼近误差交叉点。
