# 交接状态（2026-05-30）

## 项目
SC4AI 开放课题《反向扩散采样的数值分析》。交付物在
`Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/`，分支 `niu-sc4ai-diffusion-project`（已 push）。
作者：牛昱琛 / 253108120111。截止 2026-05-31，提交邮箱 qsun_irl@tongji.edu.cn。

## 已完成
- 11 个实验、§1–§14、自包含中文 notebook（核心库内联），曾多次以“隐藏 diffusion_na.py 后执行”验证单文件可跑。
- 已洗去 AI 腔、过两轮审查 + 一份独立验收报告（A/优、0 blocker）。
- 实验⑪（二维高斯混合非线性 score 误差三分解）已加入并推送（commit f45f935）。
- 最后一步“README 同步 + N8 docstring”已改好（见下“未提交改动”）。

## 当前未提交改动（已在工作树，待最终验证后提交）
1. `README.md` 整体重写：实验数 10→11、去 AI 腔、修掉旧“数值完全吻合/<0.1%”为“≈7.5e-6”、timeout 900→1200。
2. `diffusion_na.py` 第 1 行 docstring：「每个解析公式」→「核心解析公式」(N8)。
3. `inline_library.py` 内联头注释同样改为「核心解析公式」。
4. `build_notebook.py` 已据上述 rebuild；notebook 已 rebuild。

## 恢复后要做（务必按序，**串行**，不要并发跑两条会 mv diffusion_na.py 的流水线——之前并发导致文件丢失）
在 `Project-SC4AI-Diffusion-Sampling-NumericalAnalysis/` 下：
```bash
# 后台任务 bprsq7f3z 是 import 版执行，先确认其结果（/tmp/e_imp.log，EXEC_IMPORT 应=0）
.venv/bin/python inline_library.py            # 内联（幂等，若已内联会 skip）
mv diffusion_na.py _hidden.py
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace "反向扩散采样的数值分析.ipynb" --ExecutePreprocessor.timeout=1200
mv _hidden.py diffusion_na.py                 # 必须恢复！
# 验证：cells/errors/figures/inlined
.venv/bin/python -c "import nbformat as nbf;nb=nbf.read('反向扩散采样的数值分析.ipynb',as_version=4);code=[c for c in nb.cells if c.cell_type=='code'];print('errors',sum(1 for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error'),'figs',sum(1 for c in code for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})),'inlined',any('_DIFFUSION_NA_SRC' in c.source for c in code))"
.venv/bin/python -m pytest -q
```
期望：errors=0、figs=13、inlined=True、23 passed。

## 提交
```bash
cd "/Users/hariseldon/Desktop/课程作业/AI-math-advanced/Tutorial-AI4SC-SC4AI"
git add -A "Project-SC4AI-Diffusion-Sampling-NumericalAnalysis"
git commit -m "docs: sync README to 11 experiments; N8 docstring wording"
git push origin niu-sc4ai-diffusion-project
```
注意中文路径带空格，命令加引号；提交前确认无 `_hidden.py`/`_chk*`/`_inspect*` 等临时文件被 staged。

## 关键事实（避免重犯）
- diffusion_na.py 若丢失：`git checkout HEAD -- diffusion_na.py` 可恢复（曾用过）。
- 关键数值：v(eps)=0.25008；c_theory=c_emp=0.35798（rel≈7.5e-6）；加性强阶≈1，GBM≈1/2；实验⑪ floor=0.017,prior=0.003,disc=0.030,score=0.032。
- 验收报告：`docs/superpowers/reviews/2026-05-30-final-audit-report.md`。

## 用户下一步意愿（已排）
1)(最有价值) 用户本人通读正文措辞，标不满意句子→精修。 2) 最终全文校对 checklist。 其余 nice-to-have 可选。
