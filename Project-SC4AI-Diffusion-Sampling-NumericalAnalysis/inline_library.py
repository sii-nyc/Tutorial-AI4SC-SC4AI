# -*- coding: utf-8 -*-
"""把 diffusion_na.py 的源码内联进 notebook 顶部，使其单文件自包含。

运行: .venv/bin/python inline_library.py
之后 notebook 中的 `import diffusion_na as dna` 解析到内联模块（无需 .py 文件）。
"""
import nbformat as nbf

with open("diffusion_na.py", "r", encoding="utf-8") as f:
    src = f.read()

nb = nbf.read("反向扩散采样的数值分析.ipynb", as_version=4)

# 若已内联则跳过
if any("_DIFFUSION_NA_SRC" in c.source for c in nb.cells if c.cell_type == "code"):
    print("already inlined; skip")
else:
    inline = (
        "# === 工具区：内联核心库 diffusion_na（使本 notebook 单文件自包含；与 diffusion_na.py 一致，经 23 项 pytest 数值自检）===\n"
        "# 说明：下方字符串即 diffusion_na.py 全部源码；执行后注册为模块 dna，后续 `import diffusion_na as dna` 直接可用。\n"
        "import types as _types, sys as _sys\n"
        "_DIFFUSION_NA_SRC = " + repr(src) + "\n"
        "_m = _types.ModuleType('diffusion_na')\n"
        "exec(compile(_DIFFUSION_NA_SRC, '<diffusion_na inlined>', 'exec'), _m.__dict__)\n"
        "_sys.modules['diffusion_na'] = _m\n"
        "print('内联核心库就绪 diffusion_na')\n"
    )
    idx = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code")
    nb.cells.insert(idx, nbf.v4.new_code_cell(inline))
    nbf.write(nb, "反向扩散采样的数值分析.ipynb")
    print("inlined at cell #%d; total cells = %d" % (idx, len(nb.cells)))
