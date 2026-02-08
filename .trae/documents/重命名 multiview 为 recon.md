## 范围（按你确认的 1/2/3）
- 仅执行：目录改名、Python 导入改名、文档/命令里的路径改名
- 不执行：额外架构调整（例如新增 recon/__init__.py、修改运行方式为 python -m ...）

## 1) 目录改名
- 将目录 `multiview/` 重命名为 `recon/`（保留目录结构与文件内容不变）

## 2) 代码引用改名（Python 导入）
- 把所有 `from multiview...` 改为 `from recon...`，覆盖：
  - ours/refine_by_sdxl.py
  - ours/refine_by_flux.py
  - ours/export.py
  - ours/evaluation.py
  - （原）multiview/trainer.py（改名后为 recon/trainer.py）
  - （原）multiview/refiner.py（改名后为 recon/refiner.py，含函数体内动态导入分支）
  - （原）multiview/refiner_uncertainty.py（改名后为 recon/refiner_uncertainty.py，含动态导入分支）

## 3) 文档/命令路径改名
- 根 README 中：`python multiview/trainer.py ...` → `python recon/trainer.py ...`
- 子 README 中：`cd multiview` → `cd recon`（文件随目录改名变为 recon/README.md）

## 4) 验证（改完后立刻做）
- 全仓库搜索确保不再出现 `multiview` 作为路径/包引用（允许自然语言描述残留，如你后续希望也可再清理）
- 运行一次 `python -m compileall` 做语法级 smoke check（不依赖外部数据集）

确认后我将开始执行以上改动。