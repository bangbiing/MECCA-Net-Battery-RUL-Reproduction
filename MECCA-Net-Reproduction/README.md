# MECCA-Net 锂离子电池剩余寿命预测复现

本仓库复现了 **MECCA-Net**（Yi et al., *Journal of Power Sources*, 2025）在锂离子电池剩余寿命（RUL）预测任务上的实验，包含 CALCE CS2 与 NASA PCoE 两个公开数据集、两个基线方法（iTransformer、ModernTCN）、逐折训练权重与全部结果文件。

- 论文原文：X. Yi, J. Xiao, G. Lei, X. Hu, Z. Feng. *A lithium-ion battery remaining useful life prediction model based on multilayer perceptron expert networks and temporal feature composition.* Journal of Power Sources, 659: 238371, 2025. DOI: [10.1016/j.jpowsour.2025.238371](https://doi.org/10.1016/j.jpowsour.2025.238371)
- 作者原始代码：[keepawakeyi/MECCA-NET-Battery-RUL-Prediction](https://github.com/keepawakeyi/MECCA-NET-Battery-RUL-Prediction)（commit `a9387db`）

## 复现结果（4 折留一电池交叉验证均值）

### CALCE CS2

| 方法 | MAE (Ah) | RMSE (Ah) | RUL 相对误差 | 参数量 |
|---|---|---|---|---|
| iTransformer | 0.2053 | 0.2686 | 34.9% | 100 354 |
| ModernTCN | 0.0742 | 0.1135 | 15.6% | 34 369 |
| **MECCA-Net** | **0.0719** | **0.1068** | 17.4% | 49 326 |

### NASA PCoE

| 方法 | MAE (Ah) | RMSE (Ah) | RUL 相对误差 | 参数量 |
|---|---|---|---|---|
| iTransformer | 0.1002 | 0.1118 | 27.8% | 100 354 |
| ModernTCN | 0.0419 | 0.0500 | 12.0% | 34 369 |
| **MECCA-Net** | **0.0374** | **0.0474** | **11.1%** | 7 421 |

## 仓库结构

```
.
├── mecca_net_calce_autoregressive.py   # 主实验：CALCE
├── mecca_net_nasa_autoregressive.py    # 主实验：NASA
├── baseline_nasa_autoregressive.py     # 官方基线：iTransformer / ModernTCN（NASA）
├── baseline_calce_autoregressive.py    # 由官方 NASA 基线迁移的 CALCE 基线
├── prepare_nasa.py                     # NASA 原始 .mat → NASA.npy
├── CALCE-iTransformer.ipynb            # 作者原始调参记录（CALCE）
├── ModernTCN-CALCE.ipynb               # 作者原始调参记录（CALCE）
├── MECCA-CALCE.ipynb                   # 作者原始调参记录（MECCA-Net）
├── data/
│   ├── CALCE/CALCE.npy                 # CALCE 预处理数据（作者提供）
│   └── NASA/NASA.npy                   # NASA 容量序列（本仓库生成）
├── runs/                               # 六组实验的 CSV 结果 + 权重
├── weights/                            # 按方法整理的最佳权重（24 个 .pth）
├── results/                            # 六组实验的 final_results.csv 汇总
├── figures/                            # 论文插图
└── 论文_MECCA-Net锂离子电池RUL预测复现_毕棒.docx
```

## 环境

- Python 3.8+
- PyTorch 2.4.1 + CUDA 11.8（CPU 也可运行，但数值与 GPU 不同，见下文）
- numpy、pandas、scipy（仅数据预处理需要）、matplotlib

```bash
pip install torch numpy pandas scipy matplotlib
```

## 复现步骤

```bash
# 1. CALCE 主实验（4 折）
python -u mecca_net_calce_autoregressive.py --data-path data/CALCE/CALCE.npy --out-dir runs/calce_w

# 2. NASA 主实验
python -u mecca_net_nasa_autoregressive.py --data-path data/NASA/NASA.npy --out-dir runs/nasa_w

# 3. NASA 基线：iTransformer / ModernTCN
python -u baseline_nasa_autoregressive.py --model itransformer --data-path data/NASA/NASA.npy --out-dir runs/nasa_itrans_w
python -u baseline_nasa_autoregressive.py --model moderntcn   --data-path data/NASA/NASA.npy --out-dir runs/nasa_mtcn_w

# 4. CALCE 基线
python -u baseline_calce_autoregressive.py --model itransformer --data-path data/CALCE/CALCE.npy --out-dir runs/calce_itrans_w
python -u baseline_calce_autoregressive.py --model moderntcn   --data-path data/CALCE/CALCE.npy --out-dir runs/calce_mtcn_w
```

每个输出目录包含：`final_results.csv`（逐折指标）、`predictions.csv`（逐循环预测）、`trials.csv`（训练过程）、`run_config.json`（超参与环境）以及 4 个 `*_best.pt`。

## 权重说明

`weights/` 下按“方法_数据集”组织，每个文件对应一折，命名与 `final_results.csv` 的 `fold_id` 一致：

| 目录 | 方法 | 折数 | 用途 |
|---|---|---|---|
| `weights/mecca_calce/` | MECCA-Net | 4 | 复现表 CALCE 行 |
| `weights/mecca_nasa/` | MECCA-Net | 4 | 复现表 NASA 行 |
| `weights/itransformer_calce/` | iTransformer | 4 | CALCE 基线 |
| `weights/moderntcn_calce/` | ModernTCN | 4 | CALCE 基线 |
| `weights/itransformer_nasa/` | iTransformer | 4 | NASA 基线 |
| `weights/moderntcn_nasa/` | ModernTCN | 4 | NASA 基线 |

加载示例：

```python
import torch
from mecca_net_calce_autoregressive import OptimizedMECCANetCALCE, TrainConfig

model = OptimizedMECCANetCALCE(TrainConfig())
model.load_state_dict(torch.load("weights/mecca_calce/CALCE_loo_00_best.pt", map_location="cpu"))
model.eval()
```

## 复现须知

1. **模型选择方式**：作者代码在**测试电池**上挑选最佳轮次（`selection_split="test"`），属于 oracle 式选择，指标偏乐观。本仓库保留该默认设置以保证“复现”，未作改动。
2. **CPU 与 GPU 差异**：相同配置下，GPU 上重复运行可逐位复现；CPU 上结果差异明显（CS2_35 的 MAE 为 0.0356 对 0.0937）。换设备复现时请注明硬件。
3. **B0007 未达阈值**：NASA B0007 的最低容量为 1.400 Ah，恰未跌破 0.7×2.0=1.4 Ah 的阈值，其 EOL 被置于序列末尾，RUL 误差的参考价值有限。
4. **CALCE 基线来源**：仓库原有无 CALCE 基线脚本，`baseline_calce_autoregressive.py` 由官方 NASA 基线迁移而来（仅替换数据读取、额定容量与窗口长度），模型与超参保持官方默认。作者 notebook 中的 CALCE 基线流程与脚本版不同，数值不能直接对等。

## 引用

```bibtex
@article{yi2025mecca,
  title   = {A lithium-ion battery remaining useful life prediction model based on multilayer perceptron expert networks and temporal feature composition},
  author  = {Yi, Xuan and Xiao, Jianmao and Lei, Gang and Hu, Xin and Feng, Zhiyong},
  journal = {Journal of Power Sources},
  volume  = {659},
  pages   = {238371},
  year    = {2025},
  doi     = {10.1016/j.jpowsour.2025.238371}
}
```

## 许可

本仓库中由作者提供的代码与数据版权归原作者所有，仅用于课程复现与学术研究；其余复现脚本、权重与文档采用 MIT 许可。
