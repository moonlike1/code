# 分类实验结果汇总

## 论文主表候选

| Method | Split | Acc (%) | Macro-F1 (%) | Balanced Acc (%) | Weighted-F1 (%) | mIoU (%) | Leak IoU (%) | Seg F1 (%) | Best Epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ClsOnly | Group 5-fold CV | 99.53 ± 0.57 | 99.46 ± 0.68 | 99.60 ± 0.51 | 99.54 ± 0.57 | 37.87 ± 5.20 | 51.19 ± 7.96 | 67.32 ± 7.58 | 183 |
| DGCNN | Group 5-fold CV | 99.76 ± 0.47 | 99.68 ± 0.65 | 99.83 ± 0.33 | 99.77 ± 0.46 | - | - | - | 213 |
| MTL-Full | Group 5-fold CV | 99.76 ± 0.47 | 99.68 ± 0.65 | 99.83 ± 0.33 | 99.77 ± 0.46 | 98.04 ± 1.44 | 98.10 ± 1.39 | 99.03 ± 0.71 | 262 |
| MTL-NoMS | Group 5-fold CV | 99.53 ± 0.58 | 99.46 ± 0.68 | 99.56 ± 0.57 | 99.53 ± 0.58 | 97.98 ± 1.44 | 98.04 ± 1.39 | 99.01 ± 0.71 | 266 |
| MTL-NoUF | Group 5-fold CV | 99.76 ± 0.47 | 99.68 ± 0.65 | 99.83 ± 0.33 | 99.77 ± 0.46 | 98.13 ± 1.41 | 98.19 ± 1.36 | 99.08 ± 0.69 | 253 |
| PointNet | Group 5-fold CV | 99.76 ± 0.47 | 99.68 ± 0.65 | 99.83 ± 0.33 | 99.77 ± 0.46 | - | - | - | 195 |
| PointNet++ | Group 5-fold CV | 99.76 ± 0.47 | 99.82 ± 0.35 | 99.83 ± 0.33 | 99.76 ± 0.47 | - | - | - | 207 |

## 逐类 F1 候选表

| Method | 清水 | 盐水 | 混合物 | 无渗漏 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClsOnly | 99.66 ± 0.76 | 99.64 ± 0.81 | 99.51 ± 1.09 | 99.05 ± 2.13 | 99.46 ± 0.68 |
| DGCNN | 99.66 ± 0.76 | 100.00 | 100.00 | 99.05 ± 2.13 | 99.68 ± 0.65 |
| MTL-Full | 99.66 ± 0.76 | 100.00 | 100.00 | 99.05 ± 2.13 | 99.68 ± 0.65 |
| MTL-NoMS | 99.35 ± 0.89 | 100.00 | 99.43 ± 1.28 | 99.05 ± 2.13 | 99.46 ± 0.68 |
| MTL-NoUF | 99.66 ± 0.76 | 100.00 | 100.00 | 99.05 ± 2.13 | 99.68 ± 0.65 |
| PointNet | 99.66 ± 0.76 | 100.00 | 100.00 | 99.05 ± 2.13 | 99.68 ± 0.65 |
| PointNet++ | 99.66 ± 0.76 | 99.64 ± 0.81 | 100.00 | 100.00 | 99.82 ± 0.35 |

## 结果来源

- CSV 明细：`/ai/0309/cloud/classification_feasibility/protocol_runs/classification_summary.csv`
- `ClsOnly` <- `/ai/0309/cloud/classification_feasibility/cv_runs_grouped_phaseB/B0_ClsOnly/experiment_summary.json`
- `DGCNN` <- `/ai/0309/cloud/classification_feasibility/cv_runs_external_grouped/DGCNN/experiment_summary.json`
- `MTL-Full` <- `/ai/0309/cloud/classification_feasibility/cv_runs_grouped_phaseB/B3_MTL_Full/experiment_summary.json`
- `MTL-NoMS` <- `/ai/0309/cloud/classification_feasibility/cv_runs_grouped_phaseC/B2_MTL_NoMS/experiment_summary.json`
- `MTL-NoUF` <- `/ai/0309/cloud/classification_feasibility/cv_runs_grouped_phaseC/B1_MTL_NoUF/experiment_summary.json`
- `PointNet` <- `/ai/0309/cloud/classification_feasibility/cv_runs_external_grouped/PointNet/experiment_summary.json`
- `PointNet++` <- `/ai/0309/cloud/classification_feasibility/cv_runs_external_grouped/PointNet2/experiment_summary.json`
