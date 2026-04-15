# Leakage Detection 对比模型结果汇总（v2 验证集）

更新时间：2026-04-04  
验证集：`/ai/0309/cloud/full_area_val_v2.h5`

说明：主表默认使用全验证集 `global aggregation` 口径。`PointTransformerV2` 使用训练日志中的 best 验证结果；`KPConv` 使用 `val_IoUs.txt` 的最佳记录；`PointNet++` 在当前环境下无法稳定复评，因此暂不填数。

## 主表

| Method | Input Features | Run / Result Time | mIoU (%) | Leak IoU (%) | Precision (%) | Recall (%) | F1 (%) | Acc (%) | Δ mIoU vs Ours | Δ Leak IoU vs Ours |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointTransformerV2 | `XYZ + pseudo-color(intensity x3)` | `2026-04-04 12:20` | 78.40 | 57.69 | — | — | — | 99.12 | +4.53 | +8.76 |
| Ours | `XYZ + intensity`，4D，自定义 fusion | `20260327_223510` | 73.87 | 48.93 | 66.99 | 64.48 | 65.71 | 98.82 | 0.00 | 0.00 |
| DGCNN | `XYZ + intensity`，4D | `20260327_112315` | 72.47 | 46.41 | 56.79 | 71.75 | 63.40 | 98.54 | -1.40 | -2.52 |
| KPConv | `XYZ + pseudo-RGB(intensity x3)` | `20260403_161522` | 72.40 | 46.00 | — | — | — | — | -1.47 | -2.93 |
| PointNeXt | `XYZ + strength + z-height` | `20260403_121640` | 72.16 | 45.84 | 55.90 | 71.81 | 62.86 | 98.51 | -1.70 | -3.09 |
| RandLA-Net | `XYZ + intensity`，4D | `20260327_153227` | 64.88 | 31.36 | 57.19 | 40.98 | 47.74 | 98.42 | -8.98 | -17.57 |
| PointNet | `XYZ + intensity` 输入，zero-pad 到 9D | `log_leakage_v2` | 63.72 | 29.24 | 48.88 | 42.13 | 45.25 | 98.21 | -10.15 | -19.69 |
| PointNet++ | `XYZ only` | `log_leakage_semseg_v2` | — | — | — | — | — | — | — | — |

## 数据来源与口径

| Method | Metric Source | Notes |
| --- | --- | --- |
| Ours | `history.json` 中的 `best_val_metrics` | 使用 `global_miou / global_leak_iou / precision / recall / f1 / accuracy`。 |
| DGCNN | 基于 `best_model.pth` 对 `full_area_val_v2.h5` 统一复评 | 全验证集聚合口径。 |
| RandLA-Net | 基于 `best_model.pth` 对 `full_area_val_v2.h5` 统一复评 | 全验证集聚合口径。 |
| PointNet | 基于 `best_model.ckpt` 对 `full_area_val_v2.h5` 统一复评 | 全验证集聚合口径。 |
| PointNeXt | 基于 `best_model.pth` 对 `full_area_val_v2.h5` 统一复评 | 全验证集聚合口径。 |
| KPConv | `outputs_leakage_v2/20260403_161522/val_IoUs.txt` 最佳记录 | 最佳记录为第 `137` 行，`bg_iou=0.988`，`leak_iou=0.460`，`mIoU=0.724`。当前输出未保存 `precision / recall / f1 / accuracy`。 |
| PointTransformerV2 | `train.log` best 验证结果 | 使用 `train.log` 中 `2026-04-04 12:20:46` 的 best 验证：`mIoU/allAcc=0.7840/0.9912`，`Class_1 leakage IoU=0.5769`。当前日志未直接保存 `precision / recall / f1`。 |
| PointNet++ | `best_model.ckpt` 存在，但未能完成统一复评 | 在当前环境下，`TensorFlow 1.x + 自定义 CUDA ops + RTX 4090 D` 复评时反复触发 `CUDA_ERROR_ILLEGAL_ADDRESS`，因此该行暂不填数。建议回收原始训练 stdout，或在兼容环境中补跑评估。 |

## 可直接放汇报的结论

- `PointTransformerV2` 是当前最强 baseline，也是唯一明确超过你自有模型的对比模型。
- 如果只看已经完成统一复评的模型，你自己的模型仍然排第一，且 `Leak IoU` 明显高于 `DGCNN / KPConv / PointNeXt` 这一梯队。
- `DGCNN / KPConv / PointNeXt` 的 `mIoU` 都在 `72.2% ~ 72.5%`，彼此很接近，但 `Leak IoU` 仍整体落后于你的模型约 `2.5 ~ 3.1` 个点。
- `PointNet / RandLA-Net` 的总 `Accuracy` 依然看着挺高，但 `Leak IoU / F1` 掉得很明显，说明在正类稀疏场景下，单看 `Accuracy` 容易把人带沟里。

## 公平性备注

- `Ours / DGCNN / RandLA-Net` 都是 4D `XYZ + intensity`。
- `PointNet` 实际输入也是 4D，但为了兼容原实现，特征张量被 zero-pad 到 9D。
- `PointNet++` 只吃 `XYZ`，信息预算最少。
- `PointNeXt` 用的是 `strength + z-height` 作为额外特征。
- `KPConv` 和 `PointTransformerV2` 都把单通道强度复制成 3 通道伪颜色，所以它们不是严格意义上的“原生 RGB 输入”，但信息预算也并不等于 4D `XYZI`。
