# Segmentation Run Comparison

- generated_at: 2026-04-05T13:48:23
- preset: legacy3

## Latest Runs

| Group | Run | Best Metric | Best Epoch | gLeak IoU (%) | F1 (%) | Recall (%) | Best Checkpoint |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Current | 20260404_185555 | val_iou | 195 | 49.43 | 66.16 | 63.10 | /ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly/20260404_185555/best_model.pt |
| Balanced | 20260326_110221 | val_iou | 200 | 48.49 | 65.31 | 61.37 | /ai/0309/cloud/output/seg_balanced/20260326_110221/best_model.pt |
| Recall | 20260326_190253 | val_iou | 210 | 49.15 | 65.90 | 65.40 | /ai/0309/cloud/output/seg_recall/20260326_190253/best_model.pt |

## Rankings

- global_leak_iou: Current (49.43%)
- f1: Current (66.16%)
- recall: Recall (65.40%)
- precision: Balanced (69.79%)
- iou: Current (73.10%)
- leak_iou: Current (47.35%)

## Deltas Vs First Group

| Group | Δ gLeak IoU | Δ F1 | Δ Recall |
| --- | ---: | ---: | ---: |
| Current | +0.00 pp | +0.00 pp | +0.00 pp |
| Balanced | -0.94 pp | -0.85 pp | -1.73 pp |
| Recall | -0.28 pp | -0.25 pp | +2.30 pp |
