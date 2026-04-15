import numpy as np
import importlib.util
from pathlib import Path


def load_paddle_reference():
    root = Path(__file__).resolve().parents[1]
    ref_path = root / "leakage_detection" / "fusion_liquid_model_v2.py"
    if not ref_path.exists():
        raise FileNotFoundError(f"Cannot find Paddle reference file: {ref_path}")

    spec = importlib.util.spec_from_file_location("fusion_liquid_model_v2_paddle_ref", str(ref_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for: {ref_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.knn, mod.get_graph_feature


def seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    try:
        import paddle

        paddle.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def knn_torch(x, k: int):
    import torch

    bsz, _c, n = x.shape
    k = min(int(k), int(n))
    xt = x.transpose(1, 2)
    inner = -2.0 * torch.matmul(xt, x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(1, 2)
    idx = torch.topk(pairwise, k=k, dim=-1, largest=True, sorted=False).indices
    return idx.clamp(min=0, max=n - 1)


def graph_feature_torch(x, idx):
    import torch

    bsz, c, n = x.shape
    idx = idx.to(dtype=torch.long).clamp(min=0, max=n - 1)
    k = idx.shape[-1]
    xt = x.transpose(1, 2).contiguous()
    x_flat = xt.reshape(bsz * n, c)
    batch_offset = (torch.arange(bsz, device=x.device, dtype=torch.long).view(bsz, 1, 1) * n).expand(bsz, n, k)
    global_idx = (batch_offset + idx).reshape(-1)
    neighbors = x_flat.index_select(0, global_idx).reshape(bsz, n, k, c)
    center = xt.unsqueeze(2).expand(bsz, n, k, c)
    edge = torch.cat([neighbors - center, center], dim=-1).permute(0, 3, 1, 2).contiguous()
    return edge


def compare_once(bsz=2, c=3, n=256, k=20, tie_case=False, device="cpu"):
    import paddle
    import torch

    if tie_case:
        x_np = np.random.randn(bsz, c, n).astype("float32")
        x_np[:, :, :10] = x_np[:, :, 0:1]
    else:
        x_np = np.random.randn(bsz, c, n).astype("float32")

    x_pd = paddle.to_tensor(x_np)
    x_th = torch.tensor(x_np, device=device)

    knn_paddle, graph_paddle = load_paddle_reference()

    idx_pd = knn_paddle(x_pd, k=k).numpy()
    idx_th = knn_torch(x_th, k=k).cpu().numpy()

    idx_equal_ratio = float((idx_pd == idx_th).mean())

    inter = 0.0
    for b in range(bsz):
        for i in range(n):
            sp = set(idx_pd[b, i].tolist())
            st = set(idx_th[b, i].tolist())
            inter += len(sp & st) / float(k)
    idx_set_recall = inter / float(bsz * n)

    idx_ref = idx_pd
    edge_pd = graph_paddle(x_pd, k=k, idx=paddle.to_tensor(idx_ref)).numpy()
    edge_th = graph_feature_torch(x_th, torch.tensor(idx_ref, device=device)).cpu().numpy()

    diff = np.abs(edge_pd - edge_th)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    return {
        "tie_case": bool(tie_case),
        "idx_equal_ratio": idx_equal_ratio,
        "idx_set_recall": idx_set_recall,
        "edge_max_abs": max_abs,
        "edge_mean_abs": mean_abs,
    }


def main():
    seed_all(0)

    res1 = compare_once(tie_case=False)
    res2 = compare_once(tie_case=True)

    print(res1)
    print(res2)


if __name__ == "__main__":
    main()
