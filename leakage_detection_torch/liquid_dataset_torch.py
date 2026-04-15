import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path


def _resolve_data_path(data_path: str) -> str:
    p = Path(str(data_path))
    if p.exists():
        return str(p)

    tried: list[Path] = []
    name = p.name if p.name else str(p)

    def add(cand: Path):
        if cand not in tried:
            tried.append(cand)

    add(Path.cwd() / name)
    add(Path(__file__).resolve().parent.parent / name)
    add(Path(__file__).resolve().parent / name)

    raw = str(p)
    if os.sep + "ai" + os.sep in raw:
        add(Path("/ai/0309/cloud") / name)
        add(Path("/ai/cloud/cloud") / name)

        if raw.startswith("/ai/0309/cloud/"):
            add(Path(raw.replace("/ai/0309/cloud/", "/ai/cloud/cloud/", 1)))
        if raw.startswith("/ai/cloud/cloud/"):
            add(Path(raw.replace("/ai/cloud/cloud/", "/ai/0309/cloud/", 1)))

    for cand in tried:
        if cand.exists():
            return str(cand)

    details = "\n".join([f"- {c}" for c in tried[:12]])
    raise FileNotFoundError(f"找不到数据文件: {data_path}\n尝试过的路径:\n{details}")


class LiquidLeakageDatasetTorch(Dataset):
    def __init__(
        self,
        data_path: str,
        num_points: int = 1024,
        augment: bool = False,
        normalize: bool = True,
        augment_mode: str = "basic",
        binary_class: bool = True,
    ):
        self.data_path = _resolve_data_path(str(data_path))
        self.num_points = int(num_points)
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.augment_mode = str(augment_mode)
        self.binary_class = bool(binary_class)
        self._cls_labels_are_binary = False
        self._cls_labels_are_dense_four_class = False
        self._load_data()

    def _load_data(self) -> None:
        with h5py.File(self.data_path, "r") as f:
            if "point_clouds" in f:
                self.point_clouds = f["point_clouds"][:]
            elif "alldata" in f:
                self.point_clouds = f["alldata"][:]
            else:
                raise KeyError("Dataset must contain 'point_clouds' or 'alldata'")

            if "seg_labels" in f:
                self.seg_labels = f["seg_labels"][:]
            elif "alllable" in f:
                self.seg_labels = f["alllable"][:]
            else:
                raise KeyError("Dataset must contain 'seg_labels' or 'alllable'")

            if "cls_labels" in f:
                self.cls_labels = f["cls_labels"][:]
            else:
                self.cls_labels = np.zeros(len(self.point_clouds), dtype=np.int32)

            if "positive_ratios" in f:
                self.positive_ratios = f["positive_ratios"][:].astype(np.float32)
            else:
                self.positive_ratios = (self.seg_labels == 1).mean(axis=1).astype(np.float32)

        cls_min = int(np.min(self.cls_labels)) if len(self.cls_labels) > 0 else 0
        cls_max = int(np.max(self.cls_labels)) if len(self.cls_labels) > 0 else 0
        if cls_min >= 0 and cls_max <= 1:
            zero_indices = np.where(self.cls_labels == 0)[0]
            if len(zero_indices) == 0:
                self._cls_labels_are_binary = True
            else:
                sample_size = min(len(zero_indices), 64)
                sampled = np.random.choice(zero_indices, sample_size, replace=False)
                has_leak = [bool(np.any(self.seg_labels[i] == 1)) for i in sampled]
                self._cls_labels_are_binary = (np.mean(has_leak) <= 0.5)
        else:
            self._cls_labels_are_binary = False

        self._cls_labels_are_dense_four_class = bool(
            (not self._cls_labels_are_binary) and cls_min >= 0 and cls_max <= 3
        )

    def __len__(self) -> int:
        return int(len(self.point_clouds))

    def __getitem__(self, idx: int):
        point_cloud = self.point_clouds[idx].copy()
        seg_label = self.seg_labels[idx].copy()
        cls_label = self.cls_labels[idx]

        if self.binary_class:
            label_int = int(cls_label)
            if self._cls_labels_are_binary:
                cls_label = 1 if label_int == 1 else 0
            else:
                cls_label = 1 if label_int in [0, 1, 2, 3, 4] else 0
        else:
            label_int = int(cls_label)
            if self._cls_labels_are_dense_four_class:
                cls_label = label_int
            else:
                cls_label_map = {0: 0, 1: 1, 4: 2, 5: 3}
                cls_label = cls_label_map.get(label_int, 0)

        if point_cloud.shape[0] != self.num_points:
            point_cloud, seg_label = self._resample(point_cloud, seg_label)

        if self.augment:
            if self.augment_mode == "strong":
                point_cloud = self._augment_strong(point_cloud)
            else:
                point_cloud = self._augment(point_cloud)

        if self.normalize:
            point_cloud = self._normalize(point_cloud)

        point_cloud = point_cloud.T.astype(np.float32)
        seg_label = seg_label.astype(np.int64)
        cls_label = np.array(cls_label, dtype=np.int64)

        return (
            torch.from_numpy(point_cloud),
            torch.from_numpy(seg_label),
            torch.from_numpy(cls_label),
        )

    def _resample(self, point_cloud: np.ndarray, seg_label: np.ndarray):
        current_num = point_cloud.shape[0]
        if current_num > self.num_points:
            choice = np.random.choice(current_num, self.num_points, replace=False)
        else:
            choice = np.random.choice(current_num, self.num_points, replace=True)
        return point_cloud[choice], seg_label[choice]

    def _normalize(self, point_cloud: np.ndarray) -> np.ndarray:
        xyz = point_cloud[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        point_cloud[:, :3] = xyz

        if point_cloud.shape[1] > 3:
            for i in range(3, point_cloud.shape[1]):
                feature = point_cloud[:, i]
                min_val = feature.min()
                max_val = feature.max()
                if max_val > min_val:
                    point_cloud[:, i] = (feature - min_val) / (max_val - min_val)
        return point_cloud

    def _augment(self, point_cloud: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        point_cloud[:, :3] = point_cloud[:, :3] @ rotation_matrix.T

        scale = np.random.uniform(0.8, 1.2)
        point_cloud[:, :3] *= scale

        point_cloud[:, :3] += np.random.uniform(-0.1, 0.1, 3)

        jitter = np.random.normal(0, 0.01, point_cloud[:, :3].shape)
        point_cloud[:, :3] += jitter

        if np.random.rand() < 0.3:
            num_dropout = int(self.num_points * 0.1)
            dropout_idx = np.random.choice(self.num_points, num_dropout, replace=False)
            keep_idx = np.setdiff1d(np.arange(self.num_points), dropout_idx)
            replace_idx = np.random.choice(keep_idx, num_dropout, replace=True)
            point_cloud[dropout_idx] = point_cloud[replace_idx]

        return point_cloud

    def _augment_strong(self, point_cloud: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            theta_z = np.random.uniform(0, 2 * np.pi)
            rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
            point_cloud[:, :3] = point_cloud[:, :3] @ rz.T

        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            point_cloud[:, :3] *= scale

        if np.random.rand() < 0.5:
            shift = np.random.uniform(-0.1, 0.1, 3)
            point_cloud[:, :3] += shift

        if np.random.rand() < 0.3:
            point_cloud[:, 0] = -point_cloud[:, 0]

        if np.random.rand() < 0.5:
            jitter = np.random.normal(0, 0.008, point_cloud[:, :3].shape)
            point_cloud[:, :3] += jitter

        if np.random.rand() < 0.3:
            num_dropout = int(self.num_points * 0.05)
            dropout_idx = np.random.choice(self.num_points, num_dropout, replace=False)
            keep_idx = np.setdiff1d(np.arange(self.num_points), dropout_idx)
            replace_idx = np.random.choice(keep_idx, num_dropout, replace=True)
            point_cloud[dropout_idx] = point_cloud[replace_idx]

        return point_cloud

    def compute_seg_class_weights(
        self,
        num_classes: int = 2,
        mode: str = "sqrt_inv",
        beta: float = 0.999999,
    ) -> torch.Tensor:
        labels = self.seg_labels.reshape(-1)
        counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        if mode == "inverse":
            weights = 1.0 / counts
        elif mode == "effective_num":
            beta = float(np.clip(beta, 0.0, 0.999999))
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        else:
            weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * float(num_classes)
        return torch.tensor(weights, dtype=torch.float32)

    def compute_sample_weights(
        self,
        mode: str = "none",
        alpha: float = 0.5,
        min_positive_ratio: float = 0.005,
        max_weight: float = 4.0,
        empty_weight: float = 1.0,
    ) -> torch.Tensor:
        num_samples = len(self.point_clouds)
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.double)

        weights = np.ones(num_samples, dtype=np.float64)
        if mode == "none":
            return torch.tensor(weights, dtype=torch.double)

        positive_ratios = self.positive_ratios.astype(np.float64)
        positive_mask = positive_ratios > 0
        positive_count = int(positive_mask.sum())
        negative_count = int(num_samples - positive_count)
        if positive_count == 0 or negative_count == 0:
            return torch.tensor(weights, dtype=torch.double)

        positive_base = num_samples / (2.0 * positive_count)
        negative_base = num_samples / (2.0 * negative_count)
        weights[~positive_mask] = negative_base * float(empty_weight)
        weights[positive_mask] = positive_base

        if mode == "small_positive":
            positive_values = positive_ratios[positive_mask]
            reference_ratio = max(float(np.median(positive_values)), float(min_positive_ratio))
            clipped = np.maximum(positive_values, float(min_positive_ratio))
            rarity = np.power(reference_ratio / clipped, float(alpha))
            rarity = np.clip(rarity, 1.0, float(max_weight))
            weights[positive_mask] *= rarity

        return torch.tensor(weights, dtype=torch.double)
