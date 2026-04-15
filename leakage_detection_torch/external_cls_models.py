from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    dist = square_distance(new_xyz, xyz)
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    batch_size, num_dims, num_points = x.size()
    xt = x.transpose(2, 1).contiguous()
    idx = knn_point(k, xt, xt)
    neighbors = index_points(xt, idx)
    center = xt.unsqueeze(2).expand(-1, -1, k, -1)
    feature = torch.cat((neighbors - center, center), dim=-1)
    return feature.permute(0, 3, 1, 2).contiguous()


class MLP1d(nn.Module):
    def __init__(self, channels: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0 and i < len(channels) - 2:
                layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedMLP2d(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetClassifier(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = MLP1d([in_channels, 64, 64, 64, 128, 1024], dropout=0.0)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        global_feat = torch.max(feat, dim=2)[0]
        return self.classifier(global_feat)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        graph_feature = get_graph_feature(x, k=k)
        out = self.conv(graph_feature)
        return out.max(dim=-1)[0]


class DGCNNClassifier(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, k: int = 20, dropout: float = 0.4):
        super().__init__()
        self.k = int(k)
        self.block1 = EdgeConvBlock(in_channels, 64)
        self.block2 = EdgeConvBlock(64, 64)
        self.block3 = EdgeConvBlock(64, 128)
        self.block4 = EdgeConvBlock(128, 256)
        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x, self.k)
        x2 = self.block2(x1, self.k)
        x3 = self.block3(x2, self.k)
        x4 = self.block4(x3, self.k)
        fusion = self.fusion(torch.cat([x1, x2, x3, x4], dim=1))
        max_pool = F.adaptive_max_pool1d(fusion, 1).squeeze(-1)
        avg_pool = F.adaptive_avg_pool1d(fusion, 1).squeeze(-1)
        return self.classifier(torch.cat([max_pool, avg_pool], dim=1))


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, k: int, in_channels: int, mlp_channels: list[int]):
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)
        self.mlp = SharedMLP2d([in_channels + 3] + mlp_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        group_idx = knn_point(self.k, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
        if features is not None:
            grouped_features = index_points(features.transpose(1, 2).contiguous(), group_idx)
            grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()
            grouped_input = torch.cat([grouped_xyz_norm.permute(0, 3, 1, 2).contiguous(), grouped_features], dim=1)
        else:
            grouped_input = grouped_xyz_norm.permute(0, 3, 1, 2).contiguous()
        new_features = self.mlp(grouped_input).max(dim=-1)[0]
        return new_xyz, new_features


class PointNet2Classifier(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.sa1 = SetAbstraction(npoint=512, k=32, in_channels=in_channels - 3, mlp_channels=[64, 64, 128])
        self.sa2 = SetAbstraction(npoint=128, k=32, in_channels=128, mlp_channels=[128, 128, 256])
        self.sa3 = SetAbstraction(npoint=32, k=32, in_channels=256, mlp_channels=[256, 512, 1024])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        features = x[:, 3:, :] if x.shape[1] > 3 else None
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        _, features = self.sa3(xyz, features)
        global_feat = torch.max(features, dim=2)[0]
        return self.classifier(global_feat)


def build_external_cls_model(model_name: str, in_channels: int = 4, num_classes: int = 4) -> nn.Module:
    name = str(model_name).strip().lower()
    if name == "pointnet":
        return PointNetClassifier(in_channels=in_channels, num_classes=num_classes)
    if name in {"pointnet2", "pointnet++", "pointnetpp"}:
        return PointNet2Classifier(in_channels=in_channels, num_classes=num_classes)
    if name == "dgcnn":
        return DGCNNClassifier(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"未支持的外部分类 baseline: {model_name}")
