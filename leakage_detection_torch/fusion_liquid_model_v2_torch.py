import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    batch_size = x.shape[0]
    num_points = x.shape[2]
    k = min(int(k), int(num_points))
    xt = x.transpose(1, 2)
    inner = -2.0 * torch.matmul(xt, x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(1, 2)
    idx = torch.topk(pairwise_distance, k=k, dim=-1, largest=True, sorted=False).indices
    idx = idx.clamp(min=0, max=num_points - 1)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor | None = None) -> torch.Tensor:
    batch_size = x.shape[0]
    num_points = x.shape[2]
    num_dims = x.shape[1]
    k = min(int(k), int(num_points))
    if idx is None:
        idx = knn(x, k=k)
    idx = idx.to(dtype=torch.long).clamp(min=0, max=num_points - 1)
    actual_k = idx.shape[-1]
    x_t = x.transpose(1, 2).contiguous()
    x_flat = x_t.reshape(batch_size * num_points, num_dims)
    batch_offset = (torch.arange(batch_size, device=x.device, dtype=torch.long).view(batch_size, 1, 1) * num_points)
    batch_offset = batch_offset.expand(batch_size, num_points, actual_k)
    global_idx = (batch_offset + idx).reshape(-1)
    neighbor_feat = x_flat.index_select(0, global_idx)
    neighbor_feat = neighbor_feat.reshape(batch_size, num_points, actual_k, num_dims)
    center_feat = x_t.unsqueeze(2).expand(batch_size, num_points, actual_k, num_dims)
    edge_feat = torch.cat([neighbor_feat - center_feat, center_feat], dim=-1)
    edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()
    return edge_feat


class SimplifiedNoiseEstimator(nn.Module):
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = int(k)
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError(f"points must be [B, 3, N] or [B, N, 3], got {tuple(points.shape)}")
        if points.shape[1] != 3 and points.shape[2] == 3:
            points = points.transpose(1, 2).contiguous()
        return self.mlp(points)


class NoiseGuidedEncoder(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels + 1, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, features: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([features, noise_map], dim=1)
        return self.encoder(combined)


class MultiScaleEdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k_scales: list[int] | None = None):
        super().__init__()
        self.k_scales = list(k_scales) if k_scales is not None else [10, 20, 40]
        self.scale_convs = nn.ModuleList()
        for _k in self.k_scales:
            conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.scale_convs.append(conv)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        multi_scale_features: list[torch.Tensor] = []
        for i, k in enumerate(self.k_scales):
            graph_feat = get_graph_feature(x, k=k)
            conv_feat = self.scale_convs[i](graph_feat)
            pooled = conv_feat.max(dim=-1).values
            multi_scale_features.append(pooled)
        return multi_scale_features


class AdaptiveScaleFusion(nn.Module):
    def __init__(self, num_scales: int = 3, channels: int = 64):
        super().__init__()
        self.num_scales = int(num_scales)
        self.attention = nn.Sequential(
            nn.Conv1d(channels * self.num_scales, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, self.num_scales, 1),
        )

    def forward(self, multi_scale_features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        concat_feat = torch.cat(multi_scale_features, dim=1)
        weights = self.attention(concat_feat)
        weights = torch.softmax(weights, dim=1)
        fused = torch.zeros_like(multi_scale_features[0])
        for i, feat in enumerate(multi_scale_features):
            w = weights[:, i : i + 1, :]
            fused = fused + w * feat
        return fused, weights


class BoundaryAwareRefinement(nn.Module):
    def __init__(self, in_channels: int, seg_classes: int, input_mode: str = "features_fine_probs"):
        super().__init__()
        valid_input_channels = {
            "features": in_channels,
            "features_fine_probs": in_channels + seg_classes,
        }
        if input_mode not in valid_input_channels:
            raise ValueError(f"Unsupported boundary input mode: {input_mode}")
        self.input_mode = input_mode
        self.boundary_detector = nn.Sequential(
            nn.Conv1d(valid_input_channels[input_mode], 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.boundary_enhance = nn.Sequential(
            nn.Conv1d(in_channels + 1, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, in_channels, 1),
            nn.BatchNorm1d(in_channels),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, fine_probs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_mode == "features_fine_probs":
            if fine_probs is None:
                raise ValueError("fine_probs is required when boundary_input_mode=features_fine_probs")
            boundary_input = torch.cat([features, fine_probs], dim=1)
        else:
            boundary_input = features
        boundary_map = self.boundary_detector(boundary_input)
        enhanced_input = torch.cat([features, boundary_map], dim=1)
        enhanced_feat = self.boundary_enhance(enhanced_input)
        gate = self.gate(torch.cat([features, enhanced_feat], dim=1))
        refined_feat = features + gate * enhanced_feat
        return refined_feat, boundary_map


class LearnableNoiseLeakageCorrelation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.correlation_net = nn.Sequential(
            nn.Conv1d(2, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, noise_map: torch.Tensor, seg_prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([noise_map, seg_prob], dim=1)
        correlation = self.correlation_net(combined)
        alpha = torch.sigmoid(self.alpha)
        adjustment = alpha * correlation * (noise_map - 0.5)
        refined = seg_prob + adjustment * seg_prob * (1.0 - seg_prob)
        refined = refined.clamp(min=1e-7, max=1.0 - 1e-7)
        return refined, correlation


class ProgressiveDetectorV2(nn.Module):
    def __init__(self, in_channels: int, seg_classes: int = 2, boundary_input_mode: str = "features_fine_probs"):
        super().__init__()
        self.coarse_head = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, seg_classes, 1),
        )
        self.fine_head = nn.Sequential(
            nn.Conv1d(in_channels + seg_classes, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, seg_classes, 1),
        )
        self.boundary_refine = BoundaryAwareRefinement(in_channels, seg_classes, input_mode=boundary_input_mode)
        self.final_head = nn.Sequential(
            nn.Conv1d(in_channels + seg_classes + 1, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, seg_classes, 1),
        )

    def forward(self, features: torch.Tensor, noise_map: torch.Tensor) -> dict[str, torch.Tensor]:
        coarse_logits = self.coarse_head(features)
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        fine_input = torch.cat([features, coarse_probs], dim=1)
        fine_logits = self.fine_head(fine_input)
        fine_probs = torch.softmax(fine_logits, dim=1)
        refined_feat, boundary_map = self.boundary_refine(features, fine_probs)
        final_input = torch.cat([refined_feat, fine_probs, boundary_map], dim=1)
        final_logits = self.final_head(final_input)
        return {
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "refined_logits": final_logits,
            "boundary_map": boundary_map,
        }


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads

        self.geo_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intensity_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intensity_value = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.intensity_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.geo_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.geo_value = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.geo_out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.intensity_out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(float(dropout))
        self.scale = self.head_dim ** -0.5

    def multi_head_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        q = query.reshape(batch_size, self.num_heads, 1, self.head_dim)
        k = key.reshape(batch_size, self.num_heads, 1, self.head_dim)
        v = value.reshape(batch_size, self.num_heads, 1, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v).reshape(batch_size, self.hidden_dim)
        return output, attn_weights

    def forward(
        self, geo_feat: torch.Tensor, intensity_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        q_geo = self.geo_query(geo_feat)
        k_intensity = self.intensity_key(intensity_feat)
        v_intensity = self.intensity_value(intensity_feat)
        geo_attended, geo_attn = self.multi_head_attention(q_geo, k_intensity, v_intensity)
        geo_enhanced = geo_feat + self.geo_out_proj(geo_attended)

        q_intensity = self.intensity_query(intensity_feat)
        k_geo = self.geo_key(geo_feat)
        v_geo = self.geo_value(geo_feat)
        intensity_attended, intensity_attn = self.multi_head_attention(q_intensity, k_geo, v_geo)
        intensity_enhanced = intensity_feat + self.intensity_out_proj(intensity_attended)

        return geo_enhanced, intensity_enhanced, {
            "geo_to_intensity": geo_attn,
            "intensity_to_geo": intensity_attn,
        }


class UncertaintyEstimator(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 6):
        super().__init__()
        self.num_classes = int(num_classes)

        self.geo_uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.intensity_uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self.geo_classifier = nn.Linear(hidden_dim, self.num_classes)
        self.intensity_classifier = nn.Linear(hidden_dim, self.num_classes)

    def forward(
        self, geo_feat: torch.Tensor, intensity_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        geo_logits = self.geo_classifier(geo_feat)
        intensity_logits = self.intensity_classifier(intensity_feat)
        geo_uncertainty = self.geo_uncertainty_net(geo_feat)
        intensity_uncertainty = self.intensity_uncertainty_net(intensity_feat)
        return geo_logits, intensity_logits, geo_uncertainty, intensity_uncertainty


class LiquidTypeAdaptiveGate(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_liquid_types: int = 6):
        super().__init__()
        self.num_liquid_types = int(num_liquid_types)
        self.hidden_dim = int(hidden_dim)

        self.type_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_liquid_types),
        )
        self.type_specific_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(2, dtype=torch.float32)) for _ in range(self.num_liquid_types)]
        )
        self.dynamic_gate_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(
        self, geo_feat: torch.Tensor, intensity_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = torch.cat([geo_feat, intensity_feat], dim=1)
        type_logits = self.type_predictor(combined)
        type_probs = torch.softmax(type_logits, dim=1)

        type_gates = torch.stack([torch.softmax(gate, dim=0) for gate in self.type_specific_gates], dim=0)
        type_adaptive_gate = torch.matmul(type_probs, type_gates)
        dynamic_gate = torch.softmax(self.dynamic_gate_net(combined), dim=1)
        adaptive_gate = 0.6 * type_adaptive_gate + 0.4 * dynamic_gate
        return adaptive_gate, type_logits, type_probs


class UncertaintyGuidedCrossModalFusion(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 128,
        num_classes: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.geo_channels = min(3, self.in_channels)
        self.intensity_channels = max(self.in_channels - self.geo_channels, 1)

        self.geo_encoder = nn.Sequential(
            nn.Conv1d(self.geo_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.intensity_encoder = nn.Sequential(
            nn.Conv1d(self.intensity_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.cross_modal_attention = CrossModalAttention(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.uncertainty_estimator = UncertaintyEstimator(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
        )
        self.adaptive_gate = LiquidTypeAdaptiveGate(
            hidden_dim=self.hidden_dim,
            num_liquid_types=self.num_classes,
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_weights: torch.Tensor | None = None,
        return_intermediate: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        batch_size, _, num_points = x.shape

        geo_input = x[:, : self.geo_channels, :]
        if x.shape[1] > self.geo_channels:
            intensity_input = x[:, self.geo_channels :, :]
        else:
            intensity_input = x.new_zeros(batch_size, self.intensity_channels, num_points)

        if intensity_input.shape[1] < self.intensity_channels:
            pad_channels = self.intensity_channels - intensity_input.shape[1]
            pad = x.new_zeros(batch_size, pad_channels, num_points)
            intensity_input = torch.cat([intensity_input, pad], dim=1)

        geo_feat = self.geo_encoder(geo_input)
        intensity_feat = self.intensity_encoder(intensity_input)

        if attn_weights is None:
            attn_weights = x.new_ones(batch_size, 1, num_points)
        norm_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        geo_desc = (geo_feat * norm_weights).sum(dim=-1)
        intensity_desc = (intensity_feat * norm_weights).sum(dim=-1)

        geo_enhanced, intensity_enhanced, cross_attn_weights = self.cross_modal_attention(
            geo_desc, intensity_desc
        )
        geo_logits, intensity_logits, geo_uncertainty, intensity_uncertainty = self.uncertainty_estimator(
            geo_enhanced, intensity_enhanced
        )

        geo_weight = 1.0 / (geo_uncertainty + 1e-6)
        intensity_weight = 1.0 / (intensity_uncertainty + 1e-6)
        total_weight = geo_weight + intensity_weight
        geo_weight_norm = geo_weight / total_weight
        intensity_weight_norm = intensity_weight / total_weight
        uncertainty_weighted_logits = geo_weight_norm * geo_logits + intensity_weight_norm * intensity_logits

        adaptive_gate_weights, type_logits, type_probs = self.adaptive_gate(geo_enhanced, intensity_enhanced)
        fused_feat = (
            adaptive_gate_weights[:, 0:1] * geo_enhanced
            + adaptive_gate_weights[:, 1:2] * intensity_enhanced
        )
        combined_feat = torch.cat([geo_enhanced, intensity_enhanced], dim=1)
        fused_logits = self.fusion_classifier(combined_feat)
        final_logits = 0.4 * fused_logits + 0.3 * uncertainty_weighted_logits + 0.3 * type_logits

        if return_intermediate:
            return final_logits, {
                "geo_logits": geo_logits,
                "intensity_logits": intensity_logits,
                "geo_uncertainty": geo_uncertainty,
                "intensity_uncertainty": intensity_uncertainty,
                "uncertainty_weighted_logits": uncertainty_weighted_logits,
                "type_logits": type_logits,
                "type_probs": type_probs,
                "adaptive_gate_weights": adaptive_gate_weights,
                "cross_attn_weights": cross_attn_weights,
                "geo_weight_norm": geo_weight_norm,
                "intensity_weight_norm": intensity_weight_norm,
                "fused_feat": fused_feat,
            }
        return final_logits


class DN_MS_LiquidNet_V2_Torch(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        seg_classes: int = 2,
        cls_classes: int = 6,
        k_scales: list[int] | None = None,
        use_noise_guidance: bool = True,
        use_progressive: bool = True,
        use_noise_leak_corr: bool = True,
        use_multi_scale: bool = True,
        use_uncertainty_fusion: bool = True,
        use_simple_uncertainty: bool = False,
        disable_cls: bool = False,
        boundary_input_mode: str = "features_fine_probs",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.seg_classes = int(seg_classes)
        self.cls_classes = int(cls_classes)
        requested_k_scales = list(k_scales) if k_scales is not None else [10, 20, 40]
        if len(requested_k_scales) == 0:
            requested_k_scales = [20]
        self.use_noise_guidance = bool(use_noise_guidance)
        self.use_progressive = bool(use_progressive)
        self.use_noise_leak_corr = bool(use_noise_leak_corr)
        self.use_multi_scale = bool(use_multi_scale)
        self.use_simple_uncertainty = bool(use_simple_uncertainty)
        self.disable_cls = bool(disable_cls)
        self.use_uncertainty_fusion = bool(use_uncertainty_fusion) and not self.disable_cls
        self.boundary_input_mode = str(boundary_input_mode)
        if self.use_multi_scale:
            self.k_scales = requested_k_scales
        else:
            self.k_scales = [requested_k_scales[len(requested_k_scales) // 2]]

        self.noise_estimator = SimplifiedNoiseEstimator(k=20)

        if self.use_noise_guidance:
            self.noise_guided_encoder = NoiseGuidedEncoder(self.in_channels)
            encoder_out_channels = 128
        else:
            self.input_encoder = nn.Sequential(
                nn.Conv1d(self.in_channels, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2, inplace=True),
            )
            encoder_out_channels = 128

        self.ms_conv1 = MultiScaleEdgeConv(encoder_out_channels, 64, self.k_scales)
        self.ms_conv2 = MultiScaleEdgeConv(64, 128, self.k_scales)
        self.ms_conv3 = MultiScaleEdgeConv(128, 256, self.k_scales)

        self.scale_fusion1 = AdaptiveScaleFusion(len(self.k_scales), 64)
        self.scale_fusion2 = AdaptiveScaleFusion(len(self.k_scales), 128)
        self.scale_fusion3 = AdaptiveScaleFusion(len(self.k_scales), 256)

        total_feat_channels = 64 + 128 + 256
        self.global_conv = nn.Sequential(
            nn.Conv1d(total_feat_channels, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if self.use_progressive:
            self.progressive_detector = ProgressiveDetectorV2(
                512,
                self.seg_classes,
                boundary_input_mode=self.boundary_input_mode,
            )
        else:
            self.seg_head = nn.Sequential(
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, self.seg_classes, 1),
            )

        if self.use_noise_leak_corr:
            self.noise_leak_corr = LearnableNoiseLeakageCorrelation()

        if not self.disable_cls:
            self.attention_pool = nn.Sequential(
                nn.Conv1d(512, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 1, 1),
                nn.Sigmoid(),
            )
            if self.use_uncertainty_fusion:
                self.use_simple_uncertainty = False
                self.uncertainty_cross_modal = UncertaintyGuidedCrossModalFusion(
                    in_channels=self.in_channels,
                    hidden_dim=128,
                    num_classes=self.cls_classes,
                    num_heads=4,
                    dropout=0.1,
                )
            else:
                self.cls_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, self.cls_classes),
                )

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        aux_outputs: dict[str, torch.Tensor] = {}

        xyz = x[:, :3, :]
        noise_map = self.noise_estimator(xyz)
        aux_outputs["noise_map"] = noise_map

        if self.use_noise_guidance:
            encoded_feat = self.noise_guided_encoder(x, noise_map)
        else:
            encoded_feat = self.input_encoder(x)

        ms_feat1_list = self.ms_conv1(encoded_feat)
        ms_feat1, scale_weights1 = self.scale_fusion1(ms_feat1_list)
        aux_outputs["scale_weights1"] = scale_weights1

        ms_feat2_list = self.ms_conv2(ms_feat1)
        ms_feat2, scale_weights2 = self.scale_fusion2(ms_feat2_list)
        aux_outputs["scale_weights2"] = scale_weights2

        ms_feat3_list = self.ms_conv3(ms_feat2)
        ms_feat3, scale_weights3 = self.scale_fusion3(ms_feat3_list)
        aux_outputs["scale_weights3"] = scale_weights3

        multi_scale_feat = torch.cat([ms_feat1, ms_feat2, ms_feat3], dim=1)
        global_feat = self.global_conv(multi_scale_feat)

        if self.use_progressive:
            prog_results = self.progressive_detector(global_feat, noise_map)
            seg_logits = prog_results["refined_logits"]
            aux_outputs["coarse_logits"] = prog_results["coarse_logits"]
            aux_outputs["fine_logits"] = prog_results["fine_logits"]
            aux_outputs["boundary_map"] = prog_results["boundary_map"]
        else:
            seg_logits = self.seg_head(global_feat)

        seg_probs = torch.softmax(seg_logits, dim=1)
        leakage_prob = seg_probs[:, 1:2, :]

        if self.use_noise_leak_corr:
            refined_leak_prob, correlation = self.noise_leak_corr(noise_map, leakage_prob)
            seg_output = torch.cat([1.0 - refined_leak_prob, refined_leak_prob], dim=1)
            aux_outputs["correlation_map"] = correlation
        else:
            seg_output = seg_probs

        if self.disable_cls:
            cls_output = None
        else:
            attn_weights = self.attention_pool(global_feat)
            aux_outputs["attention_weights"] = attn_weights
            if self.use_uncertainty_fusion:
                cls_output, uncertainty_aux = self.uncertainty_cross_modal(
                    x, attn_weights, return_intermediate=True
                )
                aux_outputs.update(uncertainty_aux)
            else:
                weighted_feat = (global_feat * attn_weights).sum(dim=-1)
                cls_output = self.cls_head(weighted_feat)

        if return_intermediate:
            return seg_output, cls_output, aux_outputs
        return seg_output, cls_output

    @staticmethod
    def postprocess_has_leak(
        leak_prob: np.ndarray,
        xyz: np.ndarray,
        prob_threshold: float = 0.4,
        voxel_size: float = 0.05,
        min_points: int = 2,
        min_voxels: int = 1,
    ) -> np.ndarray:
        leak_prob = np.asarray(leak_prob)
        xyz = np.asarray(xyz)
        if leak_prob.ndim != 2 or xyz.ndim != 3 or leak_prob.shape[0] != xyz.shape[0] or leak_prob.shape[1] != xyz.shape[1] or xyz.shape[2] != 3:
            raise ValueError(f"Invalid shapes: leak_prob={leak_prob.shape}, xyz={xyz.shape}")

        if min_points <= 0:
            min_points = 1
        if min_voxels <= 0:
            min_voxels = 1
        if voxel_size <= 0:
            voxel_size = 0.05

        shift_y = 21
        shift_z = 42
        mask = (1 << 21) - 1
        offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)

        def pack(vx, vy, vz):
            return (vx & mask) | ((vy & mask) << shift_y) | ((vz & mask) << shift_z)

        def unpack(pid):
            vx = pid & mask
            vy = (pid >> shift_y) & mask
            vz = (pid >> shift_z) & mask
            return vx, vy, vz

        bsz = leak_prob.shape[0]
        has_leak = np.zeros((bsz,), dtype=bool)

        xyz_min = xyz.min(axis=1, keepdims=True)
        vox = np.floor((xyz - xyz_min) / voxel_size).astype(np.int64)

        for i in range(bsz):
            idx = np.flatnonzero(leak_prob[i] > prob_threshold)
            if idx.size < min_points:
                continue
            vx = vox[i, idx, 0]
            vy = vox[i, idx, 1]
            vz = vox[i, idx, 2]
            pids = pack(vx, vy, vz)
            uniq, counts = np.unique(pids, return_counts=True)
            if uniq.size < min_voxels:
                continue

            voxel_set = set(int(v) for v in uniq.tolist())
            count_map = {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}
            visited = set()

            keep = False
            for pid in voxel_set:
                if pid in visited:
                    continue
                q = [pid]
                visited.add(pid)
                comp_vox = 0
                comp_pts = 0
                while q:
                    cur = q.pop()
                    comp_vox += 1
                    comp_pts += count_map.get(cur, 0)
                    cx, cy, cz = unpack(np.int64(cur))
                    for dx, dy, dz in offsets:
                        nid = pack(cx + dx, cy + dy, cz + dz)
                        nid = int(np.int64(nid))
                        if nid in voxel_set and nid not in visited:
                            visited.add(nid)
                            q.append(nid)
                if comp_pts >= min_points and comp_vox >= min_voxels:
                    keep = True
                    break

            has_leak[i] = keep

        return has_leak
