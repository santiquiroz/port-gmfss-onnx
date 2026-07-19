"""Composicion propia (no vendoreada) del pipeline GMFSS_Fortuna, variante PG/base, CPU-only.

Combina los modulos vendoreados de HolyWu/vs-gmfss_fortuna (FeatureNet, GMFlow, MetricNet,
FusionNet_b) con softsplat_torch de 98mxr/GMFSS_Fortuna (softmax-splatting sin cupy). Ningun
codigo de este archivo viene copiado de GMFSS.py ni de GMFSS_infer_b.py -- es una reescritura
propia de la misma composicion matematica (FeatureNet -> GMFlow -> MetricNet -> softsplat x8 ->
FusionNet) usando exclusivamente los bloques MIT vendoreados. Ver docs/vendored-sources.md para
las fuentes y commits exactos.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from vs_gmfss_fortuna.FeatureNet import FeatureNet  # noqa: E402
from vs_gmfss_fortuna.FusionNet_b import GridNet  # noqa: E402
from vs_gmfss_fortuna.gmflow.gmflow import GMFlow  # noqa: E402
from vs_gmfss_fortuna.MetricNet import MetricNet  # noqa: E402
from gmfss_fortuna_98mxr.softsplat_torch import softsplat as warp  # noqa: E402

PAD_MULTIPLE = 64

Recorder = Callable[[str, torch.Tensor], None]


def _load_state_dict(weights_path: Path) -> dict:
    # weights_only=True: these .pkl are plain tensor state_dicts vendored from a trusted MIT
    # repo we cloned ourselves (see docs/vendored-sources.md) -- refuses to unpickle anything
    # beyond tensors/basic containers, closing the arbitrary-code-exec pickle hole.
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def target_padded_size(height: int, width: int, multiple: int = PAD_MULTIPLE) -> tuple[int, int]:
    padded_height = ((height - 1) // multiple + 1) * multiple
    padded_width = ((width - 1) // multiple + 1) * multiple
    return padded_height, padded_width


def resize_bilinear(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Matches upstream GMFSS.py / inference_video.py: resize (stretch), not zero-pad."""
    return F.interpolate(tensor, (height, width), mode="bilinear", align_corners=False)


def normalize_frame(frame_uint8_hwc_rgb: np.ndarray) -> torch.Tensor:
    """np.uint8 [H,W,3] RGB -> torch.float32 [1,3,H,W] in [0,1]."""
    chw = np.transpose(frame_uint8_hwc_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(chw).unsqueeze(0).contiguous()


def denormalize_frame(tensor_1chw: torch.Tensor) -> np.ndarray:
    """torch.float32 [1,3,H,W] in [0,1] -> np.uint8 [H,W,3] RGB."""
    clamped = tensor_1chw.clamp(0, 1)
    array = (clamped[0] * 255.0).round().byte().cpu().numpy()
    return np.transpose(array, (1, 2, 0))


class GMFSSBasePipeline(nn.Module):
    """GMFSS_Fortuna, variante 'base' (PG/pg104 -- ver docs/vendored-sources.md), sin cupy."""

    def __init__(self) -> None:
        super().__init__()
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()
        self.eval()

    def load_weights(self, models_dir: Path) -> None:
        self.flownet.load_state_dict(_load_state_dict(models_dir / "flownet.pkl"))
        self.metricnet.load_state_dict(_load_state_dict(models_dir / "metric_base.pkl"))
        self.feat_ext.load_state_dict(_load_state_dict(models_dir / "feat_base.pkl"))
        self.fusionnet.load_state_dict(_load_state_dict(models_dir / "fusionnet_base.pkl"))

    @torch.inference_mode()
    def extract_features(self, img0: torch.Tensor, img1: torch.Tensor, record: Optional[Recorder]):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)
        if record is not None:
            record("feat0_scale1", feat11)
            record("feat0_scale2", feat12)
            record("feat0_scale3", feat13)
            record("feat1_scale1", feat21)
            record("feat1_scale2", feat22)
            record("feat1_scale3", feat23)
        return (feat11, feat12, feat13), (feat21, feat22, feat23)

    @torch.inference_mode()
    def estimate_flow_and_metric(self, img0_half: torch.Tensor, img1_half: torch.Tensor, record: Optional[Recorder]):
        flow01 = self.flownet(img0_half, img1_half)
        flow10 = self.flownet(img1_half, img0_half)
        if record is not None:
            record("flow01", flow01)
            record("flow10", flow10)

        metric0, metric1 = self.metricnet(img0_half, img1_half, flow01, flow10)
        if record is not None:
            record("metric0", metric0)
            record("metric1", metric1)
        return flow01, flow10, metric0, metric1

    @torch.inference_mode()
    def reuse(self, img0: torch.Tensor, img1: torch.Tensor, record: Optional[Recorder] = None):
        """Mirrors upstream GMFSS.reuse(): the per-pair-independent-of-t computation."""
        feat0, feat1 = self.extract_features(img0, img1, record)

        img0_half = resize_bilinear(img0, img0.shape[2] // 2, img0.shape[3] // 2)
        img1_half = resize_bilinear(img1, img1.shape[2] // 2, img1.shape[3] // 2)
        flow01, flow10, metric0, metric1 = self.estimate_flow_and_metric(img0_half, img1_half, record)

        return flow01, flow10, metric0, metric1, feat0, feat1

    @staticmethod
    def _splat_pyramid_level(feat0, feat1, flow0t, flow1t, z0t, z1t, scale: float):
        if scale != 1.0:
            flow0t = resize_bilinear(flow0t, int(flow0t.shape[2] * scale), int(flow0t.shape[3] * scale)) * scale
            flow1t = resize_bilinear(flow1t, int(flow1t.shape[2] * scale), int(flow1t.shape[3] * scale)) * scale
            z0t = resize_bilinear(z0t, int(z0t.shape[2] * scale), int(z0t.shape[3] * scale))
            z1t = resize_bilinear(z1t, int(z1t.shape[2] * scale), int(z1t.shape[3] * scale))
        splat0 = warp(feat0, flow0t, z0t, strMode="soft")
        splat1 = warp(feat1, flow1t, z1t, strMode="soft")
        return splat0, splat1

    @torch.inference_mode()
    def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: torch.Tensor,
                record: Optional[Recorder] = None) -> torch.Tensor:
        flow01, flow10, metric0, metric1, feat0, feat1 = self.reuse(img0, img1, record)
        feat11, feat12, feat13 = feat0
        feat21, feat22, feat23 = feat1

        f1t = timestep * flow01
        f2t = (1 - timestep) * flow10
        z1t = timestep * metric0
        z2t = (1 - timestep) * metric1
        if record is not None:
            record("F1t", f1t)
            record("F2t", f2t)
            record("Z1t", z1t)
            record("Z2t", z2t)

        img0_half = resize_bilinear(img0, img0.shape[2] // 2, img0.shape[3] // 2)
        img1_half = resize_bilinear(img1, img1.shape[2] // 2, img1.shape[3] // 2)

        i1t = warp(img0_half, f1t, z1t, strMode="soft")
        i2t = warp(img1_half, f2t, z2t, strMode="soft")
        if record is not None:
            record("splat_I1t", i1t)
            record("splat_I2t", i2t)

        feat1t1, feat2t1 = self._splat_pyramid_level(feat11, feat21, f1t, f2t, z1t, z2t, scale=1.0)
        if record is not None:
            record("splat_feat1t1", feat1t1)
            record("splat_feat2t1", feat2t1)

        feat1t2, feat2t2 = self._splat_pyramid_level(feat12, feat22, f1t, f2t, z1t, z2t, scale=0.5)
        if record is not None:
            record("splat_feat1t2", feat1t2)
            record("splat_feat2t2", feat2t2)

        feat1t3, feat2t3 = self._splat_pyramid_level(feat13, feat23, f1t, f2t, z1t, z2t, scale=0.25)
        if record is not None:
            record("splat_feat1t3", feat1t3)
            record("splat_feat2t3", feat2t3)

        fusion_rgb = torch.cat([img0_half, i1t, i2t, img1_half], dim=1)
        fusion_feat1 = torch.cat([feat1t1, feat2t1], dim=1)
        fusion_feat2 = torch.cat([feat1t2, feat2t2], dim=1)
        fusion_feat3 = torch.cat([feat1t3, feat2t3], dim=1)

        raw_out = self.fusionnet(fusion_rgb, fusion_feat1, fusion_feat2, fusion_feat3)
        if record is not None:
            record("fusionnet_out", raw_out)

        final = torch.clamp(raw_out, 0, 1)
        if record is not None:
            record("final_frame_padded", final)
        return final
