"""Shared reconstruction of the pipeline's 8 real softsplat call sites from
refs/golden/, used by tests/test_softsplat.py (CPU bit-parity) and
tests/test_softsplat_cl.py (GPU kernel vs CPU reference). See
toolkit.gmfss_pg_pipeline for how each call site is derived from the
pipeline's forward()/_splat_pyramid_level().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from toolkit.gmfss_pg_pipeline import resize_bilinear

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "refs" / "golden"
PAIRS = ["vf_t006", "vs_t013", "vwarm_t019"]

CALL_SITE_TO_GOLDEN = {
    "I1t": "splat_I1t",
    "I2t": "splat_I2t",
    "feat1t1": "splat_feat1t1",
    "feat2t1": "splat_feat2t1",
    "feat1t2": "splat_feat1t2",
    "feat2t2": "splat_feat2t2",
    "feat1t3": "splat_feat1t3",
    "feat2t3": "splat_feat2t3",
}

REAL_CASES = [(pair, call_site) for pair in PAIRS for call_site in CALL_SITE_TO_GOLDEN]

_CALL_SITE_INPUT_NAMES = (
    "input_norm_img0",
    "input_norm_img1",
    "F1t",
    "F2t",
    "Z1t",
    "Z2t",
    "feat0_scale1",
    "feat1_scale1",
    "feat0_scale2",
    "feat1_scale2",
    "feat0_scale3",
    "feat1_scale3",
)


def golden_is_available() -> bool:
    required = _CALL_SITE_INPUT_NAMES + tuple(CALL_SITE_TO_GOLDEN.values())
    return all(_golden_path(pair, name).is_file() for pair in PAIRS for name in required)


def load_golden(pair: str, name: str) -> np.ndarray:
    return np.load(_golden_path(pair, name))


def _golden_path(pair: str, name: str) -> Path:
    return GOLDEN_DIR / pair / f"{name}.npy"


def _load_golden_tensor(pair: str, name: str) -> torch.Tensor:
    return torch.from_numpy(load_golden(pair, name))


def _resized_flow_metric(
    flow: torch.Tensor, metric: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirrors GMFSSBasePipeline._splat_pyramid_level's per-scale resize
    # formula exactly (flow * scale, metric unscaled) -- see
    # toolkit/gmfss_pg_pipeline.py.
    if scale == 1.0:
        return flow, metric
    h, w = flow.shape[2], flow.shape[3]
    new_h, new_w = int(h * scale), int(w * scale)
    flow_resized = resize_bilinear(flow, new_h, new_w) * scale
    metric_resized = resize_bilinear(metric, new_h, new_w)
    return flow_resized, metric_resized


def build_real_call_sites(pair: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reconstructs the exact tensors fed to each of the pipeline's 8 warp() calls.

    Two calls (I1t/I2t) use img0_half/img1_half (resize_bilinear of the dumped
    normalized frames) with F1t/F2t/Z1t/Z2t unmodified. Six calls (3 pyramid
    scales x 2 directions) use feat0_scale{1,2,3}/feat1_scale{1,2,3} with
    flow/metric resized per _splat_pyramid_level's formula (scale=1.0 is a
    no-op, matching the pipeline).
    """
    img0 = _load_golden_tensor(pair, "input_norm_img0")
    img1 = _load_golden_tensor(pair, "input_norm_img1")
    img0_half = resize_bilinear(img0, img0.shape[2] // 2, img0.shape[3] // 2)
    img1_half = resize_bilinear(img1, img1.shape[2] // 2, img1.shape[3] // 2)

    f1t = _load_golden_tensor(pair, "F1t")
    f2t = _load_golden_tensor(pair, "F2t")
    z1t = _load_golden_tensor(pair, "Z1t")
    z2t = _load_golden_tensor(pair, "Z2t")

    feat0_scale1 = _load_golden_tensor(pair, "feat0_scale1")
    feat1_scale1 = _load_golden_tensor(pair, "feat1_scale1")
    feat0_scale2 = _load_golden_tensor(pair, "feat0_scale2")
    feat1_scale2 = _load_golden_tensor(pair, "feat1_scale2")
    feat0_scale3 = _load_golden_tensor(pair, "feat0_scale3")
    feat1_scale3 = _load_golden_tensor(pair, "feat1_scale3")

    f1t_half, z1t_half = _resized_flow_metric(f1t, z1t, 0.5)
    f2t_half, z2t_half = _resized_flow_metric(f2t, z2t, 0.5)
    f1t_quarter, z1t_quarter = _resized_flow_metric(f1t, z1t, 0.25)
    f2t_quarter, z2t_quarter = _resized_flow_metric(f2t, z2t, 0.25)

    return {
        "I1t": (img0_half, f1t, z1t),
        "I2t": (img1_half, f2t, z2t),
        "feat1t1": (feat0_scale1, f1t, z1t),
        "feat2t1": (feat1_scale1, f2t, z2t),
        "feat1t2": (feat0_scale2, f1t_half, z1t_half),
        "feat2t2": (feat1_scale2, f2t_half, z2t_half),
        "feat1t3": (feat0_scale3, f1t_quarter, z1t_quarter),
        "feat2t3": (feat1_scale3, f2t_quarter, z2t_quarter),
    }
