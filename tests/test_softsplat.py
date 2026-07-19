"""Bit-parity + synthetic correctness tests for driver.softsplat.splat_softmax.

Ground truth is toolkit.gmfss_pg_pipeline.warp, which is a direct re-export of
the vendored gmfss_fortuna_98mxr.softsplat_torch.softsplat (MIT, see
toolkit/vendor/gmfss_fortuna_98mxr/softsplat_torch.py) -- the same function
that produced refs/golden/. This file freely imports from toolkit/ (unlike
driver/softsplat.py, which must stay standalone); see toolkit.gmfss_pg_pipeline
for how each of the 8 real call sites reconstructed here is derived from the
pipeline's forward()/_splat_pyramid_level().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from driver.softsplat import splat_softmax
from toolkit.gmfss_pg_pipeline import resize_bilinear, warp

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "refs" / "golden"
PAIRS = ["vf_t006", "vs_t013", "vwarm_t019"]
NORMALIZE_EPS = 1e-7


def _load_golden(pair: str, name: str) -> np.ndarray:
    return np.load(GOLDEN_DIR / pair / f"{name}.npy")


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


def _build_real_call_sites(pair: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reconstructs the exact tensors fed to each of the pipeline's 8 warp() calls.

    Two calls (I1t/I2t) use img0_half/img1_half (resize_bilinear of the dumped
    normalized frames) with F1t/F2t/Z1t/Z2t unmodified. Six calls (3 pyramid
    scales x 2 directions) use feat0_scale{1,2,3}/feat1_scale{1,2,3} with
    flow/metric resized per _splat_pyramid_level's formula (scale=1.0 is a
    no-op, matching the pipeline).
    """
    img0 = torch.from_numpy(_load_golden(pair, "input_norm_img0"))
    img1 = torch.from_numpy(_load_golden(pair, "input_norm_img1"))
    img0_half = resize_bilinear(img0, img0.shape[2] // 2, img0.shape[3] // 2)
    img1_half = resize_bilinear(img1, img1.shape[2] // 2, img1.shape[3] // 2)

    f1t = torch.from_numpy(_load_golden(pair, "F1t"))
    f2t = torch.from_numpy(_load_golden(pair, "F2t"))
    z1t = torch.from_numpy(_load_golden(pair, "Z1t"))
    z2t = torch.from_numpy(_load_golden(pair, "Z2t"))

    feat0_scale1 = torch.from_numpy(_load_golden(pair, "feat0_scale1"))
    feat1_scale1 = torch.from_numpy(_load_golden(pair, "feat1_scale1"))
    feat0_scale2 = torch.from_numpy(_load_golden(pair, "feat0_scale2"))
    feat1_scale2 = torch.from_numpy(_load_golden(pair, "feat1_scale2"))
    feat0_scale3 = torch.from_numpy(_load_golden(pair, "feat0_scale3"))
    feat1_scale3 = torch.from_numpy(_load_golden(pair, "feat1_scale3"))

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


@pytest.mark.parametrize("pair,call_site", REAL_CASES)
def test_real_call_site_bit_parity_vs_vendored(pair: str, call_site: str) -> None:
    sites = _build_real_call_sites(pair)
    ten_in, ten_flow, ten_metric = sites[call_site]

    ground_truth = warp(ten_in, ten_flow, ten_metric, strMode="soft").numpy()
    ours = splat_softmax(ten_in.numpy(), ten_flow.numpy(), ten_metric.numpy())

    assert ours.shape == ground_truth.shape
    assert np.array_equal(ours, ground_truth), (
        f"{pair}/{call_site}: max diff vs vendored softsplat_torch "
        f"= {np.abs(ours - ground_truth).max():.3e}"
    )


@pytest.mark.parametrize("pair,call_site", REAL_CASES)
def test_real_call_site_matches_phase0_golden_dump(pair: str, call_site: str) -> None:
    sites = _build_real_call_sites(pair)
    ten_in, ten_flow, ten_metric = sites[call_site]
    golden = _load_golden(pair, CALL_SITE_TO_GOLDEN[call_site])

    ours = splat_softmax(ten_in.numpy(), ten_flow.numpy(), ten_metric.numpy())

    assert np.array_equal(ours, golden), (
        f"{pair}/{call_site}: max diff vs refs/golden dump "
        f"= {np.abs(ours - golden).max():.3e}"
    )


def _run_vendored(tenIn: np.ndarray, tenFlow: np.ndarray, tenMetric: np.ndarray) -> np.ndarray:
    return warp(
        torch.from_numpy(tenIn), torch.from_numpy(tenFlow), torch.from_numpy(tenMetric),
        strMode="soft",
    ).numpy()


def test_zero_flow_is_identity_within_normalization_epsilon() -> None:
    ten_in = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32)
    ten_metric = np.array(
        [[[[0.0, 0.5, -0.5], [1.0, -1.0, 0.2], [0.3, -0.3, 0.0]]]], dtype=np.float32
    )
    ten_flow = np.zeros((1, 2, 3, 3), dtype=np.float32)

    weight = np.exp(ten_metric)
    expected = ten_in * weight / (weight + NORMALIZE_EPS)

    ours = splat_softmax(ten_in, ten_flow, ten_metric)

    np.testing.assert_allclose(ours, expected, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ours, _run_vendored(ten_in, ten_flow, ten_metric), atol=0, rtol=0)


def test_integer_flow_shifts_exactly_without_bilinear_blending() -> None:
    values = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    ten_metric = np.zeros((1, 1, 5, 5), dtype=np.float32)
    dx, dy = 1, 2
    ten_flow = np.zeros((1, 2, 5, 5), dtype=np.float32)
    ten_flow[:, 0, :, :] = dx
    ten_flow[:, 1, :, :] = dy

    expected = np.zeros_like(values)
    expected[:, :, dy:, dx:] = values[:, :, : 5 - dy, : 5 - dx]

    ours = splat_softmax(values, ten_flow, ten_metric)

    np.testing.assert_allclose(ours, expected, atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        ours, _run_vendored(values, ten_flow, ten_metric), atol=0, rtol=0
    )


def test_colliding_destinations_sum_softmax_weighted_contributions() -> None:
    ten_in = np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float32)
    ten_metric = np.array([[[[0.0, 1.0, 2.0]]]], dtype=np.float32)
    # All three source pixels (x=0,1,2) target destination x=1.
    ten_flow = np.array([[[[1.0, 0.0, -1.0]], [[0.0, 0.0, 0.0]]]], dtype=np.float32)

    weight = np.exp(ten_metric[0, 0])
    numerator = np.sum(ten_in[0, 0] * weight)
    denominator = np.sum(weight) + NORMALIZE_EPS
    expected_center = numerator / denominator

    ours = splat_softmax(ten_in, ten_flow, ten_metric)

    assert ours[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert ours[0, 0, 0, 1] == pytest.approx(expected_center, abs=1e-5)
    assert ours[0, 0, 0, 2] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(ours, _run_vendored(ten_in, ten_flow, ten_metric), atol=0, rtol=0)


def test_out_of_bounds_flow_drops_pixel_without_corrupting_neighbors() -> None:
    ten_in = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)
    ten_metric = np.zeros((1, 1, 1, 4), dtype=np.float32)
    ten_flow = np.zeros((1, 2, 1, 4), dtype=np.float32)
    ten_flow[:, 0, :, 2] = 1000.0  # pixel x=2 flies far outside the frame

    ours = splat_softmax(ten_in, ten_flow, ten_metric)

    assert np.all(np.isfinite(ours))
    np.testing.assert_allclose(ours[0, 0, 0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(ours[0, 0, 0, 1], 2.0, atol=1e-6)
    np.testing.assert_allclose(ours[0, 0, 0, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(ours[0, 0, 0, 3], 4.0, atol=1e-6)
    np.testing.assert_allclose(ours, _run_vendored(ten_in, ten_flow, ten_metric), atol=0, rtol=0)
