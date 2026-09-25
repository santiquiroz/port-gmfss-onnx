"""Bit-parity + synthetic correctness tests for driver.softsplat.splat_softmax.

Ground truth is toolkit.gmfss_pg_pipeline.warp, which is a direct re-export of
the vendored gmfss_fortuna_98mxr.softsplat_torch.softsplat (MIT, see
toolkit/vendor/gmfss_fortuna_98mxr/softsplat_torch.py) -- the same function
that produced refs/golden/. This file freely imports from toolkit/ (unlike
driver/softsplat.py, which must stay standalone); see toolkit.gmfss_pg_pipeline
for how each of the 8 real call sites reconstructed in tests/_golden_sites.py
is derived from the pipeline's forward()/_splat_pyramid_level().
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from driver.softsplat import splat_softmax
from tests._golden_sites import CALL_SITE_TO_GOLDEN, REAL_CASES, build_real_call_sites, load_golden
from toolkit.gmfss_pg_pipeline import warp

NORMALIZE_EPS = 1e-7


@pytest.mark.requires_golden
@pytest.mark.parametrize("pair,call_site", REAL_CASES)
def test_real_call_site_bit_parity_vs_vendored(pair: str, call_site: str) -> None:
    sites = build_real_call_sites(pair)
    ten_in, ten_flow, ten_metric = sites[call_site]

    ground_truth = warp(ten_in, ten_flow, ten_metric, strMode="soft").numpy()
    ours = splat_softmax(ten_in.numpy(), ten_flow.numpy(), ten_metric.numpy())

    assert ours.shape == ground_truth.shape
    assert np.array_equal(ours, ground_truth), (
        f"{pair}/{call_site}: max diff vs vendored softsplat_torch "
        f"= {np.abs(ours - ground_truth).max():.3e}"
    )


@pytest.mark.requires_golden
@pytest.mark.parametrize("pair,call_site", REAL_CASES)
def test_real_call_site_matches_phase0_golden_dump(pair: str, call_site: str) -> None:
    sites = build_real_call_sites(pair)
    ten_in, ten_flow, ten_metric = sites[call_site]
    golden = load_golden(pair, CALL_SITE_TO_GOLDEN[call_site])

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
