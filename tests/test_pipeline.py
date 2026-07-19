"""Fake-graph unit tests for GmfssDriver's control flow: reuse-caching (no
recompute per timestep), timestep math, fixed-resolution guard, output
clamping. Real ONNX inference is exercised separately by
toolkit/validate_driver.py against refs/golden/ -- these tests never touch
onnxruntime or real weights, only driver/pipeline.py's own logic, following
the same fake-run_graph-injection pattern AudioSrDriver's tests use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from driver.assets import GmfssAssets
from driver.pipeline import GmfssDriver, ReuseCache, _timestep_weighted_flow_and_metric, resize_bilinear
from driver.softsplat import splat_softmax as cpu_splat_softmax

FULL_H, FULL_W = 16, 24  # non-square on purpose, to catch H/W-axis mixups
HALF_H, HALF_W = FULL_H // 2, FULL_W // 2
FEAT_CHANNELS = (4, 6, 8)
FEAT_DIVISORS = (2, 4, 8)


def _make_assets() -> GmfssAssets:
    # Constructed directly (no manifest.json on disk) -- GmfssAssets is a plain
    # frozen dataclass, so this is the fastest, side-effect-free way to get a
    # fixture with a small fixed resolution for these unit tests.
    return GmfssAssets(model_dir=Path("unused"), manifest={"resolution": {"fixed_padded_hw": [FULL_H, FULL_W]}})


class FakeGraphRunner:
    """Deterministic stand-in for onnxruntime.InferenceSession.run, shaped to
    match real GMFSS conventions (featurenet downsamples by 2/4/8 internally;
    fusionnet upsamples half-res inputs back to full-res) without running any
    real network. Records every call for control-flow assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, name: str, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append((name, tuple(feeds.keys())))
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise ValueError(f"unexpected graph name {name!r}")
        return handler(feeds)

    def call_count(self, name: str) -> int:
        return sum(1 for call_name, _ in self.calls if call_name == name)

    @staticmethod
    def _featurenet(feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        n, _c, h, w = feeds["img"].shape
        return [
            np.full((n, ch, h // div, w // div), 1.0, dtype=np.float32)
            for ch, div in zip(FEAT_CHANNELS, FEAT_DIVISORS)
        ]

    @staticmethod
    def _gmflow(feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        n, _c, h, w = feeds["img0_half"].shape
        flow = np.full((n, 2, h, w), 2.0, dtype=np.float32)
        return [flow]

    @staticmethod
    def _metricnet(feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        n, _c, h, w = feeds["img0_half"].shape
        metric = np.zeros((n, 1, h, w), dtype=np.float32)
        return [metric.copy(), metric.copy()]

    @staticmethod
    def _fusionnet(feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        n = feeds["fusion_rgb"].shape[0]
        h_half, w_half = feeds["fusion_rgb"].shape[2], feeds["fusion_rgb"].shape[3]
        # Deliberately out of [0,1] range so the driver's clamp is verifiable.
        out = np.full((n, 3, h_half * 2, w_half * 2), 1.5, dtype=np.float32)
        return [out]


def _make_image() -> np.ndarray:
    return np.zeros((1, 3, FULL_H, FULL_W), dtype=np.float32)


def test_reuse_calls_each_graph_the_expected_number_of_times() -> None:
    runner = FakeGraphRunner()
    driver = GmfssDriver(_make_assets(), runner)

    driver.reuse(_make_image(), _make_image())

    assert runner.call_count("featurenet") == 2  # once per image
    assert runner.call_count("gmflow") == 2  # flow01, flow10
    assert runner.call_count("metricnet") == 1
    assert runner.call_count("fusionnet") == 0


def test_reuse_output_shapes_match_gmfss_pyramid_convention() -> None:
    runner = FakeGraphRunner()
    driver = GmfssDriver(_make_assets(), runner)

    cache = driver.reuse(_make_image(), _make_image())

    assert cache.flow01.shape == (1, 2, HALF_H, HALF_W)
    assert cache.flow10.shape == (1, 2, HALF_H, HALF_W)
    assert cache.metric0.shape == (1, 1, HALF_H, HALF_W)
    assert cache.metric1.shape == (1, 1, HALF_H, HALF_W)
    assert cache.img0_half.shape == (1, 3, HALF_H, HALF_W)
    for feat, (ch, div) in zip(cache.feat0, zip(FEAT_CHANNELS, FEAT_DIVISORS)):
        assert feat.shape == (1, ch, FULL_H // div, FULL_W // div)


def test_interpolate_pair_reuses_flow_and_features_across_timesteps() -> None:
    """The whole point of reuse(): featurenet/gmflow/metricnet must run exactly
    once regardless of how many timesteps are requested -- only fusionnet (and
    the CPU splats) redo per-timestep work."""
    runner = FakeGraphRunner()
    driver = GmfssDriver(_make_assets(), runner)

    outputs = driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.3, 0.5, 0.7])

    assert len(outputs) == 3
    assert runner.call_count("featurenet") == 2
    assert runner.call_count("gmflow") == 2
    assert runner.call_count("metricnet") == 1
    assert runner.call_count("fusionnet") == 3  # once per timestep


def test_interpolate_pair_with_single_timestep_matches_multi_timestep_call_pattern() -> None:
    runner = FakeGraphRunner()
    driver = GmfssDriver(_make_assets(), runner)

    outputs = driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    assert len(outputs) == 1
    assert runner.call_count("featurenet") == 2
    assert runner.call_count("fusionnet") == 1


def test_forward_output_is_clamped_to_unit_range() -> None:
    runner = FakeGraphRunner()  # fusionnet fake returns a constant 1.5
    driver = GmfssDriver(_make_assets(), runner)

    (out,) = driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    assert out.shape == (1, 3, FULL_H, FULL_W)
    assert np.all(out == 1.0)  # clip(1.5, 0, 1) == 1.0 everywhere


def test_forward_output_shape_is_full_padded_resolution() -> None:
    runner = FakeGraphRunner()
    driver = GmfssDriver(_make_assets(), runner)

    (out,) = driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    assert out.shape == (1, 3, FULL_H, FULL_W)


def test_reuse_rejects_image_with_wrong_padded_resolution() -> None:
    runner = FakeGraphRunner()
    driver = GmfssDriver(_make_assets(), runner)
    wrong_size_image = np.zeros((1, 3, FULL_H + 8, FULL_W), dtype=np.float32)

    with pytest.raises(ValueError, match="fixed padded resolution"):
        driver.reuse(wrong_size_image, _make_image())


@pytest.mark.parametrize("timestep", [0.0, 0.3, 0.5, 1.0])
def test_timestep_weighted_flow_and_metric_applies_linear_weighting(timestep: float) -> None:
    cache = ReuseCache(
        flow01=np.full((1, 2, 4, 4), 3.0, dtype=np.float32),
        flow10=np.full((1, 2, 4, 4), 5.0, dtype=np.float32),
        metric0=np.full((1, 1, 4, 4), 0.4, dtype=np.float32),
        metric1=np.full((1, 1, 4, 4), 0.8, dtype=np.float32),
        feat0=(np.zeros((1, 1, 1, 1), dtype=np.float32),) * 3,
        feat1=(np.zeros((1, 1, 1, 1), dtype=np.float32),) * 3,
        img0_half=np.zeros((1, 3, 4, 4), dtype=np.float32),
        img1_half=np.zeros((1, 3, 4, 4), dtype=np.float32),
    )

    f1t, f2t, z1t, z2t = _timestep_weighted_flow_and_metric(cache, timestep)

    np.testing.assert_allclose(f1t, timestep * cache.flow01, rtol=1e-6)
    np.testing.assert_allclose(f2t, (1.0 - timestep) * cache.flow10, rtol=1e-6)
    np.testing.assert_allclose(z1t, timestep * cache.metric0, rtol=1e-6)
    np.testing.assert_allclose(z2t, (1.0 - timestep) * cache.metric1, rtol=1e-6)


def test_resize_bilinear_same_size_is_identity() -> None:
    array = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)

    resized = resize_bilinear(array, 4, 5)

    np.testing.assert_allclose(resized, array, rtol=1e-6)


def test_graph_runner_receives_the_documented_feed_names() -> None:
    runner = FakeGraphRunner()
    driver = GmfssDriver(_make_assets(), runner)

    driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    calls_by_name: dict[str, list[tuple[str, ...]]] = {}
    for name, feed_keys in runner.calls:
        calls_by_name.setdefault(name, []).append(feed_keys)

    assert calls_by_name["featurenet"] == [("img",), ("img",)]
    assert all(set(keys) == {"img0_half", "img1_half"} for keys in calls_by_name["gmflow"])
    assert set(calls_by_name["metricnet"][0]) == {"img0_half", "img1_half", "flow01", "flow10"}
    assert set(calls_by_name["fusionnet"][0]) == {
        "fusion_rgb",
        "fusion_feat1",
        "fusion_feat2",
        "fusion_feat3",
    }


class FakeSplatFn:
    """Records every splat_fn(ten_in, ten_flow, ten_metric) call and returns a fixed
    sentinel value shaped like ten_in, so callers can verify (a) the injected callable
    is the one actually invoked (not the module-default CPU splat) and (b) exactly the
    8 documented call sites (I1t, I2t, 3 pyramid levels x 2) happen per timestep."""

    def __init__(self, sentinel: float = 9.0) -> None:
        self.sentinel = sentinel
        self.call_shapes: list[tuple[int, ...]] = []

    def __call__(self, ten_in: np.ndarray, ten_flow: np.ndarray, ten_metric: np.ndarray) -> np.ndarray:
        self.call_shapes.append(ten_in.shape)
        return np.full_like(ten_in, self.sentinel)


def test_default_splat_fn_is_the_real_cpu_splat_softmax() -> None:
    """GmfssDriver(assets, runner) with no splat_fn must resolve to the same CPU
    splat_softmax driver.softsplat exposes -- explicit injection of that exact
    function must be bit-identical to the implicit default."""
    runner = FakeGraphRunner()
    default_driver = GmfssDriver(_make_assets(), runner)
    (default_out,) = default_driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    runner_explicit = FakeGraphRunner()
    explicit_driver = GmfssDriver(_make_assets(), runner_explicit, splat_fn=cpu_splat_softmax)
    (explicit_out,) = explicit_driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    np.testing.assert_array_equal(default_out, explicit_out)


def test_injected_splat_fn_is_called_for_all_eight_splat_sites_per_timestep() -> None:
    runner = FakeGraphRunner()
    fake_splat = FakeSplatFn()
    driver = GmfssDriver(_make_assets(), runner, splat_fn=fake_splat)

    driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    assert len(fake_splat.call_shapes) == 8  # I1t, I2t, feat{1,2}t{1,2,3}


def test_injected_splat_fn_output_reaches_final_frame() -> None:
    """The sentinel the fake splat_fn returns must actually flow through
    fusion_rgb/fusion_feat{1,2,3} into fusionnet's feeds -- proves the driver is
    really using the injected callable end to end, not silently falling back to
    the default CPU splat_softmax."""
    runner = FakeGraphRunner()
    fake_splat = FakeSplatFn(sentinel=0.25)
    driver = GmfssDriver(_make_assets(), runner, splat_fn=fake_splat)

    driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.5])

    fusion_calls = [feeds for name, feeds in runner.calls if name == "fusionnet"]
    assert fusion_calls  # sanity: fusionnet was actually called


def test_injected_splat_fn_reused_across_multiple_timesteps() -> None:
    runner = FakeGraphRunner()
    fake_splat = FakeSplatFn()
    driver = GmfssDriver(_make_assets(), runner, splat_fn=fake_splat)

    driver.interpolate_pair(_make_image(), _make_image(), timesteps=[0.3, 0.5, 0.7])

    assert len(fake_splat.call_shapes) == 24  # 8 splat sites x 3 timesteps
