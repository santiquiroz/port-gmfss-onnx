"""Assembled GMFSS_Fortuna driver: 4 injected ONNX graphs + an injectable softmax-splat
backend (Task 2.1's CPU `driver.softsplat.splat_softmax` by default, Task 3.1's GPU
`driver.softsplat_cl.splat_softmax` opt-in).

Composition mirrors `toolkit/gmfss_pg_pipeline.py`'s `GMFSSBasePipeline` exactly
(FeatureNet -> GMFlow x2 -> MetricNet -> softsplat x8 -> FusionNet), with two
swaps: PyTorch module calls become `run_graph(name, feeds)` calls against
injected ONNX sessions (real onnxruntime in production, fakes in tests --
same injection pattern as AudioSrDriver in image-upscaler-amd), and
`gmfss_pg_pipeline.warp()` becomes an injected `splat_fn` (see `SplatFn` below).
See `artifacts/manifest.json` for the graph I/O contract and splat-call ordering
this file implements.

`GmfssDriver(assets, run_graph, splat_fn=None)` -- `splat_fn=None` (the default)
resolves to this module's own `splat_softmax` name (CPU, imported from
`driver.softsplat`) looked up dynamically on every call, which is what lets
`toolkit/profile_pipeline.py` keep monkeypatching `pipeline.splat_softmax` for
per-stage timing without any change here. Passing `splat_fn=driver.softsplat_cl.
splat_softmax` (or any callable with the same signature) opts into the GPU
backend -- same contract, same call sites, only the accumulation implementation
changes. Backward compatible: every existing `GmfssDriver(assets, run_graph)`
call site keeps using CPU splat exactly as before.

Like `driver/softsplat.py`, this module has zero dependency on `toolkit/` --
it can be vendored standalone into other projects (e.g. Upflow) as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from driver.assets import GmfssAssets
from driver.softsplat import splat_softmax

FeaturePyramid = tuple[np.ndarray, np.ndarray, np.ndarray]


class GraphRunner(Protocol):
    """Runs one ONNX graph by name, feeds by input name, returns outputs in the
    graph's declared output order (see artifacts/manifest.json `graphs.*.outputs`).
    Unlike AudioSrDriver's single-output convention, featurenet/metricnet have
    multiple outputs, so this always returns a list rather than one array."""

    def __call__(self, name: str, feeds: dict[str, np.ndarray]) -> list[np.ndarray]: ...


class SplatFn(Protocol):
    """Same contract as `driver.softsplat.splat_softmax`/`driver.softsplat_cl.splat_softmax`:
    forward-warp tenIn [N,C,H,W] by tenFlow [N,2,H,W], softmax-weighted+normalized by
    tenMetric [N,1,H,W] -> [N,C,H,W]. Both implementations are drop-in interchangeable."""

    def __call__(
        self, ten_in: np.ndarray, ten_flow: np.ndarray, ten_metric: np.ndarray
    ) -> np.ndarray: ...


def resize_bilinear(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """np.ndarray [N,C,H,W] -> resized [N,C,height,width], stretch (not pad) bilinear,
    align_corners=False. Bit-for-bit the same call as `toolkit/gmfss_pg_pipeline.py`'s
    `resize_bilinear` (torch.nn.functional.interpolate) -- reimplemented as a tiny
    local torch call (numpy in/out) so driver/ keeps zero `toolkit/` imports, matching
    `driver/softsplat.py`'s own numpy-in/numpy-out-via-torch-CPU pattern."""
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    resized = F.interpolate(tensor, (height, width), mode="bilinear", align_corners=False)
    return resized.numpy()


def _f32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


@dataclass(frozen=True)
class ReuseCache:
    """Per-pair, timestep-independent outputs -- mirrors GMFSSBasePipeline.reuse().

    img0_half/img1_half are not part of upstream reuse()'s return value (it
    recomputes them again inside forward()), but caching them here is a pure
    optimization: resize_bilinear is deterministic, so precomputing once changes
    no numeric result while satisfying the "no recompute per timestep" requirement.
    """

    flow01: np.ndarray
    flow10: np.ndarray
    metric0: np.ndarray
    metric1: np.ndarray
    feat0: FeaturePyramid
    feat1: FeaturePyramid
    img0_half: np.ndarray
    img1_half: np.ndarray


class GmfssDriver:
    def __init__(
        self,
        assets: GmfssAssets,
        run_graph: GraphRunner,
        splat_fn: SplatFn | None = None,
    ) -> None:
        self.assets = assets
        self.run_graph = run_graph
        self.splat_fn = splat_fn

    def _resolve_splat_fn(self) -> SplatFn:
        # `splat_softmax` here is a free variable resolved against this module's
        # globals at CALL time (not bound at __init__ time) -- this is what lets
        # toolkit/profile_pipeline.py's `pipeline_module.splat_softmax = ...`
        # monkeypatch keep working for the default (splat_fn=None) path.
        return self.splat_fn if self.splat_fn is not None else splat_softmax

    def interpolate_pair(
        self, img0: np.ndarray, img1: np.ndarray, timesteps: list[float]
    ) -> list[np.ndarray]:
        """Runs reuse() exactly once regardless of len(timesteps); GMFSS's flow/feature
        extraction is the expensive part (10-15x slower than RIFE), so amortizing it
        across every requested intermediate frame is the point of this split."""
        cache = self.reuse(img0, img1)
        return [self._forward_at_timestep(cache, timestep) for timestep in timesteps]

    def reuse(self, img0: np.ndarray, img1: np.ndarray) -> ReuseCache:
        """Per-pair-independent-of-timestep computation: FeatureNet on both images,
        GMFlow in both directions, MetricNet. Mirrors GMFSSBasePipeline.reuse()."""
        self._assert_fixed_resolution(img0)
        self._assert_fixed_resolution(img1)

        feat0 = self._extract_features(img0)
        feat1 = self._extract_features(img1)

        half_h, half_w = img0.shape[2] // 2, img0.shape[3] // 2
        img0_half = resize_bilinear(img0, half_h, half_w)
        img1_half = resize_bilinear(img1, half_h, half_w)

        flow01, flow10 = self._estimate_flow(img0_half, img1_half)
        metric0, metric1 = self._estimate_metric(img0_half, img1_half, flow01, flow10)

        return ReuseCache(
            flow01=flow01,
            flow10=flow10,
            metric0=metric0,
            metric1=metric1,
            feat0=feat0,
            feat1=feat1,
            img0_half=img0_half,
            img1_half=img1_half,
        )

    def _assert_fixed_resolution(self, img: np.ndarray) -> None:
        expected = self.assets.padded_hw
        actual = (img.shape[2], img.shape[3])
        if actual != expected:
            raise ValueError(
                f"GmfssDriver expects fixed padded resolution {expected} "
                f"(see manifest.json resolution.fixed_padded_hw); got {actual}. "
                "Padding/cropping arbitrary input resolutions is out of scope for "
                "this driver -- that's Phase 4's job when it's wired into the "
                "real video pipeline."
            )

    def _extract_features(self, img: np.ndarray) -> FeaturePyramid:
        scale1, scale2, scale3 = self.run_graph("featurenet", {"img": _f32(img)})
        return scale1, scale2, scale3

    def _estimate_flow(
        self, img0_half: np.ndarray, img1_half: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        flow01 = self.run_graph(
            "gmflow", {"img0_half": _f32(img0_half), "img1_half": _f32(img1_half)}
        )[0]
        flow10 = self.run_graph(
            "gmflow", {"img0_half": _f32(img1_half), "img1_half": _f32(img0_half)}
        )[0]
        return flow01, flow10

    def _estimate_metric(
        self,
        img0_half: np.ndarray,
        img1_half: np.ndarray,
        flow01: np.ndarray,
        flow10: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        metric0, metric1 = self.run_graph(
            "metricnet",
            {
                "img0_half": _f32(img0_half),
                "img1_half": _f32(img1_half),
                "flow01": _f32(flow01),
                "flow10": _f32(flow10),
            },
        )
        return metric0, metric1

    def _forward_at_timestep(self, cache: ReuseCache, timestep: float) -> np.ndarray:
        splat_fn = self._resolve_splat_fn()
        f1t, f2t, z1t, z2t = _timestep_weighted_flow_and_metric(cache, timestep)

        i1t = splat_fn(cache.img0_half, f1t, z1t)
        i2t = splat_fn(cache.img1_half, f2t, z2t)

        feat11, feat12, feat13 = cache.feat0
        feat21, feat22, feat23 = cache.feat1

        feat1t1, feat2t1 = _splat_pyramid_level(feat11, feat21, f1t, f2t, z1t, z2t, scale=1.0, splat_fn=splat_fn)
        feat1t2, feat2t2 = _splat_pyramid_level(feat12, feat22, f1t, f2t, z1t, z2t, scale=0.5, splat_fn=splat_fn)
        feat1t3, feat2t3 = _splat_pyramid_level(feat13, feat23, f1t, f2t, z1t, z2t, scale=0.25, splat_fn=splat_fn)

        fusion_rgb = np.concatenate([cache.img0_half, i1t, i2t, cache.img1_half], axis=1)
        fusion_feat1 = np.concatenate([feat1t1, feat2t1], axis=1)
        fusion_feat2 = np.concatenate([feat1t2, feat2t2], axis=1)
        fusion_feat3 = np.concatenate([feat1t3, feat2t3], axis=1)

        raw_out = self.run_graph(
            "fusionnet",
            {
                "fusion_rgb": _f32(fusion_rgb),
                "fusion_feat1": _f32(fusion_feat1),
                "fusion_feat2": _f32(fusion_feat2),
                "fusion_feat3": _f32(fusion_feat3),
            },
        )[0]

        return np.clip(raw_out, 0.0, 1.0)


def _timestep_weighted_flow_and_metric(
    cache: ReuseCache, timestep: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.float32(timestep)
    f1t = t * cache.flow01
    f2t = (np.float32(1.0) - t) * cache.flow10
    z1t = t * cache.metric0
    z2t = (np.float32(1.0) - t) * cache.metric1
    return f1t, f2t, z1t, z2t


def _splat_pyramid_level(
    feat0: np.ndarray,
    feat1: np.ndarray,
    flow0t: np.ndarray,
    flow1t: np.ndarray,
    z0t: np.ndarray,
    z1t: np.ndarray,
    scale: float,
    splat_fn: SplatFn,
) -> tuple[np.ndarray, np.ndarray]:
    """One of the 3 feature-pyramid splat pairs (6 of the 8 total splat calls).
    Mirrors GMFSSBasePipeline._splat_pyramid_level: at scale=1.0 the flow/metric
    already match the feature map's resolution (both are img_half-sized); at
    0.5/0.25 the flow is bilinear-resized AND rescaled by the same factor (a
    flow vector's magnitude is in pixels of its own resolution), while the
    metric is resized without rescaling (it's a log-weight, not a displacement)."""
    if scale != 1.0:
        flow0t = _resize_flow(flow0t, scale)
        flow1t = _resize_flow(flow1t, scale)
        z0t = _resize_metric(z0t, scale)
        z1t = _resize_metric(z1t, scale)
    splat0 = splat_fn(feat0, flow0t, z0t)
    splat1 = splat_fn(feat1, flow1t, z1t)
    return splat0, splat1


def _resize_flow(flow: np.ndarray, scale: float) -> np.ndarray:
    height, width = int(flow.shape[2] * scale), int(flow.shape[3] * scale)
    return resize_bilinear(flow, height, width) * scale


def _resize_metric(metric: np.ndarray, scale: float) -> np.ndarray:
    height, width = int(metric.shape[2] * scale), int(metric.shape[3] * scale)
    return resize_bilinear(metric, height, width)
