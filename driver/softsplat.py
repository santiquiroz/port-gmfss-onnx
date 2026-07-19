"""Softmax splatting (forward warping) -- numpy-in/numpy-out driver.

Derived exclusively from 98mxr/GMFSS_Fortuna's softsplat_torch.py (MIT,
vendored verbatim at toolkit/vendor/gmfss_fortuna_98mxr/softsplat_torch.py),
restricted to the one call pattern GMFSS_Fortuna's own pipeline actually
uses: strMode="soft" with no mode_sub suffix (softmax-weighted forward
warp, "+1e-7" normalization epsilon).

Control flow is rewritten so every tensor keeps a fixed shape end to end.
The original filters non-finite / out-of-bounds splat targets by boolean
masking, which changes array sizes mid-computation. This version instead
clamps out-of-range target indices into `[0, H)` / `[0, W)` and forces
their accumulation weight to exactly 0.0, so they still scatter into a
valid slot but contribute nothing -- numerically identical to being
dropped, but shape-stable (a prerequisite for a future fixed-buffer OpenCL
port; DirectML rejects ScatterND with reduction="add" outright, which is
why this needs a portable driver at all instead of running on DML).

This module depends on numpy and torch only (torch used CPU-side purely as
an index_add_ accumulator, which mirrors the vendored algorithm closely
enough to reach bit-exact parity on real data -- see tests/test_softsplat.py).
It has zero dependency on this repo's toolkit/, so it can be vendored
standalone into other projects as-is.
"""

from __future__ import annotations

import numpy as np
import torch

_NORMALIZE_EPS = 1e-7


def splat_softmax(
    tenIn: np.ndarray,
    tenFlow: np.ndarray,
    tenMetric: np.ndarray,
) -> np.ndarray:
    """Forward-warp tenIn by tenFlow, softmax-weighted+normalized by tenMetric.

    tenIn: [N,C,H,W], tenFlow: [N,2,H,W], tenMetric: [N,1,H,W] -> [N,C,H,W].
    Matches softsplat_torch.softsplat(tenIn, tenFlow, tenMetric, "soft").
    """
    ten_in = torch.from_numpy(np.ascontiguousarray(tenIn))
    ten_flow = torch.from_numpy(np.ascontiguousarray(tenFlow))
    ten_metric = torch.from_numpy(np.ascontiguousarray(tenMetric))

    splat_weight = ten_metric.exp()
    augmented_in = torch.cat([ten_in * splat_weight, splat_weight], dim=1)

    splatted = _forward_splat(augmented_in, ten_flow)

    normalizer = splatted[:, -1:, :, :] + _NORMALIZE_EPS
    out = splatted[:, :-1, :, :] / normalizer
    return out.numpy()


def _forward_splat(tenIn: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    n, c, h, w = tenIn.shape
    dtype = tenIn.dtype

    target_x, target_y = _flat_target_coordinates(tenFlow, h, w, dtype)
    batch_index = _flat_batch_index(n, h, w)
    source_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, c)

    finite = torch.isfinite(target_x) & torch.isfinite(target_y)
    safe_x, safe_y = _replace_nonfinite_with_zero(target_x, target_y, finite)

    corner_x = torch.floor(safe_x).to(torch.int64)
    corner_y = torch.floor(safe_y).to(torch.int64)

    out_flat = torch.zeros((n * h * w, c), dtype=dtype)
    for dx, dy, weight in _corner_weights(safe_x, safe_y, corner_x, corner_y):
        _accumulate_corner(
            out_flat=out_flat,
            source_flat=source_flat,
            batch_index=batch_index,
            idx_x=corner_x + dx,
            idx_y=corner_y + dy,
            weight=weight,
            in_frame=finite,
            h=h,
            w=w,
        )

    return out_flat.view(n, h, w, c).permute(0, 3, 1, 2)


def _flat_target_coordinates(
    tenFlow: torch.Tensor, h: int, w: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    n = tenFlow.shape[0]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(n, 1, h, w)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(n, 1, h, w)
    target_x = (grid_x + tenFlow[:, 0:1, :, :]).reshape(-1)
    target_y = (grid_y + tenFlow[:, 1:2, :, :]).reshape(-1)
    return target_x, target_y


def _flat_batch_index(n: int, h: int, w: int) -> torch.Tensor:
    return torch.arange(n).view(n, 1, 1).expand(n, h, w).reshape(-1)


def _replace_nonfinite_with_zero(
    target_x: torch.Tensor, target_y: torch.Tensor, finite: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # A non-finite target would make floor()/int-cast undefined; substitute a
    # finite dummy so shapes/dtypes stay well-formed, then zero its weight
    # below via `finite` so it never contributes -- same outcome as the
    # original's mask-based drop, without a data-dependent array size.
    zero_x = torch.zeros_like(target_x)
    zero_y = torch.zeros_like(target_y)
    safe_x = torch.where(finite, target_x, zero_x)
    safe_y = torch.where(finite, target_y, zero_y)
    return safe_x, safe_y


def _corner_weights(
    safe_x: torch.Tensor,
    safe_y: torch.Tensor,
    corner_x: torch.Tensor,
    corner_y: torch.Tensor,
) -> tuple[tuple[int, int, torch.Tensor], ...]:
    x0 = corner_x.to(safe_x.dtype)
    y0 = corner_y.to(safe_y.dtype)
    x1 = x0 + 1
    y1 = y0 + 1
    return (
        (0, 0, (x1 - safe_x) * (y1 - safe_y)),  # north-west
        (1, 0, (safe_x - x0) * (y1 - safe_y)),  # north-east
        (0, 1, (x1 - safe_x) * (safe_y - y0)),  # south-west
        (1, 1, (safe_x - x0) * (safe_y - y0)),  # south-east
    )


def _accumulate_corner(
    out_flat: torch.Tensor,
    source_flat: torch.Tensor,
    batch_index: torch.Tensor,
    idx_x: torch.Tensor,
    idx_y: torch.Tensor,
    weight: torch.Tensor,
    in_frame: torch.Tensor,
    h: int,
    w: int,
) -> None:
    in_bounds = in_frame & (idx_x >= 0) & (idx_x < w) & (idx_y >= 0) & (idx_y < h)
    clamped_x = idx_x.clamp(0, w - 1)
    clamped_y = idx_y.clamp(0, h - 1)
    masked_weight = torch.where(in_bounds, weight, torch.zeros_like(weight))

    linear_index = batch_index * h * w + clamped_y * w + clamped_x
    values = source_flat * masked_weight.unsqueeze(1)
    out_flat.index_add_(0, linear_index, values)
