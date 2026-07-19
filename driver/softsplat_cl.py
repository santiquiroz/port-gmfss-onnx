"""OpenCL GPU backend for softmax splatting -- drop-in alternative to
`driver.softsplat.splat_softmax` (same numpy in/out contract, same
public function name so callers can swap the import).

The core bilinear scatter-add runs as a hand-written OpenCL kernel
(driver/kernels/splat.cl) using an atomic_cmpxchg CAS-loop for float
accumulation -- the target AMD driver (RX 7800 XT, OpenCL 2.1) exposes no
native float-atomic extension, only the standard int32 atomics. Softmax
weighting (exp(metric)) and post-scatter normalization are done in numpy
around the kernel call, mirroring how driver/softsplat.py structures the
CPU version.

pyopencl is an OPTIONAL dependency (see toolkit/requirements-gpu-splat.txt).
If it is not importable, or the kernel fails to compile/run on whatever
OpenCL device is available, every call transparently falls back to
`driver.softsplat.splat_softmax` (the bit-exact CPU reference) after
printing a warning exactly once -- not once per call.

Like driver/softsplat.py, this module depends only on numpy (+ optional
pyopencl) and driver.softsplat, so it can be vendored standalone.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from driver.softsplat import splat_softmax as _splat_softmax_cpu

_KERNEL_PATH = Path(__file__).resolve().parent / "kernels" / "splat.cl"
_NORMALIZE_EPS = 1e-7
_LOCAL_WORK_SIZE = 256

_warned_once = False
_gpu_unavailable = False
_gpu_context: Optional["_GpuContext"] = None


class _GpuContext:
    def __init__(self, ctx, queue, kernel) -> None:
        self.ctx = ctx
        self.queue = queue
        self.kernel = kernel


def splat_softmax(
    tenIn: np.ndarray,
    tenFlow: np.ndarray,
    tenMetric: np.ndarray,
) -> np.ndarray:
    """Forward-warp tenIn by tenFlow, softmax-weighted+normalized by tenMetric.

    Same contract as driver.softsplat.splat_softmax: tenIn [N,C,H,W],
    tenFlow [N,2,H,W], tenMetric [N,1,H,W] -> [N,C,H,W]. Runs on the OpenCL
    GPU kernel when available (rel-err < 1e-5 vs the CPU reference, due to
    GPU floating-point accumulation order); falls back to the CPU
    implementation transparently otherwise.
    """
    context = _get_gpu_context()
    if context is None:
        return _splat_softmax_cpu(tenIn, tenFlow, tenMetric)

    try:
        return _splat_softmax_gpu(context, tenIn, tenFlow, tenMetric)
    except Exception as exc:  # pragma: no cover - depends on real driver failures
        _disable_gpu(f"kernel run failed: {exc!r}")
        return _splat_softmax_cpu(tenIn, tenFlow, tenMetric)


def _get_gpu_context() -> Optional[_GpuContext]:
    global _gpu_context
    if _gpu_unavailable:
        return None
    if _gpu_context is None:
        _gpu_context = _build_gpu_context()
    return _gpu_context


def _build_gpu_context() -> Optional[_GpuContext]:
    try:
        import pyopencl as cl
    except ImportError:
        _disable_gpu("pyopencl is not installed")
        return None

    try:
        device = _select_device(cl)
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        source = _KERNEL_PATH.read_text(encoding="utf-8")
        program = cl.Program(ctx, source).build()
        return _GpuContext(ctx, queue, program.splat_scatter_add)
    except Exception as exc:  # pragma: no cover - depends on real driver failures
        _disable_gpu(f"OpenCL init/compile failed: {exc!r}")
        return None


def _select_device(cl):
    """Picks the GPU device with the most compute units across every OpenCL
    platform, de-duplicating identical (vendor, name) entries -- this
    machine registers the same AMD driver under two platforms, both
    exposing the same devices (integrated + RX 7800 XT)."""
    seen: set[tuple[str, str]] = set()
    candidates = []
    for platform in cl.get_platforms():
        for device in platform.get_devices(device_type=cl.device_type.GPU):
            key = (device.vendor, device.name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(device)
    if not candidates:
        raise RuntimeError("no OpenCL GPU device found")
    return max(candidates, key=lambda d: d.max_compute_units)


def _disable_gpu(reason: str) -> None:
    global _gpu_unavailable, _gpu_context
    _gpu_unavailable = True
    _gpu_context = None
    _warn_fallback(reason)


def _warn_fallback(reason: str) -> None:
    global _warned_once
    if _warned_once:
        return
    _warned_once = True
    warnings.warn(
        f"softsplat_cl: OpenCL GPU splat unavailable ({reason}); "
        "falling back to CPU splat_softmax for all calls.",
        RuntimeWarning,
        stacklevel=3,
    )


def _splat_softmax_gpu(
    context: _GpuContext,
    tenIn: np.ndarray,
    tenFlow: np.ndarray,
    tenMetric: np.ndarray,
) -> np.ndarray:
    ten_in = np.ascontiguousarray(tenIn, dtype=np.float32)
    ten_flow = np.ascontiguousarray(tenFlow, dtype=np.float32)
    ten_metric = np.ascontiguousarray(tenMetric, dtype=np.float32)
    n, c, h, w = ten_in.shape

    splat_weight = np.exp(ten_metric)
    augmented_in = np.ascontiguousarray(
        np.concatenate([ten_in * splat_weight, splat_weight], axis=1)
    )
    channels = c + 1

    out_host = np.zeros((n, channels, h, w), dtype=np.float32)
    _run_kernel(context, augmented_in, ten_flow, out_host, n, channels, h, w)

    normalizer = out_host[:, -1:, :, :] + _NORMALIZE_EPS
    return out_host[:, :-1, :, :] / normalizer


def _run_kernel(
    context: _GpuContext,
    augmented_in: np.ndarray,
    ten_flow: np.ndarray,
    out_host: np.ndarray,
    n: int,
    channels: int,
    h: int,
    w: int,
) -> None:
    import pyopencl as cl

    mf = cl.mem_flags
    source_buf = cl.Buffer(context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=augmented_in)
    flow_buf = cl.Buffer(context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ten_flow)
    out_buf = cl.Buffer(context.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out_host)

    global_size = (_round_up(n * h * w, _LOCAL_WORK_SIZE),)
    context.kernel(
        context.queue,
        global_size,
        (_LOCAL_WORK_SIZE,),
        source_buf,
        flow_buf,
        out_buf,
        np.int32(n),
        np.int32(channels),
        np.int32(h),
        np.int32(w),
    )
    cl.enqueue_copy(context.queue, out_host, out_buf)
    context.queue.finish()


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
