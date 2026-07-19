"""Isolated wall-clock benchmark of driver.softsplat_cl's OpenCL kernel
(Task 3.1), at the real tensor shapes the pipeline's 8 splat call sites use.

Times ONLY the GPU path (driver.softsplat_cl._splat_softmax_gpu), not the CPU
reference and not any ONNX graph -- this is the isolated-kernel bench the
brief asks for, separate from Task 3.2's whole-pipeline fps question. A
throwaway warm-up call runs first (untimed) so measurements don't pay
pyopencl/driver first-call lazy compilation cost, mirroring the existing
precedent in toolkit/profile_pipeline.py.

Usage:
  .venv/Scripts/python.exe toolkit/bench_splat_cl.py
  .venv/Scripts/python.exe toolkit/bench_splat_cl.py --iterations 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import driver.softsplat_cl as softsplat_cl  # noqa: E402

# (label, channels, height, width) -- the 8 real call sites collapse to 3
# distinct (channels, H, W) shapes at the "1t1/I" pyramid level (I1t/I2t
# share H,W with feat*t1 but fewer channels) plus 2 smaller pyramid levels.
# See artifacts/manifest.json splat_calls / refs/golden/*/*.npy shapes.
CALL_SHAPES = [
    ("I1t/I2t (img, half-res)", 3, 544, 960),
    ("feat*t1 (pyramid scale1, half-res)", 64, 544, 960),
    ("feat*t2 (pyramid scale2, quarter-res)", 128, 272, 480),
    ("feat*t3 (pyramid scale3, eighth-res)", 192, 136, 240),
]

WARMUP_ITERATIONS = 3


def _make_inputs(channels: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    ten_in = rng.standard_normal((1, channels, height, width)).astype(np.float32)
    ten_flow = (rng.standard_normal((1, 2, height, width)).astype(np.float32)) * 3.0
    ten_metric = rng.standard_normal((1, 1, height, width)).astype(np.float32)
    return ten_in, ten_flow, ten_metric


def _bench_shape(label: str, channels: int, height: int, width: int, iterations: int) -> None:
    context = softsplat_cl._get_gpu_context()
    if context is None:
        print(f"{label}: SKIPPED -- no working OpenCL GPU (see softsplat_cl fallback warning)")
        return

    ten_in, ten_flow, ten_metric = _make_inputs(channels, height, width)

    for _ in range(WARMUP_ITERATIONS):
        softsplat_cl._splat_softmax_gpu(context, ten_in, ten_flow, ten_metric)

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        softsplat_cl._splat_softmax_gpu(context, ten_in, ten_flow, ten_metric)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    mean_ms = sum(samples_ms) / len(samples_ms)
    median_ms = samples_ms[len(samples_ms) // 2]
    p95_ms = samples_ms[int(len(samples_ms) * 0.95)]
    verdict = "HIT" if mean_ms < 20.0 else "MISS"
    print(
        f"{label}: shape=(1,{channels},{height},{width}) "
        f"mean={mean_ms:.2f}ms median={median_ms:.2f}ms p95={p95_ms:.2f}ms "
        f"min={samples_ms[0]:.2f}ms max={samples_ms[-1]:.2f}ms "
        f"[{verdict} vs <20ms target]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    print(f"Benchmarking driver.softsplat_cl kernel, {args.iterations} iterations per shape "
          f"(+{WARMUP_ITERATIONS} untimed warmup)\n")
    for label, channels, height, width in CALL_SHAPES:
        _bench_shape(label, channels, height, width, args.iterations)


if __name__ == "__main__":
    main()
