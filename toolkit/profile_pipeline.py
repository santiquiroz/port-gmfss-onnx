"""Per-stage wall-clock profiling of GmfssDriver.interpolate_pair(), reusing
toolkit/validate_driver.py's make_run_graph() for warmed ONNX sessions (same
session-creation/caching used by the parity gate -- not reimplemented here).

Exists to fix a review finding on Task 2.2: the README's claim that GMFlow is
56-78% of total pipeline time (the root-cause explanation for measured fps
coming in below the plan's 0.2-0.6 estimate) came from an ad hoc, uncommitted,
one-off instrumented run -- not reproducible from any committed code. This
script IS that instrumentation, committed and re-runnable, so the claim is
backed by code instead of prose.

Times each stage GmfssDriver.interpolate_pair() calls internally:
  - featurenet (x2: img0, img1)
  - gmflow (x2: flow01, flow10)
  - metricnet (x1)
  - splat (x8: I1t/I2t + 3 feature-pyramid-level pairs, via driver.softsplat.splat_softmax)
  - fusionnet (x1 per requested timestep; 1 timestep here)
  - unaccounted: resize_bilinear/concatenate/timestep-weighting between stages

A throwaway warm-up call to interpolate_pair() runs first (untimed) so the
timed pass doesn't pay ONNX Runtime's first-call lazy kernel-compilation /
allocation cost -- especially real on DirectML. This mirrors the existing
precedent in validate_driver.py, where validate_stage_by_stage() already
exercises every graph before validate_end_to_end() measures timing.

Usage:
  .venv/Scripts/python.exe toolkit/profile_pipeline.py                # both providers
  .venv/Scripts/python.exe toolkit/profile_pipeline.py --provider cpu
  .venv/Scripts/python.exe toolkit/profile_pipeline.py --provider dml
  .venv/Scripts/python.exe toolkit/profile_pipeline.py --pair vs_t013
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

sys.path.insert(0, str(ROOT))

from toolkit.validate_driver import make_run_graph, load, PRIMARY_PAIR  # noqa: E402
from driver.assets import GmfssAssets  # noqa: E402
from driver import pipeline as pipeline_module  # noqa: E402
from driver.pipeline import GmfssDriver  # noqa: E402

PROVIDER_EPS = {
    "cpu": ["CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider"],
}
PROVIDER_LABELS = {"cpu": "CPU-EP", "dml": "DirectML"}
STAGE_ORDER = ("featurenet", "gmflow", "metricnet", "splat", "fusionnet")


class StageTimer:
    """Accumulates elapsed wall-clock seconds per named stage across one interpolate_pair() call."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def record(self, stage: str, elapsed: float) -> None:
        self.totals[stage] += elapsed
        self.counts[stage] += 1


def _timed_run_graph(run_graph, timer: StageTimer):
    def wrapped(name: str, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        start = time.perf_counter()
        result = run_graph(name, feeds)
        timer.record(name, time.perf_counter() - start)
        return result

    return wrapped


def _timed_splat_softmax(timer: StageTimer, original):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        timer.record("splat", time.perf_counter() - start)
        return result

    return wrapped


def profile_provider(provider: str, pair: str) -> tuple[StageTimer, float]:
    label = PROVIDER_LABELS[provider]
    run_graph = make_run_graph(PROVIDER_EPS[provider])
    assets = GmfssAssets.load(ART)

    img0 = load(pair, "input_norm_img0").astype(np.float32)
    img1 = load(pair, "input_norm_img1").astype(np.float32)

    warmup_driver = GmfssDriver(assets, run_graph)
    warmup_driver.interpolate_pair(img0, img1, timesteps=[0.5])

    timer = StageTimer()
    original_splat = pipeline_module.splat_softmax
    pipeline_module.splat_softmax = _timed_splat_softmax(timer, original_splat)
    try:
        driver = GmfssDriver(assets, _timed_run_graph(run_graph, timer))
        t0 = time.perf_counter()
        driver.interpolate_pair(img0, img1, timesteps=[0.5])
        total = time.perf_counter() - t0
    finally:
        pipeline_module.splat_softmax = original_splat

    print_breakdown(label, pair, timer, total)
    return timer, total


def print_breakdown(label: str, pair: str, timer: StageTimer, total: float) -> None:
    print(f"[{label}] per-stage breakdown, pair={pair} (interpolate_pair, timesteps=[0.5], warm session)")
    accounted = 0.0
    for stage in STAGE_ORDER:
        elapsed = timer.totals.get(stage, 0.0)
        count = timer.counts.get(stage, 0)
        accounted += elapsed
        pct = 100.0 * elapsed / total if total else 0.0
        print(f"    {stage:>12}: {elapsed:7.3f}s  ({pct:5.1f}%)  x{count} call(s)")
    unaccounted = total - accounted
    pct = 100.0 * unaccounted / total if total else 0.0
    print(f"    {'unaccounted':>12}: {unaccounted:7.3f}s  ({pct:5.1f}%)  (resize/concat/weighting between stages)")
    fps = 1.0 / total if total else 0.0
    print(f"    {'TOTAL':>12}: {total:7.3f}s  ({fps:.3f} fps)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["cpu", "dml", "both"], default="both")
    parser.add_argument("--pair", default=PRIMARY_PAIR)
    args = parser.parse_args()

    providers = ["cpu", "dml"] if args.provider == "both" else [args.provider]
    for provider in providers:
        profile_provider(provider, args.pair)


if __name__ == "__main__":
    main()
