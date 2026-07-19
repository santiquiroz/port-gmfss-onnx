"""Reproduces this repo's central Task 3.2 finding: a lone GMFlow DirectML session is fast
(~600-700ms/call, matching Task 1.2's isolated `validate_ort.py` benchmark), but GMFlow's
own per-call latency is dramatically slower once featurenet/metricnet/fusionnet DirectML
sessions also exist (created + warmed) in the same process -- exactly what `GmfssDriver`'s
cached-session `run_graph` closure does across one real `interpolate_pair()` call, and
exactly what production (Phase 4) does by keeping all 4 graph sessions resident for reuse
across frames.

Why this exists: the original investigation behind this repo's README claim ("genuine
DirectML/driver-level GPU resource contention across coexisting sessions") was run from 3
throwaway scratchpad scripts, never committed to this repo -- flagged in review as the same
"ad hoc, uncommitted, one-off instrumented run" problem `toolkit/profile_pipeline.py`
already exists to fix for Task 2.2's GMFlow-percentage claim (see that module's docstring).
This script is the DML-contention equivalent: a minimal, focused, re-runnable reproduction
of the core comparison -- isolated fast vs. coexisting slow -- so the claim is backed by
code instead of prose.

Methodology (single process, same DML device, so the two phases are directly comparable --
not cross-run noise):

  1. ISOLATED: build ONLY a raw gmflow DirectML session (via toolkit/validate_ort.py's own
     `make_session`), warm it once, time N calls.
  2. COEXISTING: run the REAL `driver.pipeline.GmfssDriver.interpolate_pair()` on DirectML
     (fp32 everywhere, CPU splat -- the original investigation's baseline config, no fp16
     mitigation) via `toolkit/profile_pipeline.py`'s own already-established stage-timing
     machinery: one untimed warm-up call creates+uses all 4 DirectML sessions in the
     driver's real order (featurenet -> gmflow -> metricnet -> fusionnet), then a second,
     timed call measures GMFlow's own per-call latency with all 4 sessions resident and
     warm -- the real-world scenario a caller actually experiences.

Note on a discarded approach: an earlier version of phase 2 instead built raw sessions by
hand and re-timed the SAME already-warm gmflow session object after featurenet/metricnet/
fusionnet were created around it (mirroring the original investigation's own scratch-script
methodology as literally as possible). That ordering reproducibly crashed this process with
a segfault/access-violation on this hardware/driver -- on BOTH attempts, including a variant
matched to the original investigation's own description of a non-crashing ordering. Per this
task's own guidance not to force through or repeatedly risk that crash (it's a dead end, not
load-bearing for the contention theory), phase 2 was redesigned to get the same "coexisting"
number through the driver's real, already-tested call path instead of raw session-object
juggling. That the raw-poking approach crashes at all is itself corroborating evidence for
this repo's "and some fragility" language about DirectML multi-session behavior on this
hardware -- see README's "GMFlow discrepancy investigation" section.

Usage: .venv/Scripts/python.exe toolkit/investigate_dml_contention.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

sys.path.insert(0, str(ROOT))

from toolkit.validate_ort import make_session  # noqa: E402
from toolkit.validate_driver import load, PRIMARY_PAIR  # noqa: E402
from toolkit import profile_pipeline  # noqa: E402
from driver.pipeline import resize_bilinear  # noqa: E402

PROVIDERS = ["DmlExecutionProvider"]
N_WARMUP = 1
N_TIMED = 5


def _half_res(img: np.ndarray) -> tuple[int, int]:
    return img.shape[2] // 2, img.shape[3] // 2


def _gmflow_feeds() -> dict[str, np.ndarray]:
    img0 = load(PRIMARY_PAIR, "input_norm_img0").astype(np.float32)
    img1 = load(PRIMARY_PAIR, "input_norm_img1").astype(np.float32)
    img0_half = resize_bilinear(img0, *_half_res(img0))
    img1_half = resize_bilinear(img1, *_half_res(img1))
    return {"img0_half": img0_half, "img1_half": img1_half}


def _time_calls(sess, feeds: dict[str, np.ndarray], n_warmup: int, n_timed: int) -> list[float]:
    for _ in range(n_warmup):
        sess.run(None, feeds)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append(time.perf_counter() - t0)
    return times


def _fmt_ms(times: list[float]) -> str:
    ms = sorted(t * 1000 for t in times)
    mean = sum(ms) / len(ms)
    return f"{ms[0]:.1f}-{ms[-1]:.1f}ms/call (mean {mean:.1f}ms, n={len(ms)})"


def measure_isolated_gmflow() -> list[float]:
    print("[phase 1: ISOLATED] building only a gmflow DirectML session", flush=True)
    sess = make_session(ART / "gmflow.onnx", PROVIDERS)
    feeds = _gmflow_feeds()
    times = _time_calls(sess, feeds, N_WARMUP, N_TIMED)
    print(f"    gmflow, isolated: {_fmt_ms(times)}", flush=True)
    return times


def measure_coexisting_gmflow() -> float:
    """GMFlow's own per-call latency (ms) inside a real, fully-warmed
    `GmfssDriver.interpolate_pair()` run on DirectML (fp32, CPU splat -- no fp16
    mitigation), via `toolkit/profile_pipeline.py`'s existing stage-timing instrumentation.
    By the time this measures anything, featurenet/gmflow/metricnet/fusionnet DirectML
    sessions all already exist and are warm (created during profile_pipeline's own untimed
    warm-up call) -- this is the real coexisting-sessions scenario a caller experiences."""
    print(
        "\n[phase 2: COEXISTING] running GmfssDriver.interpolate_pair() on DirectML "
        "(fp32, CPU splat) -- warms featurenet/gmflow/metricnet/fusionnet DirectML "
        "sessions together, then times a second call with all 4 resident",
        flush=True,
    )
    timer, _total = profile_pipeline.profile_provider("dml", PRIMARY_PAIR, splat="cpu", fp16=False)
    gmflow_total = timer.totals["gmflow"]
    gmflow_count = timer.counts["gmflow"]
    per_call_ms = 1000 * gmflow_total / gmflow_count
    print(
        f"    gmflow, coexisting (inside real driver call, all 4 sessions resident): "
        f"{per_call_ms:.1f}ms/call (mean over {gmflow_count} call(s) in this pair)",
        flush=True,
    )
    return per_call_ms


def main() -> None:
    isolated = measure_isolated_gmflow()
    coexisting_per_call_ms = measure_coexisting_gmflow()

    isolated_mean_ms = 1000 * sum(isolated) / len(isolated)
    ratio = coexisting_per_call_ms / isolated_mean_ms if isolated_mean_ms else float("nan")

    print("\n[summary]")
    print(f"  isolated:   {_fmt_ms(isolated)}")
    print(f"  coexisting: {coexisting_per_call_ms:.1f}ms/call")
    print(f"  slowdown:   {ratio:.2f}x")


if __name__ == "__main__":
    main()
