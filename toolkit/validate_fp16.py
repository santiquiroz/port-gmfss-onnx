"""Validates the fp16 graphs (artifacts/fp16/*.onnx, produced by toolkit/convert_fp16.py)
against the same real golden-derived cases toolkit/validate_ort.py uses for fp32 --
reuses validate_ort.py's `validate_provider`/`discover_cases`/`load_case` directly
(model_dir parametrized), not reimplemented.

No hardcoded pass/fail assert here, unlike validate_ort.py's fp32 gate: fp16 legitimately
trades precision for speed, and this script's job is to report the real rel-err/timing
numbers so Task 3.2 can make an informed keep-or-drop call, not to enforce the same 1e-3/
1e-2 thresholds fp32 uses. DirectML is the only execution provider that matters here
(fp16 is a DML/GPU speed play; CPU-EP fp16 kernel coverage is poor and not the target).

Usage: .venv/Scripts/python.exe toolkit/validate_fp16.py [featurenet metricnet fusionnet gmflow]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FP16_DIR = ART / "fp16"

sys.path.insert(0, str(ROOT))

from toolkit.validate_ort import GRAPHS, discover_cases, validate_provider  # noqa: E402


def validate_fp16_graph(name: str) -> None:
    fp16_path = FP16_DIR / f"{name}.onnx"
    if not fp16_path.exists():
        print(f"[{name}] SKIPPED: {fp16_path} not found (fp16 conversion did not produce this graph)")
        return

    cases = discover_cases(name)
    if not cases:
        print(f"[{name}] SKIPPED: no validation cases found in {ART}")
        return

    print(f"[{name}] cases={cases}")

    fp32_err, fp32_ms = validate_provider(name, ["DmlExecutionProvider"], "DirectML fp32 (baseline)", cases, model_dir=ART)
    fp16_err, fp16_ms = validate_provider(name, ["DmlExecutionProvider"], "DirectML fp16", cases, model_dir=FP16_DIR)

    speedup = fp32_ms / fp16_ms if fp16_ms else float("nan")
    print(
        f"[{name}] summary: fp32 {fp32_ms:.2f}ms (rel-err {fp32_err:.6f})  "
        f"fp16 {fp16_ms:.2f}ms (rel-err {fp16_err:.6f})  speedup {speedup:.2f}x\n"
    )


def main() -> None:
    names = sys.argv[1:] or GRAPHS
    for name in names:
        try:
            validate_fp16_graph(name)
        except Exception as exc:  # noqa: BLE001 -- report and continue to the next graph
            print(f"[{name}] FAILED: {exc!r}\n")


if __name__ == "__main__":
    main()
