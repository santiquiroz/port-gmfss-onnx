"""Converts the 4 exported ONNX graphs to fp16 via onnxconverter-common (Task 3.2's brief).

If the converter chokes on any graph (crash/hang/broken output on this hardware), that
graph stays fp32-only -- per the brief's explicit "si el converter se ahoga como con
AudioSR, medir solo fp32 y documentar" allowance (same precedent as port-audiosr-onnx's
UNet, which hit a real wall on a large dynamo-exported graph). This script does not force
the conversion through at all costs; a handled per-graph failure is a valid, documented
outcome, not a bug to work around.

keep_io_types=True: graph inputs/outputs stay float32 (onnxconverter-common inserts Cast
nodes at the boundary) so driver/pipeline.py's existing float32 feed/marshal code (see
`_f32` in driver/pipeline.py) works completely unchanged against fp16 graphs -- only
internal weights/activations convert. This is the standard mixed-precision mode and
requires zero driver-side code changes to try.

Known onnxconverter-common failure mode hit here on 3 of 4 graphs (featurenet, metricnet,
gmflow -- only fusionnet converts cleanly): when a tensor is BOTH a declared graph output
(kept float32 by keep_io_types) AND consumed by further internal computation downstream
(e.g. featurenet's "scale1" pyramid output also feeds later encoder blocks), the
converter's Cast-insertion bookkeeping leaves a stale/conflicting dtype on the internal
consumer's own value_info, which onnxruntime rejects at load time ("Type
(tensor(float16)) ... does not match expected type (tensor(float))"). Confirmed real (not
a shape-inference staleness this script can fix): re-running onnx.shape_inference.
infer_shapes() post-conversion does NOT resolve it (tried, still fails); confirmed the
narrower cause by converting with keep_io_types=False, which loads fine but requires
threading fp16 dtype through the ENTIRE driver end to end (resize_bilinear, both splat
backends, every graph boundary) -- a materially larger, riskier change than this task's
"try fp16, document either way" scope calls for. This script therefore verifies each
converted graph actually LOADS before keeping it, and deletes+reports-failed any graph
that doesn't, so a broken fp16 file never silently ships.

Usage: .venv/Scripts/python.exe toolkit/convert_fp16.py [featurenet metricnet fusionnet gmflow]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
FP16_DIR = ART / "fp16"

GRAPHS = ["featurenet", "metricnet", "fusionnet", "gmflow"]


def _verify_loadable(path: Path) -> None:
    """Raises if onnxruntime can't even load the graph (catches the stale-value_info
    failure mode this module's docstring documents) -- cheap CPU-EP load, no inference."""
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def convert_graph(name: str) -> bool:
    src = ART / f"{name}.onnx"
    if not src.exists():
        print(f"[fp16] {name}: SKIPPED, {src} not found (export it first)")
        return False

    print(f"[fp16] {name}: loading {src}", flush=True)
    model = onnx.load(str(src), load_external_data=True)

    t0 = time.perf_counter()
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    model_fp16 = onnx.shape_inference.infer_shapes(model_fp16)
    elapsed = time.perf_counter() - t0

    FP16_DIR.mkdir(exist_ok=True)
    dst = FP16_DIR / f"{name}.onnx"
    onnx.save(model_fp16, str(dst))

    try:
        _verify_loadable(dst)
    except Exception as exc:  # noqa: BLE001 -- reported to caller, file removed below
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"converted but onnxruntime refused to load it: {exc}") from exc

    size_mb = dst.stat().st_size / 1e6
    print(f"[fp16] {name}: converted+verified in {elapsed:.1f}s -> {dst} ({size_mb:.1f} MB)", flush=True)
    return True


def main() -> None:
    names = sys.argv[1:] or GRAPHS
    results: dict[str, bool] = {}
    for name in names:
        try:
            results[name] = convert_graph(name)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
            print(f"[fp16] {name}: CONVERSION FAILED: {exc}", flush=True)
            results[name] = False

    print("\n[fp16] summary:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED/SKIPPED (fp32-only, see module docstring)'}")


if __name__ == "__main__":
    main()
