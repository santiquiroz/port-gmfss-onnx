"""Valida cada grafo exportado (artifacts/*.onnx) contra sus tensores de referencia REALES
(refs/golden/, no randn) en CPUExecutionProvider y DmlExecutionProvider. Imprime rel-err por
salida y por caso, mas una tabla de timing CPU vs DML (patron de port-audiosr-onnx/toolkit/
validate_ort.py, adaptado a grafos multi-input/multi-output/multi-caso).

Umbral CPU-EP: rel-err < 1e-3 (gate duro, igual que AudioSR y que el brief de esta fase).
Umbral DirectML: rel-err < 1e-2. Estas tres redes son convnets feed-forward de una sola
pasada (sin acumulacion iterativa tipo diffusion/autoregresivo como el UNet de AudioSR), asi
que en teoria deberian estar mas cerca de CPU que un pipeline iterativo -- pero se mantiene el
umbral mas laxo de AudioSR por precedente: distintas GPUs/drivers DirectML pueden variar el
orden de reduccion en conv/grid_sample y no vale la pena que el gate sea fragil entre hardware.
Si DirectML falla (op no soportado, error de runtime) no es bloqueante: se reporta y se sigue
(igual que el precedente de AudioSR), el assert duro es solo sobre CPU-EP.

Uso: .venv/Scripts/python.exe toolkit/validate_ort.py [featurenet metricnet fusionnet]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
THIS_FILE = Path(__file__).resolve()

GRAPHS = ["featurenet", "metricnet", "fusionnet"]
CPU_REL_ERR_THRESHOLD = 1e-3
DML_REL_ERR_THRESHOLD = 1e-2
RESULT_LINE_PREFIX = "RESULT_JSON: "
# Defense-in-depth against *future* unknown hangs on other hardware/drivers (the known
# DXGI_ERROR_DEVICE_HUNG cause is already worked around via ORT_DISABLE_ALL in
# make_session). These are small feed-forward convnets -- CPU+DML together finish in low
# single-digit seconds per graph, so 300s is generous headroom, not a tight budget.
WORKER_TIMEOUT_SECONDS = 300


def discover_cases(name: str) -> list[str]:
    cases = []
    i = 0
    while (ART / f"{name}_case{i}_in0.npy").exists():
        cases.append(f"case{i}")
        i += 1
    return cases


def load_case(name: str, case_id: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inputs = []
    i = 0
    while (ART / f"{name}_{case_id}_in{i}.npy").exists():
        inputs.append(np.load(ART / f"{name}_{case_id}_in{i}.npy"))
        i += 1
    refs = []
    i = 0
    while (ART / f"{name}_{case_id}_ref{i}.npy").exists():
        refs.append(np.load(ART / f"{name}_{case_id}_ref{i}.npy"))
        i += 1
    return inputs, refs


def rel_err(actual: np.ndarray, expected: np.ndarray) -> float:
    denom = np.abs(expected).max()
    if denom == 0:
        return float(np.abs(actual - expected).max())
    return float(np.abs(actual - expected).max() / denom)


def feed_for(sess: ort.InferenceSession, inputs: list[np.ndarray]) -> dict:
    return {inp.name: arr for inp, arr in zip(sess.get_inputs(), inputs)}


def run_case(sess: ort.InferenceSession, case_id: str, inputs: list[np.ndarray], refs: list[np.ndarray]) -> list[float]:
    outputs = sess.run(None, feed_for(sess, inputs))
    errors = [rel_err(out.astype(np.float32), ref.astype(np.float32)) for out, ref in zip(outputs, refs)]
    print(f"    {case_id}: rel-err per output = {[f'{e:.6f}' for e in errors]}")
    return errors


def time_case(sess: ort.InferenceSession, inputs: list[np.ndarray], n_warmup: int = 1, n_timed: int = 3) -> float:
    feed = feed_for(sess, inputs)
    for _ in range(n_warmup):
        sess.run(None, feed)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)


def make_session(path: Path, providers: list[str]) -> ort.InferenceSession:
    """DirectML-specific: ORT's graph-fusion optimizer built a fused DML kernel for
    MetricNet (DmlFusedNode_2_11) that reproducibly hangs the GPU (DXGI_ERROR_DEVICE_HUNG,
    887A0006) after ~3 invocations on this hardware/driver -- disabling graph
    optimization for DML sessions avoids the fusion entirely, matches CPU-EP within the
    same rel-err, and costs no measurable speed (verified on featurenet/fusionnet too)."""
    sess_options = ort.SessionOptions()
    if "DmlExecutionProvider" in providers:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)


def validate_provider(name: str, providers: list[str], label: str, cases: list[str]) -> tuple[float, float]:
    path = ART / f"{name}.onnx"
    sess = make_session(path, providers)
    print(f"  [{label}]")
    max_err = 0.0
    for case_id in cases:
        inputs, refs = load_case(name, case_id)
        errors = run_case(sess, case_id, inputs, refs)
        max_err = max(max_err, max(errors))
    primary_inputs, _ = load_case(name, cases[0])
    ms = time_case(sess, primary_inputs)
    print(f"    timing (case0): {ms:.2f} ms")
    return max_err, ms


def validate_graph(name: str) -> dict:
    cases = discover_cases(name)
    if not cases:
        raise SystemExit(f"no validation cases found for {name} in {ART}")
    print(f"[{name}] cases={cases}")

    cpu_err, cpu_ms = validate_provider(name, ["CPUExecutionProvider"], "CPU-EP", cases)
    assert cpu_err < CPU_REL_ERR_THRESHOLD, f"{name}: CPU rel-err too high: {cpu_err}"

    dml_err: float | None
    dml_ms: float | None
    try:
        dml_err, dml_ms = validate_provider(name, ["DmlExecutionProvider"], "DirectML", cases)
        assert dml_err < DML_REL_ERR_THRESHOLD, f"{name}: DML rel-err too high: {dml_err}"
    except Exception as exc:  # noqa: BLE001
        print(f"  DirectML FAILED: {exc}")
        dml_err, dml_ms = None, None

    return {"name": name, "cpu_err": cpu_err, "cpu_ms": cpu_ms, "dml_err": dml_err, "dml_ms": dml_ms}


def print_summary_table(results: list[dict]) -> None:
    print("\n[summary]")
    header = f"{'graph':<12}{'CPU ms':>10}{'CPU rel-err':>14}{'DML ms':>10}{'DML rel-err':>14}{'speedup':>10}"
    print(header)
    for r in results:
        dml_ms = f"{r['dml_ms']:.2f}" if r["dml_ms"] is not None else "FAILED"
        dml_err = f"{r['dml_err']:.6f}" if r["dml_err"] is not None else "-"
        speedup = f"{r['cpu_ms'] / r['dml_ms']:.2f}x" if r["dml_ms"] else "-"
        print(f"{r['name']:<12}{r['cpu_ms']:>10.2f}{r['cpu_err']:>14.6f}{dml_ms:>10}{dml_err:>14}{speedup:>10}")


def run_worker(name: str) -> None:
    """Runs inside an isolated subprocess (see run_graph_isolated) and hands the
    result back to the parent as a single tagged JSON line after the normal logs."""
    result = validate_graph(name)
    print(RESULT_LINE_PREFIX + json.dumps(result))


def run_graph_isolated(name: str) -> dict:
    """Validates one graph in a fresh subprocess.

    Reproduced during Task 1.1: MetricNet's DirectML session hangs the GPU
    (DXGI_ERROR_DEVICE_HUNG / 887A0006) after ~3 invocations on this hardware/driver.
    Once that happens, onnxruntime silently falls back the *next* graph's DirectML
    session to CPUExecutionProvider inside the same process -- it only prints a
    warning, it does not raise -- which produced a false "DirectML PASS" (1.01x
    "speedup", i.e. actually CPU) for FusionNet in-process. A fresh process per graph
    is what actually prevents that cross-contamination.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(THIS_FILE), "--worker", name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{name}: worker subprocess hung and was killed after {WORKER_TIMEOUT_SECONDS}s "
            f"(likely an unrecovered DirectML device hang -- see make_session docstring)"
        ) from exc
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_LINE_PREFIX):
            result = json.loads(line[len(RESULT_LINE_PREFIX):])
        else:
            print(line)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if result is None:
        raise RuntimeError(f"{name}: worker subprocess produced no result (exit code {proc.returncode})")
    return result


def main() -> None:
    if "--worker" in sys.argv:
        name = sys.argv[sys.argv.index("--worker") + 1]
        run_worker(name)
        return

    names = sys.argv[1:] or [g for g in GRAPHS if (ART / f"{g}.onnx").exists()]
    results = [run_graph_isolated(name) for name in names]
    print_summary_table(results)


if __name__ == "__main__":
    main()
