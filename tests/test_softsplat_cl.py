"""Correctness + fallback tests for driver.softsplat_cl.splat_softmax (Task 3.1).

Reuses tests/test_softsplat.py's real-call-site reconstruction (same 8 call
sites x 3 golden pairs = 24 combinations that Task 2.1's suite already
derives from refs/golden/ -- see that file's _build_real_call_sites for how
each tensor triple is assembled from the pipeline's forward()).

Ground truth here is driver.softsplat.splat_softmax (the CPU reference,
already proven bit-exact vs the vendored PyTorch implementation), not the
vendored torch module directly -- this suite is only responsible for
validating the GPU kernel against the CPU driver it's meant to replace.

Tolerance: whole-tensor L2 (Frobenius-norm) relative error < 1e-5 per call
site, not a per-element max-relative-error. Per-element relative error is
undefined/explosive at elements whose CPU reference value is near zero
(division by ~0), which several feature-pyramid channels legitimately are --
the same outlier-domination problem this repo's README already documents for
end-to-end rel-err (see "End-to-end is gated on RMS-rel-err, not
max-abs-rel-err" in README.md). L2-relative-error is the standard tolerant
metric for comparing two accumulation orders of the same sum and is what the
brief's "tolerancia float por orden de acumulacion" is about.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import driver.softsplat_cl as softsplat_cl
from driver.softsplat import splat_softmax as cpu_splat_softmax
from driver.softsplat_cl import splat_softmax as gpu_splat_softmax
from tests.test_softsplat import CALL_SITE_TO_GOLDEN, PAIRS, _build_real_call_sites

REAL_CASES = [(pair, call_site) for pair in PAIRS for call_site in CALL_SITE_TO_GOLDEN]
REL_ERR_TOLERANCE = 1e-5


def _l2_relative_error(gpu_out: np.ndarray, cpu_out: np.ndarray) -> float:
    diff = gpu_out.astype(np.float64) - cpu_out.astype(np.float64)
    cpu_norm = np.linalg.norm(cpu_out.astype(np.float64))
    return float(np.linalg.norm(diff) / (cpu_norm + 1e-12))


@pytest.fixture(autouse=True)
def _require_real_gpu_for_correctness_tests(request):
    """Correctness tests need the actual OpenCL kernel to run, not the CPU
    fallback (which would trivially match itself). Skip instead of silently
    passing if this machine has no working OpenCL GPU."""
    if request.node.get_closest_marker("gpu_correctness") is None:
        return
    context = softsplat_cl._get_gpu_context()
    if context is None:
        pytest.skip("no working OpenCL GPU on this machine -- see softsplat_cl fallback warning")


@pytest.mark.gpu_correctness
@pytest.mark.parametrize("pair,call_site", REAL_CASES)
def test_gpu_kernel_matches_cpu_reference_within_rel_err(pair: str, call_site: str) -> None:
    sites = _build_real_call_sites(pair)
    ten_in, ten_flow, ten_metric = sites[call_site]
    ten_in, ten_flow, ten_metric = ten_in.numpy(), ten_flow.numpy(), ten_metric.numpy()

    cpu_out = cpu_splat_softmax(ten_in, ten_flow, ten_metric)
    gpu_out = gpu_splat_softmax(ten_in, ten_flow, ten_metric)

    assert gpu_out.shape == cpu_out.shape
    rel_err = _l2_relative_error(gpu_out, cpu_out)
    assert rel_err < REL_ERR_TOLERANCE, (
        f"{pair}/{call_site}: L2 rel-err {rel_err:.3e} >= {REL_ERR_TOLERANCE:.0e} "
        f"(max abs diff {np.abs(gpu_out - cpu_out).max():.3e})"
    )


@pytest.fixture
def _reset_gpu_module_state():
    """Snapshots and restores driver.softsplat_cl's module-level GPU cache
    (context/unavailable-flag/warned-once-flag) so fallback tests don't leak
    state into each other or into the correctness tests above."""
    saved = (
        softsplat_cl._gpu_context,
        softsplat_cl._gpu_unavailable,
        softsplat_cl._warned_once,
    )
    softsplat_cl._gpu_context = None
    softsplat_cl._gpu_unavailable = False
    softsplat_cl._warned_once = False
    yield
    softsplat_cl._gpu_context, softsplat_cl._gpu_unavailable, softsplat_cl._warned_once = saved


def _tiny_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    ten_in = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    ten_flow = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    ten_metric = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    return ten_in, ten_flow, ten_metric


def test_falls_back_to_cpu_when_pyopencl_unimportable(monkeypatch, _reset_gpu_module_state) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pyopencl", None)  # forces `import pyopencl` -> ImportError

    ten_in, ten_flow, ten_metric = _tiny_inputs()
    expected = cpu_splat_softmax(ten_in, ten_flow, ten_metric)

    with pytest.warns(RuntimeWarning, match="pyopencl is not installed"):
        result = gpu_splat_softmax(ten_in, ten_flow, ten_metric)

    np.testing.assert_array_equal(result, expected)
    assert softsplat_cl._gpu_unavailable is True


def test_missing_pyopencl_warns_exactly_once_across_calls(monkeypatch, _reset_gpu_module_state) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pyopencl", None)
    ten_in, ten_flow, ten_metric = _tiny_inputs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gpu_splat_softmax(ten_in, ten_flow, ten_metric)
        gpu_splat_softmax(ten_in, ten_flow, ten_metric)
        gpu_splat_softmax(ten_in, ten_flow, ten_metric)

    fallback_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(fallback_warnings) == 1


def test_falls_back_to_cpu_when_kernel_fails_to_compile(
    monkeypatch, tmp_path, _reset_gpu_module_state
) -> None:
    pytest.importorskip("pyopencl")

    broken_kernel = tmp_path / "broken.cl"
    broken_kernel.write_text("this is not valid OpenCL C {{{", encoding="utf-8")
    monkeypatch.setattr(softsplat_cl, "_KERNEL_PATH", broken_kernel)

    ten_in, ten_flow, ten_metric = _tiny_inputs()
    expected = cpu_splat_softmax(ten_in, ten_flow, ten_metric)

    with pytest.warns(RuntimeWarning, match="OpenCL init/compile failed"):
        result = gpu_splat_softmax(ten_in, ten_flow, ten_metric)

    np.testing.assert_array_equal(result, expected)
    assert softsplat_cl._gpu_unavailable is True


def test_falls_back_to_cpu_when_no_gpu_device_found(monkeypatch, _reset_gpu_module_state) -> None:
    pytest.importorskip("pyopencl")

    def _no_devices(cl):
        raise RuntimeError("no OpenCL GPU device found")

    monkeypatch.setattr(softsplat_cl, "_select_device", _no_devices)

    ten_in, ten_flow, ten_metric = _tiny_inputs()
    expected = cpu_splat_softmax(ten_in, ten_flow, ten_metric)

    with pytest.warns(RuntimeWarning, match="OpenCL init/compile failed"):
        result = gpu_splat_softmax(ten_in, ten_flow, ten_metric)

    np.testing.assert_array_equal(result, expected)
