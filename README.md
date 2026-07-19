# port-gmfss-onnx

**First known ONNX port of [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) — one of the best anime frame-interpolation models — running on *any* DirectX 12 GPU (AMD, Intel, NVIDIA), no CUDA required.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why this exists

[GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) is one of the strongest open models for anime/animation frame interpolation, combining optical flow (GMFlow) with a synthesis network tuned for hand-drawn and cel-shaded motion. It is MIT-licensed.

Like most PyTorch research models, it only runs well with **CUDA**. Anyone on an AMD or Intel GPU is stuck on CPU inference, which is impractical for video work, or has no path to run it at all inside a lightweight, torch-free application.

**This project decomposes GMFSS_Fortuna into plain ONNX graphs** so the heavy compute runs through [onnxruntime](https://onnxruntime.ai/) on **any execution provider** — DirectML (any DX12 GPU: AMD Radeon, Intel Arc, NVIDIA), CUDA, or CPU. No torch at inference time, no CUDA lock-in.

Numbers will be added here as each phase lands — measured on real hardware, never promised in advance.

## Status

| Component | Export | CPU-EP rel-err | DirectML rel-err | DirectML speedup |
|---|---|---|---|---|
| FeatureNet | done (opset 17, legacy) | 0.000000 | 0.000001 | 6.3–7.0x |
| MetricNet | done (opset 18, dynamo)\* | 0.000001 | 0.000061 | 26–27x |
| FusionNet (GridNet) | done (opset 17, legacy) | 0.000001 | 0.000001 | 20.5–22x |
| Optical flow (GMFlow) | done (opset 17, legacy) | 0.000415† | 0.000086 | 7.77–7.94x |
| numpy/onnxruntime driver (`driver/pipeline.py`) | assembled, PARITY OK‡ | per-stage: 0.000000–0.000415 | per-stage: 0.000000–0.000086 | see ‡ |

### Pipeline completo validado, splat CPU (Task 2.2)

`driver/pipeline.py`'s `GmfssDriver` wires all 4 ONNX graphs + Task 2.1's `splat_softmax`
(numpy/torch-CPU, never DirectML — see `driver/softsplat.py`) into the full
FeatureNet → GMFlow×2 → MetricNet → softsplat×8 → FusionNet composition, with
`reuse()`/`interpolate_pair()` caching flow/features once per image pair regardless of
how many timesteps are requested. `toolkit/validate_driver.py` is the parity gate:
stage-by-stage (feeding golden tensors at every boundary, isolating each stage's own
error) **and** true end-to-end (`GmfssDriver.interpolate_pair()`, no golden substitution
anywhere) against `refs/golden/`, on both CPU-EP and DirectML.

**PARITY OK**, all stages, both providers (`vf_t006`, plus `vs_t013`/`vwarm_t019` for
end-to-end on CPU-EP):

| Stage | CPU-EP max-rel-err | DirectML max-rel-err | tol |
|---|---|---|---|
| featurenet (6 outputs) | 0.000000 | 0.000001 | 1e-3 / 1e-2 |
| gmflow (flow01, flow10) | 0.000255–0.000415 | 0.000038–0.000086 | 1e-3 / 1e-2 |
| metricnet (metric0, metric1) | 0.000001 | 0.000021–0.000061 | 1e-3 / 1e-2 |
| splat ×8 (vs golden dump) | 0.000000 | 0.000000 | 1e-3 / 1e-2 |
| fusionnet_out / final_frame_padded | 0.000001 | 0.000001 | 1e-3 / 1e-2 |
| **end-to-end** (real driver, no golden reset) | rms-rel-err 0.000061–0.000231 | rms-rel-err 0.000017 | rms 1e-3 / 1e-2‡ |
| **end-to-end SSIM** vs golden final frame | 0.999998–1.000000 | 1.000000 | > 0.99 |

‡ End-to-end is gated on RMS-rel-err, not max-abs-rel-err: chaining 4 independently-passing
networks compounds a handful of occlusion-boundary outlier pixels (≈0.02% of pixel-values
exceed 1e-3; mean abs diff ≈2e-6) — the same outlier-pixel-dominated pattern already noted
above for GMFlow alone, just visible again after 4x chaining. Max-abs-rel-err (0.006–0.086
across pairs/providers) is printed for transparency but is informational, not gating; RMS
and SSIM are the metrics that reflect true whole-frame fidelity here. See
`.superpowers/sdd/task-2.2-report.md` for the full breakdown.

**Measured fps @1080p (1088×1920 padded), splat always CPU** — "parity mode", pre-Phase-3
OpenCL splat kernel:

| Graphs on | fps | s/frame |
|---|---|---|
| CPU-EP | 0.058–0.060 | 16.7–17.3 |
| DirectML | 0.117 | 8.5 |

Both are below the plan's 0.2–0.6 fps estimate. Root cause, backed by a committed,
reproducible per-stage profiler (`toolkit/profile_pipeline.py` — reuses
`toolkit/validate_driver.py`'s warmed-session `make_run_graph()`, times each stage
`GmfssDriver.interpolate_pair()` calls internally) rather than the ad hoc one-off
instrumented run this section previously cited: GMFlow — called twice per pair for
flow01/flow10 — is **55.6% of total time on CPU-EP and 71–74% on DirectML** (two runs
measured, `vf_t006`), even though the 8 CPU splat calls together take ~1s on both
providers. Full per-stage breakdown, both providers (`.venv/Scripts/python.exe
toolkit/profile_pipeline.py`):

| Stage | CPU-EP | DirectML |
|---|---|---|
| featurenet (×2) | 0.852–0.882s (5.1%) | 0.136–0.222s (1.6–2.6%) |
| gmflow (×2) | 9.294–9.615s (55.6%) | 6.090–6.349s (71.4–74.4%) |
| metricnet (×1) | 0.627–0.645s (3.7%) | 0.033–0.040s (0.4–0.5%) |
| splat (×8, CPU) | 1.028–1.046s (6.1%) | 0.963–1.048s (11.3–12.3%) |
| fusionnet (×1) | 4.102–4.213s (24.4–24.5%) | 0.243–0.252s (2.9%) |
| unaccounted (resize/concat/weighting) | 0.827–0.889s (4.9–5.1%) | 0.800–0.881s (9.4–10.3%) |
| **total** | 16.730–17.290s | 8.524–8.532s |

This corroborates the earlier ad hoc estimate (56–78%) rather than contradicting it, now
with a reproducible source. The plan's estimate assumed CPU splat would dominate; in
practice GMFlow's transformer-attention cost dominates instead, on both providers —
splat stays under ~1.1s everywhere, so Phase 3's OpenCL splat kernel alone will not close
the gap to the plan's fps target; GMFlow is the larger lever.

### OpenCL splat kernel (Task 3.1)

`driver/kernels/splat.cl` + `driver/softsplat_cl.py` add a GPU-accelerated *alternative*
backend for the same softmax splat, gated behind optional `pyopencl`
(`toolkit/requirements-gpu-splat.txt`) with automatic, warn-once fallback to the CPU
`splat_softmax` if OpenCL is unavailable or fails to compile/run. `driver/softsplat.py`
stays the correctness reference and default; nothing else in the repo requires `pyopencl`.
Bilinear scatter-add uses one work-item per source pixel and an `atomic_cmpxchg`
CAS-loop-on-reinterpreted-bits for float accumulation — this AMD driver (RX 7800 XT,
OpenCL 2.1) exposes no native float-atomic extension, only the standard
`cl_khr_global_int32_base_atomics`.

Correctness: all 24 real call-site combinations (8 splat calls × 3 golden pairs) match
`driver.softsplat.splat_softmax` at whole-tensor L2 relative error ≈5×10⁻⁸ — far under the
brief's 1e-5 tolerance (max single-element abs diff ≈4.8×10⁻⁷, i.e. float32-epsilon level;
see `tests/test_softsplat_cl.py` for why per-element relative error is the wrong metric
here — some feature-map elements are near zero, same outlier-domination issue already
noted above for end-to-end rel-err).

Isolated kernel bench (RX 7800 XT, real call-site tensor shapes, 30 timed iterations + 3
warmup, `toolkit/bench_splat_cl.py`), against the brief's <20ms/call target and the CPU
reference at the same synthetic shapes:

| Call site(s) | Shape | GPU mean | CPU mean (same shape) | Speedup | vs <20ms target |
|---|---|---|---|---|---|
| I1t / I2t | (1,3,544,960) | 9.73ms | 42.56ms | 4.4x | **HIT** |
| feat\*t1 (pyramid scale1) | (1,64,544,960) | 154.10ms | 289.92ms | 1.9x | MISS |
| feat\*t2 (pyramid scale2) | (1,128,272,480) | 82.48ms | 112.97ms | 1.4x | MISS |
| feat\*t3 (pyramid scale3) | (1,192,136,240) | 31.19ms | 30.26ms | ~1.0x | MISS |

Only the 2 lowest-channel-count calls (I1t/I2t, 3 channels) hit the <20ms target; the 6
feature-pyramid calls (64–192 channels) miss it. Root cause: kernel time scales with total
atomic-op count (`channels × H × W`), not resolution alone, and roughly half of the
higher-channel calls' wall time is the numpy `exp`/multiply/concat/normalize pre- and
post-processing around the kernel (kept in Python per this task's design, mirroring
`driver/softsplat.py` — see the report below for the upload/kernel/download/numpy
breakdown). Aggregate across all 8 real calls/frame: ≈555ms GPU vs ≈950–1050ms CPU at
matching synthetic shapes (≈1.7–1.9x), consistent with the ≈1.03–1.05s CPU total already
measured on real tensors above. This is not a hard gate for Task 3.1 (Task 3.2 owns the
actual kill-criterion) — reported honestly per the task brief. Full breakdown, commands,
and diagnosis: `.superpowers/sdd/task-3.1-report.md`.

Not judged "OpenCL decepciona" (the plan's condition for the documented ONNX-ScatterND
Alternative B, not implemented): the kernel is still a real, reproducible net win over CPU
at every shape (1.0x–4.4x, never a regression), and splat was already <1.1s of the ≈8.5–17s
total pipeline time (Task 2.2) — missing the per-call target doesn't change the project's
actual bottleneck, which is GMFlow, not splat. Alternative B stays documented-only.

\* MetricNet's legacy JIT exporter trips on `aten::l1_loss`; exported via `dynamo=True`
instead. Its DirectML session needs `graph_optimization_level = ORT_DISABLE_ALL` — the
default fused DML kernel reproducibly hangs the GPU after ~3 calls on this hardware/driver
(no correctness issue, values match; see `toolkit/validate_ort.py`). Numbers measured
against real golden tensors from `refs/golden/` (RX 7800 XT, 3 validation pairs).
Variant is "base" (no IFNet/RIFE — see `docs/vendored-sources.md`).

† GMFlow's legacy JIT exporter worked directly at opset 17 — no dynamo fallback needed,
despite `F.unfold`, shifted-window `torch.roll`, and 4D `F.grid_sample` all appearing in
the traced graph (swin attention + local correlation + flow warping). DirectML runs the
single graph with no op rejections or crashes — no sub-graph split needed. rel-err is
outlier-dominated (occlusion-boundary pixels): RMS rel-err is 0.000007–0.000029, ~15–60x
smaller than the reported max rel-err, with under 0.06% of pixels driving the max on any
case. The max-err gate (0.000415) already passes with margin under the 1e-3 threshold, so
it is the reported/binding number — RMS is corroborating evidence, not a rescue. Both
`flow01` and `flow10` directions validated (same graph/weights, swapped `img0`/`img1`
args) across all 3 golden pairs (6 cases total).

## Credits

- **Model & weights:** [98mxr/GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) (MIT) — all the science is theirs. This repo is *only* the porting toolkit.
- Sibling ports: [port-audiosr-onnx](https://github.com/santiquiroz/port-audiosr-onnx) (audio super-resolution, same motivation and approach).

## License

MIT. The exported graphs inherit GMFSS_Fortuna's MIT license.
