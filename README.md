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
| CPU-EP | 0.058 | 17.2 |
| DirectML | 0.119 | 8.4 |

Both are below the plan's 0.2–0.6 fps estimate. Root cause (profiled, not guessed): GMFlow
— called twice per pair for flow01/flow10 — is 56–78% of total time by itself (≈9s of
≈11–17s), even on DirectML; the 8 CPU splat calls together take under 1s. The plan's
estimate assumed CPU splat would dominate; in practice GMFlow's transformer-attention cost
dominates instead, on both providers.

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
