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
| Optical flow (GMFlow) | | | | |
| numpy/onnxruntime driver | | | | |

\* MetricNet's legacy JIT exporter trips on `aten::l1_loss`; exported via `dynamo=True`
instead. Its DirectML session needs `graph_optimization_level = ORT_DISABLE_ALL` — the
default fused DML kernel reproducibly hangs the GPU after ~3 calls on this hardware/driver
(no correctness issue, values match; see `toolkit/validate_ort.py`). Numbers measured
against real golden tensors from `refs/golden/` (RX 7800 XT, 3 validation pairs).
Variant is "base" (no IFNet/RIFE — see `docs/vendored-sources.md`).

## Credits

- **Model & weights:** [98mxr/GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) (MIT) — all the science is theirs. This repo is *only* the porting toolkit.
- Sibling ports: [port-audiosr-onnx](https://github.com/santiquiroz/port-audiosr-onnx) (audio super-resolution, same motivation and approach).

## License

MIT. The exported graphs inherit GMFSS_Fortuna's MIT license.
