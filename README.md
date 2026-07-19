# port-gmfss-onnx

**First known ONNX port of [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) — one of the best anime frame-interpolation models — running on *any* DirectX 12 GPU (AMD, Intel, NVIDIA), no CUDA required.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why this exists

[GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) is one of the strongest open models for anime/animation frame interpolation, combining optical flow (GMFlow) with a synthesis network tuned for hand-drawn and cel-shaded motion. It is MIT-licensed.

Like most PyTorch research models, it only runs well with **CUDA**. Anyone on an AMD or Intel GPU is stuck on CPU inference, which is impractical for video work, or has no path to run it at all inside a lightweight, torch-free application.

**This project decomposes GMFSS_Fortuna into plain ONNX graphs** so the heavy compute runs through [onnxruntime](https://onnxruntime.ai/) on **any execution provider** — DirectML (any DX12 GPU: AMD Radeon, Intel Arc, NVIDIA), CUDA, or CPU. No torch at inference time, no CUDA lock-in.

Numbers will be added here as each phase lands — measured on real hardware, never promised in advance.

## Status

| Component | Export | Parity | DirectML |
|---|---|---|---|
| Optical flow (GMFlow) | | | |
| Synthesis / fusion network | | | |
| numpy/onnxruntime driver | | | |

## Credits

- **Model & weights:** [98mxr/GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) (MIT) — all the science is theirs. This repo is *only* the porting toolkit.
- Sibling ports: [port-audiosr-onnx](https://github.com/santiquiroz/port-audiosr-onnx) (audio super-resolution, same motivation and approach).

## License

MIT. The exported graphs inherit GMFSS_Fortuna's MIT license.
