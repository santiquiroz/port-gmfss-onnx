"""Corre la referencia dorada de GMFSS_Fortuna (variante PG/base) en CPU puro, sin cupy.

Para cada triplete en refs/inputs/ (img0, frame intermedio real, img1) corre
FeatureNet -> GMFlow -> MetricNet -> softsplat_torch x8 -> FusionNet a t=0.5, dumpea cada
tensor intermedio a refs/golden/<par>/*.npy, guarda el frame interpolado como PNG, y compara
por SSIM contra el frame real intermedio del clip.

Uso: .venv/Scripts/python.exe toolkit/run_reference.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmfss_pg_pipeline import (  # noqa: E402
    GMFSSBasePipeline,
    denormalize_frame,
    normalize_frame,
    resize_bilinear,
    target_padded_size,
)

ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = ROOT / "refs" / "inputs"
GOLDEN_DIR = ROOT / "refs" / "golden"
MODELS_DIR = ROOT / "toolkit" / "vendor" / "vs_gmfss_fortuna" / "models"
TIMESTEP = 0.5
SSIM_THRESHOLD = 0.9

HOLYWU_COMMIT = "f4f990a456678942beb7bcbca3fd5645d139ebe4"
MXR98_COMMIT = "0fb7ac1dc292e2615217110dd9d82557845fb919"


class TensorDumper:
    """Guarda cada tensor intermedio de un par a refs/golden/<pair_name>/*.npy."""

    def __init__(self, pair_dir: Path) -> None:
        self.pair_dir = pair_dir
        self.pair_dir.mkdir(parents=True, exist_ok=True)
        self.shapes: dict[str, list[int]] = {}

    def __call__(self, name: str, tensor: torch.Tensor) -> None:
        array = tensor.detach().cpu().numpy()
        np.save(self.pair_dir / f"{name}.npy", array)
        self.shapes[name] = list(array.shape)


def discover_triplets(inputs_dir: Path) -> list[dict]:
    """refs/inputs/<name>_img0.png + <name>_gt_mid.png + <name>_img1.png -> lista de triples."""
    triplets = []
    for img0_path in sorted(inputs_dir.glob("*_img0.png")):
        prefix = img0_path.name[: -len("_img0.png")]
        gt_mid_path = inputs_dir / f"{prefix}_gt_mid.png"
        img1_path = inputs_dir / f"{prefix}_img1.png"
        if not gt_mid_path.exists() or not img1_path.exists():
            continue
        triplets.append({
            "name": prefix,
            "img0": img0_path,
            "gt_mid": gt_mid_path,
            "img1": img1_path,
        })
    return triplets


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_ssim(frame_a_rgb: np.ndarray, frame_b_rgb: np.ndarray) -> float:
    return float(structural_similarity(frame_a_rgb, frame_b_rgb, channel_axis=2, data_range=255))


def prepare_padded_input(frame_rgb: np.ndarray, padded_h: int, padded_w: int) -> torch.Tensor:
    normalized = normalize_frame(frame_rgb)
    return resize_bilinear(normalized, padded_h, padded_w)


def run_single_pair(pipeline: GMFSSBasePipeline, triplet: dict) -> dict:
    img0_rgb = load_rgb(triplet["img0"])
    img1_rgb = load_rgb(triplet["img1"])
    gt_mid_rgb = load_rgb(triplet["gt_mid"])
    orig_h, orig_w = img0_rgb.shape[:2]
    padded_h, padded_w = target_padded_size(orig_h, orig_w)

    pair_dir = GOLDEN_DIR / triplet["name"]
    dumper = TensorDumper(pair_dir)

    img0 = prepare_padded_input(img0_rgb, padded_h, padded_w)
    img1 = prepare_padded_input(img1_rgb, padded_h, padded_w)
    dumper("input_norm_img0", img0)
    dumper("input_norm_img1", img1)

    timestep = torch.tensor([TIMESTEP], dtype=torch.float32)

    start = time.perf_counter()
    padded_out = pipeline(img0, img1, timestep, record=dumper)
    elapsed_seconds = time.perf_counter() - start

    final_out = resize_bilinear(padded_out, orig_h, orig_w)
    dumper("final_frame", final_out)

    interp_rgb = denormalize_frame(final_out)
    cv2.imwrite(str(pair_dir / "interp.png"), cv2.cvtColor(interp_rgb, cv2.COLOR_RGB2BGR))

    ssim = compute_ssim(interp_rgb, gt_mid_rgb)

    return {
        "name": triplet["name"],
        "img0": str(triplet["img0"].relative_to(ROOT)),
        "gt_mid": str(triplet["gt_mid"].relative_to(ROOT)),
        "img1": str(triplet["img1"].relative_to(ROOT)),
        "original_hw": [orig_h, orig_w],
        "padded_hw": [padded_h, padded_w],
        "cpu_seconds": elapsed_seconds,
        "ssim_vs_real_mid_frame": ssim,
        "ssim_pass": ssim > SSIM_THRESHOLD,
        "tensor_shapes": dumper.shapes,
    }


def write_meta(results: list[dict]) -> None:
    meta = {
        "task": "0.2 - referencia dorada GMFSS_Fortuna PG en CPU",
        "model_variant": "base (PG/pg104, ver docs/vendored-sources.md)",
        "timestep": TIMESTEP,
        "device": "cpu",
        "softsplat_impl": "toolkit/vendor/gmfss_fortuna_98mxr/softsplat_torch.py (pure PyTorch, no cupy)",
        "vendored_commits": {
            "HolyWu/vs-gmfss_fortuna": HOLYWU_COMMIT,
            "98mxr/GMFSS_Fortuna": MXR98_COMMIT,
        },
        "normalization": "uint8 RGB HWC /255.0 -> float32 CHW [0,1], then bilinear resize"
        " (stretch, not zero-pad) to nearest multiple of 64 in each dim,"
        " matching upstream GMFSS.py / inference_video.py exactly",
        "pad_multiple": 64,
        "ssim_threshold": SSIM_THRESHOLD,
        "pairs": results,
    }
    (GOLDEN_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    triplets = discover_triplets(INPUTS_DIR)
    if not triplets:
        raise SystemExit(f"no triplets found in {INPUTS_DIR}")

    pipeline = GMFSSBasePipeline()
    pipeline.load_weights(MODELS_DIR)

    results = []
    for triplet in triplets:
        print(f"[run_reference] pair={triplet['name']}")
        result = run_single_pair(pipeline, triplet)
        print(
            f"  ssim={result['ssim_vs_real_mid_frame']:.4f}"
            f" ({'PASS' if result['ssim_pass'] else 'FAIL'})"
            f" cpu={result['cpu_seconds']:.1f}s"
        )
        results.append(result)

    write_meta(results)

    primary = results[0]["name"]
    shutil.copy(GOLDEN_DIR / primary / "interp.png", GOLDEN_DIR / "interp_ref.png")

    all_pass = all(r["ssim_pass"] for r in results)
    print(f"[run_reference] done, {len(results)} pares, all_ssim_pass={all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
