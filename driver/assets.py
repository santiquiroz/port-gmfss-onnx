"""Model directory contract for the assembled GMFSS driver (Task 2.2).

Mirrors santiquiroz/port-audiosr-onnx's AudioSrAssets pattern (consumed form:
image-upscaler-amd/app/services/engines/audiosr/assets.py) -- `load()` reads
`manifest.json` plus whatever else the manifest points at, `is_complete()`
checks the manifest's own `required_files` list, `graph_path()` resolves a
graph name to its `.onnx` file. GMFSS has no auxiliary numpy weight files
(no mel basis / alphas_cumprod equivalent) -- the manifest itself is the only
non-graph asset, so `load()` is a thin JSON read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GRAPH_NAMES = ("featurenet", "gmflow", "metricnet", "fusionnet")


@dataclass(frozen=True)
class GmfssAssets:
    """Model directory contract produced by santiquiroz/port-gmfss-onnx."""

    model_dir: Path
    manifest: dict

    @staticmethod
    def load(model_dir: Path) -> "GmfssAssets":
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        return GmfssAssets(model_dir=model_dir, manifest=manifest)

    @staticmethod
    def is_complete(model_dir: Path) -> bool:
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        required = _manifest_required_files(manifest_path)
        if required is None:
            return False
        return all((model_dir / name).exists() for name in required)

    def graph_path(self, name: str) -> Path:
        return self.model_dir / f"{name}.onnx"

    @property
    def padded_hw(self) -> tuple[int, int]:
        height, width = self.manifest["resolution"]["fixed_padded_hw"]
        return int(height), int(width)


def _manifest_required_files(manifest_path: Path) -> list[str] | None:
    # None (unreadable/corrupt manifest) means NOT complete, distinct from an
    # empty-but-valid list -- mirrors AudioSrAssets's fallback contract.
    try:
        listed = json.loads(manifest_path.read_text(encoding="utf-8")).get("required_files")
    except (OSError, json.JSONDecodeError):
        return None
    if listed:
        return [str(name) for name in listed]
    return ["manifest.json"] + [f"{name}.onnx" for name in GRAPH_NAMES]
