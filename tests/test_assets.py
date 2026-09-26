"""Tests for driver.assets.GmfssAssets: manifest loading, completeness check,
graph path resolution. Mirrors the coverage AudioSrAssets gets in its sibling
project -- see driver/assets.py's module docstring for the precedent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from driver.assets import GmfssAssets

ART = Path(__file__).resolve().parent.parent / "artifacts"


def test_load_reads_real_manifest() -> None:
    assets = GmfssAssets.load(ART)

    assert assets.model_dir == ART
    assert assets.manifest["model_variant"].startswith("base")
    assert assets.padded_hw == (1088, 1920)


def test_graph_path_resolves_onnx_file_under_model_dir() -> None:
    assets = GmfssAssets.load(ART)

    assert assets.graph_path("featurenet") == ART / "featurenet.onnx"
    assert assets.graph_path("metricnet") == ART / "metricnet.onnx"


@pytest.mark.requires_artifacts
def test_is_complete_true_for_real_artifacts_dir() -> None:
    assert GmfssAssets.is_complete(ART) is True


def test_is_complete_false_when_manifest_missing(tmp_path: Path) -> None:
    assert GmfssAssets.is_complete(tmp_path) is False


def test_is_complete_false_when_a_required_graph_is_missing(tmp_path: Path) -> None:
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in manifest["required_files"]:
        if name == "manifest.json":
            continue
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "gmflow.onnx").unlink()

    assert GmfssAssets.is_complete(tmp_path) is False


def test_is_complete_true_when_all_required_files_present(tmp_path: Path) -> None:
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in manifest["required_files"]:
        if name == "manifest.json":
            continue
        (tmp_path / name).write_bytes(b"")

    assert GmfssAssets.is_complete(tmp_path) is True


def test_is_complete_false_for_corrupt_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not valid json", encoding="utf-8")

    assert GmfssAssets.is_complete(tmp_path) is False


def test_manifest_file_entries_are_flat_release_asset_names() -> None:
    # The published models-v1.0 release is a flat list of assets; a subdirectory in
    # any manifest file entry names a file no downloader of the release can find.
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    fp16_files = [v["file"] for v in manifest["fp16_variants"].values() if isinstance(v, dict)]
    listed = manifest["required_files"] + manifest["optional_files"] + fp16_files

    assert [name for name in listed if "/" in name or "\\" in name] == []


def test_manifest_optional_files_cover_every_fp16_variant() -> None:
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
    fp16_files = {v["file"] for v in manifest["fp16_variants"].values() if isinstance(v, dict)}

    assert fp16_files <= set(manifest["optional_files"])
