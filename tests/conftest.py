from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._golden_sites import golden_is_available

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

_MISSING_GOLDEN = "refs/golden/ is missing or incomplete (gitignored; copy it from a checkout that has it)"
_MISSING_ARTIFACTS = "real ONNX graphs are missing from artifacts/ (gitignored; copy them or run the export)"


def _real_artifacts_are_available() -> bool:
    manifest = json.loads((ARTIFACTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    return all((ARTIFACTS_DIR / name).is_file() for name in manifest["required_files"])


def pytest_collection_modifyitems(config, items) -> None:
    _skip_marked_items(items, "requires_golden", golden_is_available(), _MISSING_GOLDEN)
    _skip_marked_items(items, "requires_artifacts", _real_artifacts_are_available(), _MISSING_ARTIFACTS)


def _skip_marked_items(items, marker: str, data_available: bool, reason: str) -> None:
    if data_available:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker(marker) is not None:
            item.add_marker(skip)
