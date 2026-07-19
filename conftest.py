import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu_correctness: needs a real working OpenCL GPU (see tests/test_softsplat_cl.py)"
    )
