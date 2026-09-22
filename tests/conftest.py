import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from po_delay.data_generation import GeneratorParams, generate  # noqa: E402


@pytest.fixture(scope="session")
def small_dataset():
    """A small, fast-to-generate synthetic dataset shared across tests."""
    return generate(GeneratorParams(n_orders=1500, n_suppliers=40, seed=7))
