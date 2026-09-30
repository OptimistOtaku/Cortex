import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def perception():
    """SigLIP2 loads once per test session (~1.5 GB RAM; this laptop has 7.3 GB)."""
    from sietch.node.perception import Perception

    return Perception("siglip2")
