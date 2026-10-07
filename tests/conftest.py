import sys
from pathlib import Path

# make repo root importable regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import requests


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Fail fast if a test forgets to stub HTTP, instead of hanging while the
    client waits for a server that isn't there. Tests that need HTTP replace
    these with their own fakes via monkeypatch."""
    def blocked(*args, **kwargs):
        raise AssertionError("test attempted a real HTTP request: stub it")

    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(requests, "post", blocked)
