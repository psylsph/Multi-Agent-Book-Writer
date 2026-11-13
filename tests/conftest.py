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


@pytest.fixture(autouse=True)
def _no_endpoint_env(monkeypatch):
    """BOOKWRITER_LLM_BASE_URL / BOOKWRITER_SEARXNG_URL override config.yaml in
    load_config; one set in the developer's shell must not leak into tests."""
    from shared.llm_client import ENDPOINT_ENV
    for _, _, env in ENDPOINT_ENV:
        monkeypatch.delenv(env, raising=False)


@pytest.fixture(autouse=True)
def _no_real_env_file(monkeypatch, tmp_path_factory):
    """main() reads the repo's .env (API keys, endpoints) into os.environ; a
    test must never see the developer's keys or leave them behind."""
    import main
    monkeypatch.setattr(main, "ENV_FILE",
                        tmp_path_factory.mktemp("noenv") / ".env")


@pytest.fixture(autouse=True)
def _no_search_pacing(monkeypatch):
    """web_search.min_interval spaces real queries 1.5 s apart; a stubbed
    search has no rate limit to respect, and the suite must stay fast. The
    pacing itself is tested with a fake clock in test_web_search.py."""
    from shared import web_search
    monkeypatch.setattr(web_search, "_last_query", [None])
    monkeypatch.setattr(web_search, "PACING", False)
