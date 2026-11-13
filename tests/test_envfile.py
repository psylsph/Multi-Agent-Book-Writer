"""The CLI reads .env like docker compose does (shared/envfile.py)."""

import os

from shared import envfile


def test_parse_handles_comments_quotes_and_export():
    text = """# a comment
TINYFISH_API_KEY=tf-123
export PARALLEL_API_KEY="p a r"
SINGLE='x # not a comment'
INLINE=value # a comment
EMPTY=
not a line
"""
    assert envfile.parse(text) == {
        "TINYFISH_API_KEY": "tf-123", "PARALLEL_API_KEY": "p a r",
        "SINGLE": "x # not a comment", "INLINE": "value", "EMPTY": ""}


def test_load_sets_missing_variables_and_never_prints_values(tmp_path, monkeypatch,
                                                              capsys):
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    monkeypatch.setenv("PARALLEL_API_KEY", "from-shell")
    path = tmp_path / ".env"
    path.write_text("TINYFISH_API_KEY=secret-tf\nPARALLEL_API_KEY=from-file\n"
                    "LLM_API_KEY=\n")
    assert envfile.load(path) == ["TINYFISH_API_KEY"]
    assert os.environ["TINYFISH_API_KEY"] == "secret-tf"
    assert os.environ["PARALLEL_API_KEY"] == "from-shell"      # the shell wins
    out = capsys.readouterr().out
    assert "TINYFISH_API_KEY" in out and "secret-tf" not in out


def test_load_skips_a_docker_only_endpoint(tmp_path, monkeypatch, capsys):
    # http://host.docker.internal:8080 in .env is for the container; on the
    # host it would point the CLI at a server it cannot reach
    for env in ("BOOKWRITER_LLM_BASE_URL", "BOOKWRITER_SEARXNG_URL"):
        monkeypatch.delenv(env, raising=False)
    path = tmp_path / ".env"
    path.write_text("BOOKWRITER_LLM_BASE_URL=http://host.docker.internal:8080\n"
                    "BOOKWRITER_SEARXNG_URL=http://172.18.0.2:8080\n")
    loaded = envfile.load(path, resolves=lambda url: "docker.internal" not in url)
    assert loaded == ["BOOKWRITER_SEARXNG_URL"]
    assert "BOOKWRITER_LLM_BASE_URL" not in os.environ
    assert "skipping BOOKWRITER_LLM_BASE_URL" in capsys.readouterr().out


def test_load_without_a_file_does_nothing(tmp_path):
    assert envfile.load(tmp_path / ".env") == []


def test_host_resolves_rejects_a_url_without_a_host():
    assert envfile.host_resolves("http://:8080") is False
    assert envfile.host_resolves("http://127.0.0.1:8080") is True


def test_main_reads_the_env_file_before_the_config(tmp_path, monkeypatch):
    import main
    calls = []
    monkeypatch.setattr(main.envfile, "load", lambda p: calls.append(("env", p)))
    monkeypatch.setattr(main, "load_config",
                        lambda p: calls.append(("config", p)) or (_ for _ in ()).throw(
                            FileNotFoundError("stop here")))
    monkeypatch.setattr("sys.argv", ["main.py", "--demo"])
    try:
        main.main()
    except SystemExit:
        pass
    assert [c[0] for c in calls] == ["env", "config"]
    assert calls[0][1] == main.ENV_FILE
