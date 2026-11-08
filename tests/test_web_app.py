"""Tests for the web GUI (web/app.py).

The app never imports the pipeline: it spawns `main.py` per book and relays
the browser's answers to shared/webanswer.py's files. These tests stub the
spawn, so no pipeline, LLM or network is involved.
"""

import importlib
import json
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402  (after importorskip)

from shared import webanswer  # noqa: E402

SEED = "# The Marsh Light\n\nA surveyor finds a letter.\n"


@pytest.fixture
def web(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n  model: test-model\n  base_url: \"http://llm.test:8080\"\n"
        "book:\n  num_chapters: 3\n  words_per_chapter: 900\n"
        "  seed_review: ask\n",
        encoding="utf-8")
    monkeypatch.setenv("BOOKWRITER_CONFIG", str(cfg))
    monkeypatch.setenv("BOOKWRITER_WEB_OUTPUT", str(tmp_path / "web"))
    from web import app as web_app
    module = importlib.reload(web_app)          # re-reads the env paths
    module._jobs.clear()

    spawned = []

    def fake_spawn(job, options=None):
        spawned.append((job, options or job.options or {}))
        job.started = 1.0

    monkeypatch.setattr(module, "_spawn", fake_spawn)
    client = TestClient(module.app)
    return SimpleNamespace(module=module, client=client, spawned=spawned,
                           config=cfg, root=tmp_path)


def start(web, **overrides):
    body = {"seed": SEED}
    body.update(overrides)
    response = web.client.post("/api/runs", json=body)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_index_serves_the_page(web):
    response = web.client.get("/")
    assert response.status_code == 200
    assert "Multi-Agent Book Writer" in response.text


def test_config_endpoint_reports_the_defaults(web):
    data = web.client.get("/api/config").json()
    assert data["model"] == "test-model"
    assert data["num_chapters"] == 3
    assert data["seed_review"] == "ask"
    assert data["web_output"] == str(web.root / "web")


def test_starting_a_run_writes_its_own_seed_and_config(web):
    run_id = start(web, chapters=4)

    job, options = web.spawned[0]
    assert job.id == run_id
    assert options["chapters"] == 4
    assert (job.control / "seed.md").read_text(encoding="utf-8") == SEED

    import yaml
    config = yaml.safe_load((job.directory / "config.yaml").read_text())
    # output.directory points back at the run, so state/ and chapters/ are its own
    assert config["output"]["directory"] == str(job.directory)
    assert config["llm"]["model"] == "test-model"


def test_the_command_line_is_the_cli(web):
    run_id = start(web, chapters=4, plan_only=True, seed_review="warn",
                   no_resume=True)
    job = web.module._job(run_id)
    cmd = web.module._command(job, {"chapters": 4, "plan_only": True,
                                    "seed_review": "warn", "no_resume": True})
    assert cmd[1:4] == ["-u", "-m", "web.child"]
    assert "--config" in cmd and "--seed" in cmd
    assert cmd[cmd.index("-c") + 1] == "4"
    assert "--plan-only" in cmd
    assert cmd[cmd.index("--seed-review") + 1] == "warn"
    assert "--no-resume" in cmd


def test_a_seed_is_required(web):
    response = web.client.post("/api/runs", json={"seed": "   "})
    assert response.status_code == 400
    assert web.spawned == []


def test_only_one_book_is_written_at_a_time(web, monkeypatch):
    monkeypatch.setattr(web.module, "_live_jobs", lambda: [object()])
    response = web.client.post("/api/runs", json={"seed": SEED})
    assert response.status_code == 409
    assert web.spawned == []


def test_a_question_is_relayed_and_answered(web, monkeypatch):
    run_id = start(web)
    job = web.module._job(run_id)
    job.pid = 4242
    monkeypatch.setattr(web.module, "_running", lambda pid, run_id: True)
    asked = webanswer.write_question(job.control / "ask", "  > ",
                                     default="d", kind="seed_review")

    pending = web.client.get(f"/api/runs/{run_id}/question").json()
    assert pending["pending"]["id"] == asked["id"]
    assert pending["pending"]["default"] == "d"
    assert "context" in pending

    response = web.client.post(f"/api/runs/{run_id}/answer",
                               json={"id": asked["id"], "answer": "banana"})
    assert response.status_code == 200
    assert webanswer.read_answer(job.control / "ask", asked["id"]) == "banana"
    assert web.client.get(f"/api/runs/{run_id}/question").json()["pending"] is None


def test_answering_a_stale_question_is_refused(web, monkeypatch):
    run_id = start(web)
    job = web.module._job(run_id)
    job.pid = 4242
    monkeypatch.setattr(web.module, "_running", lambda pid, run_id: True)
    asked = webanswer.write_question(job.control / "ask", ">")
    response = web.client.post(f"/api/runs/{run_id}/answer",
                               json={"id": "not-the-question", "answer": "x"})
    assert response.status_code == 409
    assert webanswer.read_answer(job.control / "ask", asked["id"]) is None


def test_answering_with_nothing_pending_is_refused(web, monkeypatch):
    run_id = start(web)
    web.module._job(run_id).pid = 4242
    monkeypatch.setattr(web.module, "_running", lambda pid, run_id: True)
    response = web.client.post(f"/api/runs/{run_id}/answer",
                               json={"id": "abc", "answer": "x"})
    assert response.status_code == 409


def test_status_reports_chapter_progress_from_state(web):
    run_id = start(web)
    job = web.module._job(run_id)
    state = job.directory / "state"
    state.mkdir()
    (state / "outline.json").write_text(json.dumps([
        {"number": 1, "title": "The Letter", "summary": "a"},
        {"number": 2, "title": "The Ferry", "summary": "b"},
    ]))
    (state / "research.json").write_text(json.dumps({"1": "brief"}))
    (state / "drafts.json").write_text(json.dumps({"1": "text"}))
    (state / "final.json").write_text(json.dumps({"1": "edited"}))
    (state / "bible.json").write_text(json.dumps({"title": "The Marsh Light"}))

    status = web.client.get(f"/api/runs/{run_id}").json()
    assert status["title"] == "The Marsh Light"
    assert status["counts"] == {"planned": 2, "researched": 1, "drafted": 1,
                                "edited": 1}
    assert [c["edited"] for c in status["chapters"]] == [True, False]
    assert status["running"] is False


def test_files_are_listed_and_served(web):
    run_id = start(web)
    job = web.module._job(run_id)
    (job.directory / "chapters").mkdir()
    (job.directory / "chapters" / "chapter_01.md").write_text("# One\n\nText.\n")
    (job.directory / "draft.md").write_text("# The Marsh Light\n")

    status = web.client.get(f"/api/runs/{run_id}").json()
    paths = [f["path"] for f in status["files"]]
    assert "draft.md" in paths
    assert "chapters/chapter_01.md" in paths

    response = web.client.get(f"/api/runs/{run_id}/file",
                              params={"path": "chapters/chapter_01.md"})
    assert response.status_code == 200
    assert "Text." in response.text


def test_a_file_outside_the_run_is_refused(web):
    run_id = start(web)
    for path in ("../config.yaml", "../../etc/passwd", "/etc/passwd"):
        response = web.client.get(f"/api/runs/{run_id}/file",
                                  params={"path": path})
        assert response.status_code == 404, path


def test_unknown_runs_are_404(web):
    assert web.client.get("/api/runs/nope").status_code == 404
    assert web.client.get("/api/runs/../../etc").status_code == 404


def test_runs_are_listed(web):
    run_id = start(web)
    runs = web.client.get("/api/runs").json()["runs"]
    assert [run["id"] for run in runs] == [run_id]
    assert runs[0]["title"] == "The Marsh Light"     # from the seed's first line


def test_resuming_reuses_the_same_directory(web):
    run_id = start(web)
    first = web.spawned[0][0].directory
    response = web.client.post("/api/runs", json={"resume": run_id})
    assert response.status_code == 200
    assert response.json()["id"] == run_id
    assert web.spawned[1][0].directory == first


def test_resuming_an_unknown_run_is_404(web):
    response = web.client.post("/api/runs", json={"resume": "nope"})
    assert response.status_code == 404


def test_stopping_a_run_that_is_not_running_is_refused(web):
    run_id = start(web)
    response = web.client.post(f"/api/runs/{run_id}/stop")
    assert response.status_code == 409


def test_epub_needs_chapters(web):
    run_id = start(web)
    response = web.client.post(f"/api/runs/{run_id}/epub")
    assert response.status_code == 400
    assert "chapter" in response.json()["detail"].lower()


def test_the_child_gets_the_question_channel_and_live_output(web):
    run_id = start(web)
    job = web.module._job(run_id)
    env = web.module._environment(job)
    assert env[webanswer.ENV_DIR] == str(job.control / "ask")
    assert env["BOOKWRITER_EXIT_FILE"] == str(job.control / "exit.json")
    assert env["PYTHONUNBUFFERED"] == "1"


def test_a_boolean_seed_review_is_named_for_the_page(web):
    """YAML reads an unquoted `seed_review: off` as False; the page must show
    'off', not 'False'."""
    assert web.module._seed_review_name(False) == "off"
    assert web.module._seed_review_name(True) == "ask"
    assert web.module._seed_review_name("warn") == "warn"


def test_elapsed_stops_when_the_run_does(web):
    run_id = start(web)
    job = web.module._job(run_id)
    job.exit_code, job.finished = 0, 10.0
    status = web.client.get(f"/api/runs/{run_id}").json()
    assert status["elapsed"] == round(10.0 - job.started, 1)


def test_a_base_url_override_points_runs_at_the_host(web, monkeypatch):
    """In a container the LLM on the host is not on localhost; the compose
    file passes BOOKWRITER_LLM_BASE_URL so config.yaml stays usable as-is."""
    import yaml

    monkeypatch.setenv(web.module.LLM_URL_ENV, "http://host.docker.internal:8080")
    run_id = start(web)
    job = web.module._job(run_id)
    config = yaml.safe_load((job.directory / "config.yaml").read_text())
    assert config["llm"]["base_url"] == "http://host.docker.internal:8080"


def test_no_base_url_override_leaves_the_config_alone(web):
    import yaml

    run_id = start(web)
    job = web.module._job(run_id)
    config = yaml.safe_load((job.directory / "config.yaml").read_text())
    assert config["llm"]["base_url"] == "http://llm.test:8080"


# ------------------------------------------------------------- persistence

def dead_pid():
    """A pid that is definitely not running any more."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def write_job(directory, pid, options=None, started=1000.0):
    (directory / "job").mkdir(exist_ok=True)
    (directory / "job" / "job.json").write_text(json.dumps({
        "id": directory.name, "pid": pid, "started": started,
        "options": options or {}}))


def new_run_dir(web, run_id="20260101-000000-abcd"):
    directory = web.root / "web" / run_id
    (directory / "job").mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text("output:\n  directory: .\n")
    (directory / "job" / "seed.md").write_text(SEED, encoding="utf-8")
    return directory


def test_a_run_is_rebuilt_from_its_directory(web):
    """The server keeps no run state of its own: a restart re-reads the
    journal and the log, so another device sees the same book."""
    directory = new_run_dir(web)
    (directory / "job" / "run.log").write_text("old line one\nold line two\n")
    write_job(directory, pid=dead_pid())
    web.module._jobs.clear()

    status = web.client.get(f"/api/runs/{directory.name}").json()
    assert status["state"] == "interrupted"
    assert status["running"] is False
    job = web.module._job(directory.name)
    assert job.snapshot()[0] == ["old line one", "old line two"]


def test_an_exit_status_on_disk_finishes_the_run(web):
    run_id = start(web)
    job = web.module._job(run_id)
    (job.control / "exit.json").write_text(
        json.dumps({"exit_code": 0, "finished": job.started + 30}))

    status = web.client.get(f"/api/runs/{run_id}").json()
    assert status["state"] == "finished"
    assert status["exit_code"] == 0
    assert status["elapsed"] == 30.0
    assert web.client.get(f"/api/runs/{run_id}/log").json()["state"] == "finished"


def test_a_live_child_is_reattached(web, monkeypatch):
    directory = new_run_dir(web)
    write_job(directory, pid=4242)
    web.module._jobs.clear()
    monkeypatch.setattr(web.module, "_running", lambda pid, run_id: True)
    monkeypatch.setattr(web.module, "_tail", lambda *args: None)

    job = web.module._job(directory.name)
    assert job.pid == 4242
    assert job.running is True
    assert web.module._status(job)["state"] == "running"


def test_a_stale_pid_is_not_a_live_book(web):
    """After a restart a pid may belong to something else: the child's command
    line names its run, so a bare pid is not enough."""
    directory = new_run_dir(web)
    write_job(directory, pid=dead_pid())
    web.module._jobs.clear()

    job = web.module._job(directory.name)
    assert job.running is False
    assert web.module._live_jobs() == []


def test_the_tailer_follows_the_log_file(web, tmp_path, monkeypatch):
    (tmp_path / "job").mkdir()
    log = tmp_path / "job" / "run.log"
    log.write_text("first\n")
    job = web.module.Job("t", tmp_path)
    job.pid = 1
    running = {"value": True}
    monkeypatch.setattr(web.module, "_running", lambda pid, run_id: running["value"])

    thread = threading.Thread(target=web.module._tail, args=(job, 6, ""),
                              daemon=True)
    thread.start()
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("second\n")
        handle.flush()
    deadline = time.time() + 5
    while "second" not in job.snapshot()[0] and time.time() < deadline:
        time.sleep(0.02)
    running["value"] = False          # the child exits
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert job.snapshot()[0] == ["second"]


def test_stop_interrupts_the_child(web, monkeypatch):
    run_id = start(web)
    job = web.module._job(run_id)
    job.pid = 4242
    alive = {"value": True}
    monkeypatch.setattr(web.module, "_running", lambda pid, run_id: alive["value"])
    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))
        alive["value"] = False        # the pipeline handled the signal

    monkeypatch.setattr(web.module.os, "kill", fake_kill)
    response = web.client.post(f"/api/runs/{run_id}/stop")
    assert response.status_code == 200
    assert killed == [(4242, signal.SIGINT)]


def test_auto_resume_resumes_the_interrupted_run(web, monkeypatch):
    directory = new_run_dir(web)
    write_job(directory, pid=dead_pid())
    web.module._jobs.clear()
    monkeypatch.setenv(web.module.AUTO_RESUME_ENV, "1")
    monkeypatch.setattr(web.module, "_running", lambda pid, run_id: False)

    web.module._auto_resume()
    assert [job.id for job, _ in web.spawned] == [directory.name]


def test_auto_resume_is_off_by_default(web, monkeypatch):
    directory = new_run_dir(web)
    write_job(directory, pid=dead_pid())
    web.module._jobs.clear()
    monkeypatch.delenv(web.module.AUTO_RESUME_ENV, raising=False)
    web.module._auto_resume()
    assert web.spawned == []


# -------------------------------------------------------------- model list

class FakeModels:
    def __init__(self, status_code=200, payload=None, error=None):
        self.status_code = status_code
        self.payload = payload
        self.error = error

    def json(self):
        if self.error:
            raise AssertionError("should not be parsed")
        return self.payload


def test_a_question_from_a_dead_run_is_not_shown(web):
    """The question file stays after a crash, but nothing can consume the
    answer, so the page must not pop a dead dialog (and answering is refused)."""
    run_id = start(web)
    job = web.module._job(run_id)
    asked = webanswer.write_question(job.control / "ask", ">", default="d")

    assert web.client.get(f"/api/runs/{run_id}/question").json()["pending"] is None
    response = web.client.post(f"/api/runs/{run_id}/answer",
                               json={"id": asked["id"], "answer": "x"})
    assert response.status_code == 409
    assert webanswer.read_answer(job.control / "ask", asked["id"]) is None


def test_models_lists_what_the_endpoint_offers(web, monkeypatch):
    reply = FakeModels(payload={"data": [{"id": "zeta"}, {"id": "alpha"},
                                         {"id": "alpha"}]})
    monkeypatch.setattr(web.module.requests, "get",
                        lambda url, **kwargs: reply)
    data = web.client.get("/api/models").json()
    assert data["models"] == ["alpha", "zeta"]      # sorted and de-duplicated
    assert data["current"] == "test-model"
    assert data["error"] is None


def test_models_reports_a_down_endpoint_without_failing(web, monkeypatch):
    def down(url, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(web.module.requests, "get", down)
    data = web.client.get("/api/models").json()
    assert data["models"] == []
    assert "connection refused" in data["error"]
    assert data["current"] == "test-model"          # the page can still start one


def test_the_pipeline_archive_never_eats_the_web_seed(web):
    """The run directory is the pipeline's output.directory, and a fresh start
    moves its top-level artifacts (including its own master seed.md) into
    archive/. The web layer's files must live outside that set."""
    from shared.llm_client import load_config
    from shared.output import archive_previous_run

    directory = new_run_dir(web)
    (directory / "state").mkdir()
    (directory / "state" / "bible.json").write_text("{}")
    (directory / "seed.md").write_text("# the pipeline's master seed\n")
    config = web.root / "archive-check.yaml"
    config.write_text(f'output:\n  directory: "{directory}"\n')
    load_config(config)

    archived = archive_previous_run()

    assert archived is not None
    assert not (directory / "state").exists()            # the pipeline's, moved
    assert not (directory / "seed.md").exists()
    assert (directory / "job" / "seed.md").exists()      # ours, untouched
    assert (directory / "job" / "seed.md").read_text() == SEED


def test_a_searxng_url_override_points_runs_at_the_sidecar(web, monkeypatch):
    import yaml

    monkeypatch.setenv(web.module.SEARXNG_URL_ENV, "http://searxng:8080")
    run_id = start(web)
    job = web.module._job(run_id)
    config = yaml.safe_load((job.control / ".." / "config.yaml").read_text())
    assert config["web_search"]["searxng_url"] == "http://searxng:8080"


def test_resume_refreshes_the_endpoint_overrides(web, monkeypatch):
    """A run keeps its config, but the container's LLM/SearXNG addresses are
    deployment facts: resuming must not go back to the old ones."""
    import yaml

    run_id = start(web)
    config_path = web.module._job(run_id).directory / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["llm"]["base_url"] = "http://old:1"
    cfg["web_search"] = {"searxng_url": "http://localhost:8888"}
    cfg["book"]["num_chapters"] = 7        # a book setting, not an endpoint
    config_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setenv(web.module.LLM_URL_ENV, "http://new:2")
    monkeypatch.setenv(web.module.SEARXNG_URL_ENV, "http://searxng:8080")
    assert web.client.post("/api/runs", json={"resume": run_id}).status_code == 200

    updated = yaml.safe_load(config_path.read_text())
    assert updated["llm"]["base_url"] == "http://new:2"
    assert updated["web_search"]["searxng_url"] == "http://searxng:8080"
    assert updated["book"]["num_chapters"] == 7
