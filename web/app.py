"""Web GUI for the book writer.

A thin shell around the CLI: every book is `main.py` in its own subprocess,
with its own directory (config, seed, state, chapters, logs). Nothing in the
pipeline is imported, so the module-level config/context/stats globals stay
exactly as each run left them, a crash cannot take the server down, and
"stop" can interrupt a model call that a thread never could.

The server holds no run state of its own that matters: a run's journal is in
its directory, so any device can pick it up and a server restart loses
nothing (see web/child.py and JOB_FILE / EXIT_FILE below).

Directory layout per book (default under output/web/, BOOKWRITER_WEB_OUTPUT):

    <run>/config.yaml      generated: output.directory points back at <run>
    <run>/job/seed.md      the author's seed (passed with --seed)
    <run>/job/job.json     {id, pid, started, options}: is it running, and how
    <run>/job/exit.json    {exit_code, finished}: written by the child when it ends
    <run>/job/run.log      the child's stdout+stderr (the page's live log)
    <run>/job/ask/         question.json / answer.json (shared/webanswer.py)
    <run>/state/ ...       the pipeline's own output, beside its master seed.md

Run it with:  uv run --extra web uvicorn web.app:app --port 8000
It is a single-user tool with no authentication: bind it to a trusted network.
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared import webanswer

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(os.environ.get("BOOKWRITER_WEB_OUTPUT") or ROOT / "output" / "web")
CONFIG_PATH = Path(os.environ.get("BOOKWRITER_CONFIG") or ROOT / "config.yaml")
# In a container the LLM and SearXNG are not on localhost: the compose file
# passes these through so config.yaml keeps working on the host unchanged.
LLM_URL_ENV = "BOOKWRITER_LLM_BASE_URL"
SEARXNG_URL_ENV = "BOOKWRITER_SEARXNG_URL"
# resume a run that was interrupted (server/container restart), once, on startup
AUTO_RESUME_ENV = "BOOKWRITER_AUTO_RESUME"
STATIC = Path(__file__).resolve().parent / "static"
# Everything the web layer owns lives in one subdirectory: the pipeline treats
# the run directory as its output.directory and moves top-level artifacts
# (state/, chapters/, its own master seed.md, the book) into archive/ on a
# fresh start, so web files at the top level would be eaten by it.
JOB_DIR = "job"
ASK = "ask"             # inside JOB_DIR: question.json / answer.json
JOB_FILE = "job.json"   # inside JOB_DIR
EXIT_FILE = "exit.json" # inside JOB_DIR
LOG_TAIL = 30           # log lines shown above a pending question


# --------------------------------------------------------------- the runs

@dataclass
class Job:
    """One book run: what is known about it, plus the live log buffer."""
    id: str
    directory: Path
    pid: int | None = None
    options: dict = field(default_factory=dict)
    exit_code: int | None = None
    started: float = 0.0
    finished: float | None = None
    lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self, start=0):
        with self.lock:
            return self.lines[start:], len(self.lines)

    def append(self, line):
        with self.lock:
            self.lines.append(str(line))

    @property
    def control(self):
        """Where this run's web-layer files live (see JOB_DIR)."""
        return self.directory / JOB_DIR

    @property
    def running(self):
        return _running(self.pid, self.id)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path, data):
    """Atomic, like every other write in the project: a server killed while
    recording a job must not leave a half-written journal it cannot re-read."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True            # someone else's process: it is alive
    except (ValueError, OSError):
        return False
    return True


def _running(pid, run_id):
    """True when the pid is alive AND is this run's child.

    After a restart a pid can be reused by an unrelated process; the child's
    command line carries its run id (--config <run>/config.yaml), so check it
    where /proc exists and trust the pid elsewhere (macOS).
    """
    if not _alive(pid):
        return False
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return True
    return str(run_id).encode() in cmdline


def _base_config():
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise HTTPException(
            500, f"No config file at {CONFIG_PATH}. Copy config.example.yaml "
                 "to config.yaml (or set BOOKWRITER_CONFIG) and restart.")
    except yaml.YAMLError as e:
        raise HTTPException(500, f"{CONFIG_PATH} is not valid YAML: {e}")


def _valid_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id or ""):
        raise HTTPException(400, "bad run id")
    return run_id


def _directory(run_id):
    path = (BASE_DIR / _valid_id(run_id)).resolve()
    if not path.is_dir() or not path.is_relative_to(BASE_DIR.resolve()):
        raise HTTPException(404, f"no run '{run_id}'")
    return path


def _tail(job, offset, partial):
    """Follow run.log into the job's buffer.

    The child writes the file itself (no pipe), so this tailer is the only
    reader: it can start at any offset, which is what makes re-attaching
    after a server restart possible. Stops once the child is gone and the
    file has been drained.
    """
    path = job.control / "run.log"
    while True:
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                data = handle.read()
                offset = handle.tell()
        except OSError:
            data = b""
        if data:
            text = partial + data.decode("utf-8", "replace")
            lines = text.split("\n")
            partial = lines.pop()
            for line in lines:
                job.append(line)
        if not job.running and not data:
            if partial:
                job.append(partial)
            return
        time.sleep(0.2)


def _load_job(directory):
    """Rebuild a job from its directory: the journal, the exit status and the
    log written so far. This is what makes the backend persistent."""
    job = Job(directory.name, directory)
    meta = _read_json(job.control / JOB_FILE) or {}
    job.options = dict(meta.get("options") or {})
    job.started = float(meta.get("started") or 0.0)
    pid = meta.get("pid")
    if _running(pid, job.id):
        job.pid = int(pid)
    exit_record = _read_json(job.control / EXIT_FILE) or {}
    if exit_record:
        code = exit_record.get("exit_code")
        job.exit_code = int(code) if isinstance(code, (int, float)) else None
        finished = exit_record.get("finished")
        job.finished = float(finished) if isinstance(finished, (int, float)) else None
    raw = b""
    try:
        raw = (job.control / "run.log").read_bytes()
    except OSError:
        pass
    lines = raw.decode("utf-8", "replace").split("\n")
    partial = lines.pop()
    job.lines = lines
    if job.running:
        threading.Thread(target=_tail, args=(job, len(raw), partial),
                         daemon=True).start()
    return job


def _job(run_id):
    with _jobs_lock:
        job = _jobs.get(run_id)
        if job:
            return job
        job = _load_job(_directory(run_id))
        _jobs[run_id] = job
        return job


def _live_jobs():
    """Every run whose child is alive, read from disk so a server restart
    cannot forget one (two books at once would overwrite each other's GPU)."""
    found: dict[str, Job] = {}
    if BASE_DIR.is_dir():
        for path in BASE_DIR.iterdir():
            meta = _read_json(path / JOB_DIR / JOB_FILE) or {}
            if _running(meta.get("pid"), path.name):
                found[path.name] = _job(path.name)
    for job in list(_jobs.values()):
        if job.running:
            found[job.id] = job
    return list(found.values())


def _title(directory):
    bible = _read_json(directory / "state" / "bible.json") or {}
    if bible.get("title"):
        return str(bible["title"])
    # the pipeline's master seed (expanded) first, then the author's own
    for name in ("seed.md", f"{JOB_DIR}/seed.md"):
        for line in _read_text(directory / name).splitlines():
            text = line.strip().lstrip("#").strip()
            if text:
                return text
    return "Untitled"


def _chapters(directory):
    """Outline chapters with what exists for each, straight from state/."""
    outline = _read_json(directory / "state" / "outline.json") or []
    drafts = _read_json(directory / "state" / "drafts.json") or {}
    final = _read_json(directory / "state" / "final.json") or {}
    research = _read_json(directory / "state" / "research.json") or {}
    chapters = []
    for entry in outline:
        number = entry.get("number")
        key = str(number)
        chapters.append({
            "number": number,
            "title": entry.get("title", ""),
            "summary": entry.get("summary", ""),
            "researched": key in research,
            "drafted": key in drafts,
            "edited": key in final,
        })
    return chapters


def _files(directory):
    found = []
    for path in sorted(directory.glob("*.md")) + sorted(directory.glob("*.epub")):
        found.append({"path": path.name, "size": path.stat().st_size})
    for path in sorted((directory / "chapters").glob("*.md")):
        found.append({"path": f"chapters/{path.name}", "size": path.stat().st_size})
    return found


def _refresh_exit(job):
    """Pick up an exit status written after this job was loaded."""
    if job.exit_code is not None or job.running:
        return
    record = _read_json(job.control / EXIT_FILE) or {}
    if not record:
        return
    code = record.get("exit_code")
    job.exit_code = int(code) if isinstance(code, (int, float)) else None
    finished = record.get("finished")
    job.finished = float(finished) if isinstance(finished, (int, float)) else None


def _state(job):
    if job.running:
        return "running"
    if job.exit_code is not None:
        return "finished"
    if job.started:
        return "interrupted"      # started once, ended without recording a code
    return "new"


def _status(job):
    directory = job.directory
    _refresh_exit(job)
    chapters = _chapters(directory)
    state = _state(job)
    end = job.finished or (time.time() if state == "running" else None)
    log = job.control / "run.log"
    return {
        "id": job.id,
        "state": state,
        "running": state == "running",
        "exit_code": job.exit_code,
        "elapsed": round(end - job.started, 1) if job.started and end else None,
        "last_output": log.stat().st_mtime if log.exists() else None,
        "title": _title(directory),
        "chapters": chapters,
        "counts": {
            "planned": len(chapters),
            "researched": sum(1 for c in chapters if c["researched"]),
            "drafted": sum(1 for c in chapters if c["drafted"]),
            "edited": sum(1 for c in chapters if c["edited"]),
        },
        "files": _files(directory),
        "log_lines": job.snapshot()[1],
    }


def _write_run_config(path, output_dir, model=None):
    cfg = _base_config()
    cfg.setdefault("output", {})["directory"] = str(output_dir)
    if model:
        cfg.setdefault("llm", {})["model"] = model
    base_url = os.environ.get(LLM_URL_ENV, "").strip()
    if base_url:
        cfg.setdefault("llm", {})["base_url"] = base_url
    searxng_url = os.environ.get(SEARXNG_URL_ENV, "").strip()
    if searxng_url:
        cfg.setdefault("web_search", {})["searxng_url"] = searxng_url
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


def _refresh_env_overrides(path):
    """Re-apply the server's endpoint overrides to an existing run's config.

    A run keeps the config it was created with (that is what --config means),
    but where the LLM server and SearXNG are is a fact about this deployment,
    not about the book: resuming after the container moved must not send the
    run back to an old address.
    """
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return
    wanted = {
        "llm": ("base_url", os.environ.get(LLM_URL_ENV, "").strip()),
        "web_search": ("searxng_url", os.environ.get(SEARXNG_URL_ENV, "").strip()),
    }
    changed = False
    for section, (key, value) in wanted.items():
        if value and (cfg.get(section) or {}).get(key) != value:
            cfg.setdefault(section, {})[key] = value
            changed = True
    if changed:
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")


def _command(job, options):
    """The CLI invocation for a job (kept separate so tests can read it)."""
    cmd = [sys.executable, "-u", "-m", "web.child",
           "--config", str(job.directory / "config.yaml"),
           "--seed", str(job.control / "seed.md")]
    if options.get("chapters"):
        cmd += ["-c", str(int(options["chapters"]))]
    if options.get("plan_only"):
        cmd += ["--plan-only"]
    if options.get("seed_review"):
        cmd += ["--seed-review", str(options["seed_review"])]
    if options.get("no_resume"):
        cmd += ["--no-resume"]
    return cmd


def _environment(job):
    """The child's environment: unbuffered output (the log is streamed live),
    the browser question channel, and where to record its exit status."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env[webanswer.ENV_DIR] = str(job.control / ASK)
    env["BOOKWRITER_EXIT_FILE"] = str(job.control / EXIT_FILE)
    return env


def _spawn(job, options=None):
    """Start the child for this job, detached from the server.

    stdout goes straight to run.log and the process gets its own session, so
    the book keeps being written if the server is restarted or the browser
    goes away. Separated out so tests can stub it.
    """
    options = options if options is not None else (job.options or {})
    job.options = options
    try:
        (job.control / EXIT_FILE).unlink()        # a stale code would lie
    except OSError:
        pass
    log_path = job.control / "run.log"
    offset = log_path.stat().st_size if log_path.exists() else 0
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            _command(job, options), cwd=str(ROOT), env=_environment(job),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    job.pid, job.started = proc.pid, time.time()
    job.exit_code, job.finished = None, None
    _write_json(job.control / JOB_FILE, {
        "id": job.id, "pid": proc.pid, "started": job.started,
        "options": options})
    threading.Thread(target=_tail, args=(job, offset, ""), daemon=True).start()
    return proc


def _auto_resume():
    """Resume the newest interrupted run, once, at startup.

    Off unless BOOKWRITER_AUTO_RESUME is set: a run that stopped because the
    LLM was down should not be restarted in a loop behind the user's back.
    """
    if os.environ.get(AUTO_RESUME_ENV, "").strip().lower() not in ("1", "true", "yes"):
        return
    if _live_jobs() or not BASE_DIR.is_dir():
        return
    for path in sorted(BASE_DIR.iterdir(), reverse=True):
        if not (path / JOB_DIR / JOB_FILE).exists():
            continue
        job = _job(path.name)
        if _state(job) != "interrupted":
            continue
        print(f"[WEB] Auto-resuming the interrupted run {job.id}")
        try:
            _spawn(job)
        except OSError as e:
            print(f"[WEB] Could not resume {job.id}: {e}")
        return


@asynccontextmanager
async def _lifespan(_app):
    try:
        _auto_resume()
    except Exception as e:                    # never stop the server for it
        print(f"[WEB] Auto-resume failed: {e}")
    yield


app = FastAPI(title="Multi-Agent Book Writer", lifespan=_lifespan)


# -------------------------------------------------------------- the API

class RunRequest(BaseModel):
    seed: str = ""
    chapters: int | None = None
    model: str | None = None
    plan_only: bool = False
    seed_review: str | None = None
    no_resume: bool = False
    resume: str | None = None      # an existing run id: continue that book


class AnswerRequest(BaseModel):
    id: str
    answer: str


def _seed_review_name(value):
    """YAML turns an unquoted on/off into a boolean; name it for the page."""
    if isinstance(value, bool):
        return "ask" if value else "off"
    return str(value)


def _llm_url(cfg):
    """Base URL as the pipeline would use it: the container override first,
    then config.yaml, without a trailing slash or '/v1'."""
    base = os.environ.get(LLM_URL_ENV) or (cfg.get("llm") or {}).get("base_url") or ""
    base = str(base).rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def _llm_key(cfg):
    raw = str((cfg.get("llm") or {}).get("api_key") or "").strip()
    if raw.startswith("env:"):
        raw = os.environ.get(raw[4:].strip(), "")
    return raw or os.environ.get("LLM_API_KEY", "")


@app.get("/api/config")
def api_config():
    cfg = _base_config()
    book = cfg.get("book") or {}
    return {
        "model": (cfg.get("llm") or {}).get("model", ""),
        "num_chapters": book.get("num_chapters", 5),
        "words_per_chapter": book.get("words_per_chapter", 2000),
        "seed_review": _seed_review_name(book.get("seed_review", "ask")),
        "output": (cfg.get("output") or {}).get("directory", "output"),
        "web_output": str(BASE_DIR),
        "config_path": str(CONFIG_PATH),
        "auto_resume": os.environ.get(AUTO_RESUME_ENV, "").strip().lower()
        in ("1", "true", "yes"),
    }


@app.get("/api/models")
def api_models():
    """The models this endpoint offers, for the page's dropdown.

    Never raises: the page falls back to the configured model and shows the
    error, because a model list is a convenience, not a requirement.
    """
    cfg = _base_config()
    current = str((cfg.get("llm") or {}).get("model") or "")
    url = _llm_url(cfg)
    reply = {"current": current, "url": url, "models": [], "error": None}
    if not url:
        reply["error"] = "llm.base_url is not set in config.yaml"
        return reply
    headers = {"Content-Type": "application/json"}
    key = _llm_key(cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        response = requests.get(f"{url}/v1/models", headers=headers, timeout=6)
    except Exception as e:                       # show it, never fail the page
        reply["error"] = f"could not reach {url} ({e})"
        return reply
    if response.status_code in (404, 405):
        reply["error"] = "the endpoint does not list its models"
        return reply
    if response.status_code >= 400:
        reply["error"] = f"HTTP {response.status_code} from {url}"
        return reply
    try:
        data = response.json().get("data", [])
    except ValueError:
        reply["error"] = "the endpoint's model list was not JSON"
        return reply
    reply["models"] = sorted({str(m.get("id")) for m in data if m.get("id")})
    if not reply["models"]:
        reply["error"] = "the endpoint listed no models"
    return reply


@app.get("/api/runs")
def api_runs():
    runs = []
    if BASE_DIR.is_dir():
        for path in sorted(BASE_DIR.iterdir(), reverse=True):
            if not (path / "config.yaml").exists():
                continue
            status = _status(_job(path.name))
            runs.append({k: status[k] for k in
                         ("id", "state", "title", "counts", "elapsed",
                          "last_output")})
    return {"runs": runs, "live": [job.id for job in _live_jobs()]}


_start_lock = threading.Lock()


@app.post("/api/runs")
def api_start(request: RunRequest):
    # two phones pressing Start at once must not start two books
    with _start_lock:
        return _start(request)


def _start(request: RunRequest):
    if _live_jobs():
        raise HTTPException(409, "a book is already being written; wait for it "
                                 "to finish or stop it first")
    if request.resume:
        job = _job(request.resume)
        if not (job.control / "seed.md").exists():
            raise HTTPException(409, "that run has no seed file to resume")
        if not job.options:
            job.options = request.model_dump()
        _refresh_env_overrides(job.directory / "config.yaml")
        _spawn(job)          # resumes from state/ with the run's own settings
        return {"id": job.id}

    if not request.seed.strip():
        raise HTTPException(400, "a seed is required")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    directory = BASE_DIR / run_id
    (directory / JOB_DIR).mkdir(parents=True, exist_ok=True)
    (directory / JOB_DIR / "seed.md").write_text(request.seed, encoding="utf-8")
    _write_run_config(directory / "config.yaml", directory, request.model)
    job = Job(run_id, directory)
    job.options = {k: v for k, v in request.model_dump().items()
                   if k not in ("seed", "resume")}
    with _jobs_lock:
        _jobs[run_id] = job
    _spawn(job, job.options)
    return {"id": job.id}


@app.post("/api/runs/{run_id}/stop")
def api_stop(run_id: str):
    job = _job(run_id)
    if not job.running:
        raise HTTPException(409, "that run is not running")
    # SIGINT, not terminate: the pipeline catches KeyboardInterrupt, saves the
    # partial book and leaves a resumable state rather than dying mid-write.
    try:
        os.kill(job.pid, signal.SIGINT)
    except ProcessLookupError:
        raise HTTPException(409, "that run is not running")
    for _ in range(40):
        if not job.running:
            break
        time.sleep(0.25)
    if job.running:
        try:
            os.kill(job.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _refresh_exit(job)
    return {"id": job.id, "stopped": True}


@app.get("/api/runs/{run_id}")
def api_status(run_id: str):
    return _status(_job(run_id))


@app.get("/api/runs/{run_id}/log")
def api_log(run_id: str, since: int = 0):
    job = _job(run_id)
    lines, total = job.snapshot(max(0, since))
    return {"lines": lines, "next": total, "exit_code": job.exit_code,
            "state": _state(job)}


@app.get("/api/runs/{run_id}/events")
async def api_events(run_id: str, since: int = 0):
    """Server-sent events: one `data:` line per output line, then `done`.

    Works after a restart too: the buffer is rebuilt from run.log, so a phone
    that reconnects sees the history and then the live tail.
    """
    job = _job(run_id)
    index = max(0, since)

    async def stream():
        nonlocal index
        quiet = 0
        while True:
            lines, total = job.snapshot(index)
            for line in lines:
                yield f"data: {json.dumps(line)}\n\n"
            index = total
            state = _state(job)
            if state in ("finished", "interrupted") and index >= total:
                yield f"event: done\ndata: {state}\n\n"
                return
            quiet = 0 if lines else quiet + 1
            if quiet > 36000:           # an hour with no output: the page is gone
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/question")
def api_question(run_id: str):
    job = _job(run_id)
    question = webanswer.read_question(job.control / ASK)
    if not question:
        return {"pending": None}
    lines, _ = job.snapshot()
    return {"pending": question, "context": lines[-LOG_TAIL:]}


@app.post("/api/runs/{run_id}/answer")
def api_answer(run_id: str, request: AnswerRequest):
    job = _job(run_id)
    question = webanswer.read_question(job.control / ASK)
    if not question:
        raise HTTPException(409, "there is no pending question")
    if question.get("id") != request.id:
        raise HTTPException(409, "that question has already been answered")
    webanswer.write_answer(job.control / ASK, request.id, request.answer)
    return {"id": request.id, "answered": True}


@app.get("/api/runs/{run_id}/file")
def api_file(run_id: str, path: str):
    directory = _directory(run_id)
    target = (directory / path).resolve()
    if not target.is_relative_to(directory) or not target.is_file():
        raise HTTPException(404, f"no file '{path}' in this run")
    return FileResponse(target, filename=target.name)


@app.post("/api/runs/{run_id}/epub")
def api_epub(run_id: str):
    """Build the EPUB beside the book (make_epub.py, no LLM)."""
    directory = _directory(run_id)
    result = subprocess.run(
        [sys.executable, "make_epub.py", "--config",
         str(directory / "config.yaml")],
        cwd=str(directory), text=True, capture_output=True, timeout=300)
    if result.returncode:
        raise HTTPException(400, (result.stderr or result.stdout or
                                  "could not build the EPUB").strip())
    return {"output": result.stdout.strip()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
