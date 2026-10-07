"""Tee everything the program prints (stdout and stderr, so tracebacks too)
into a log file, so a long unattended run can be reviewed afterwards."""

import sys
import time
from pathlib import Path

_state = {"streams": None, "file": None}


class _Tee:
    """File-like object writing to the original stream and the log file."""

    def __init__(self, stream, logfile):
        self._stream, self._file = stream, logfile

    def write(self, text):
        self._stream.write(text)
        try:
            self._file.write(text)
            self._file.flush()          # a crash must not lose the tail
        except (OSError, ValueError):
            pass
        return len(text)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def __getattr__(self, name):        # encoding, fileno, ...
        return getattr(self._stream, name)


def start_log(directory):
    """Start teeing output into <directory>/run-<timestamp>.log.
    Returns the log path, or None if the file can't be opened (the run
    carries on unlogged). Calling it again restarts logging."""
    stop_log()
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"run-{time.strftime('%Y%m%d-%H%M%S')}.log"
        logfile = open(path, "a", encoding="utf-8")
    except OSError as e:
        print(f"[LOG] Could not open a log file in {directory}: {e}")
        return None
    _state["streams"] = (sys.stdout, sys.stderr)
    _state["file"] = logfile
    sys.stdout = _Tee(sys.stdout, logfile)
    sys.stderr = _Tee(sys.stderr, logfile)
    return path


def stop_log():
    """Restore the original streams and close the log file."""
    if _state["streams"]:
        sys.stdout, sys.stderr = _state["streams"]
    if _state["file"]:
        try:
            _state["file"].close()
        except OSError:
            pass
    _state["streams"] = _state["file"] = None
