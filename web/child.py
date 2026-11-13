"""Run the pipeline as one detached job and record how it ended.

The web server starts this module (``python -m web.child ...``) instead of
``main.py`` for three reasons:

- the book must outlive the server: a restart, a redeploy or a closed laptop
  must not kill a six-hour run. The server starts it in a new session and its
  output goes straight to ``<run>/run.log``, so nobody has to be listening;
- the exit code has to be on disk. A ``Popen`` object the server forgot when it
  died knows nothing; ``<run>/exit.json`` survives, so the page can say
  "finished" instead of "interrupted" after a restart;
- arguments are passed through unchanged, so a run is still exactly the CLI.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path


def record(path, exit_code):
    """Write the exit status atomically (a kill mid-write must not corrupt it)."""
    from shared.output import atomic_write_text
    atomic_write_text(Path(path), json.dumps(
        {"exit_code": exit_code, "finished": time.time()}))


def main(argv):
    exit_file = Path(os.environ["BOOKWRITER_EXIT_FILE"])
    code = 1
    try:
        sys.argv = ["main.py", *argv]
        from main import main as pipeline
        pipeline()
        code = 0
    except SystemExit as e:               # main() reports through sys.exit
        if isinstance(e.code, int):
            code = e.code
        else:
            code = 0 if e.code is None else 1
    except KeyboardInterrupt:
        code = 130                        # what Ctrl-C would have returned
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        try:
            record(exit_file, code)
        except OSError as e:              # the log is still the evidence
            print(f"[WEB] Could not record the exit status: {e}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
