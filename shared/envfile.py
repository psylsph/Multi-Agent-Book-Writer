"""
The .env file for the CLI.

`docker compose` reads `.env` next to docker-compose.yml; the CLI reads the
same file, so an API key (TINYFISH_API_KEY, LLM_API_KEY, ...) put there once
works for both. Standard library only: KEY=VALUE lines, `#` comments, an
optional `export `, single or double quotes.

The real environment always wins over the file. One exception to "load
everything": the endpoint variables (BOOKWRITER_LLM_BASE_URL,
BOOKWRITER_SEARXNG_URL) in .env are usually written for the container
(http://host.docker.internal:8080, http://searxng:8080), and those names do
not resolve on the host. Loading one would send the CLI to a server it cannot
reach, so a .env endpoint whose host does not resolve here is skipped, with a
line saying so, and config.yaml's address stands.
"""

import os
import re
import socket
from urllib.parse import urlparse

from shared.llm_client import ENDPOINT_ENV

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse(text):
    """{name: value} from .env text. Malformed lines are ignored."""
    values = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, raw = m.groups()
        if raw[:1] in ("'", '"'):
            end = raw.find(raw[0], 1)
            value = raw[1:end] if end > 0 else raw[1:]
        else:
            value = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        values[name] = value
    return values


def host_resolves(url):
    """True when the URL's host can be looked up from this machine."""
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    return True


def load(path, resolves=host_resolves):
    """Put the variables in `path` into os.environ, without replacing any that
    are already set. Returns the names loaded. Prints the names (never the
    values: the file holds API keys) and every endpoint it skipped."""
    try:
        with open(path, encoding="utf-8") as f:
            values = parse(f.read())
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as e:
        print(f"[CONFIG] Could not read {path} ({e}); ignoring it.")
        return []
    endpoints = {env for _, _, env in ENDPOINT_ENV}
    loaded = []
    for name, value in values.items():
        if name in os.environ or not value:
            continue
        if name in endpoints and not resolves(value):
            print(f"[CONFIG] {path}: skipping {name}={value} - that host does "
                  "not resolve here (a Docker-only address?); config.yaml's "
                  f"address is used. Export {name} to override it for the CLI.")
            continue
        os.environ[name] = value
        loaded.append(name)
    if loaded:
        print(f"[CONFIG] Read {path}: {', '.join(loaded)}")
    return loaded
