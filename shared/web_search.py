"""Web search for the researcher, via a SearXNG instance's JSON API.

Search is opt-in (web_search.enabled). Only generic real-world queries are
ever sent (see safe_queries), results are returned as plain data for the
caller to quote, and every failure degrades to "no search" rather than
stopping a run.

If the configured SearXNG is not reachable and Docker is available,
offer_docker_setup() can pull and start a local container (after asking).
"""

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from shared import config_schema
from shared.llm_client import SEARXNG_URL_ENV, get_config

CONTAINER_NAME = "book-writer-searxng"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

# availability for this process: None = not checked yet
_status = None
# when the last query was sent (time.monotonic()), for web_search.min_interval
_last_query = [None]
PACING = True           # off in the test suite: stubbed searches have no limit


class SearchError(RuntimeError):
    """A search could not be completed (unreachable, refused, bad reply)."""


# ------------------------------------------------------------------ settings

def settings():
    return get_config()["web_search"]


def enabled():
    return bool(settings().get("enabled"))


def _clean(text):
    """Collapse whitespace and drop control characters."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text or ""))
    return " ".join(text.split())


# -------------------------------------------------------------------- search

PROVIDERS = config_schema.SEARCH_PROVIDERS
PARALLEL_KEY_ENV = "PARALLEL_API_KEY"
_warned = set()                       # providers already reported as failing


def providers():
    """web_search.providers: the search backends to ask, in order. Unknown
    names are dropped (config_schema warns about them); none -> searxng."""
    raw = settings().get("providers") or ["searxng"]
    if isinstance(raw, str):
        raw = [raw]
    chosen = [str(p).strip().lower() for p in raw]
    return [p for p in dict.fromkeys(chosen) if p in PROVIDERS] or ["searxng"]


def search(query, objective=None, only=None):
    """Run one query on every configured provider (or just `only`); returns
    [{title, url, snippet}] merged in provider order, one entry per URL.

    objective: what the query is meant to find, in words (the claim being
    checked); Parallel uses it to pick the excerpts, SearXNG ignores it.
    Raises SearchError only when every provider failed: one backend down
    must not cost the others' results."""
    ws = settings()
    names = [only] if only else providers()
    _pace(float(ws.get("min_interval") or 0))
    results, seen, errors = [], set(), []
    for name in names:
        try:
            found = _SEARCHES[name](ws, query, objective or query)
        except SearchError as e:
            errors.append(f"{name}: {e}")
            if len(names) > 1 and name not in _warned:
                _warned.add(name)
                print(f"[SEARCH] {name} failed ({e}); using the other "
                      "provider(s).")
            continue
        for item in found[:int(ws["results_per_query"])]:
            if item["url"] not in seen:
                seen.add(item["url"])
                results.append(item)
    if errors and len(errors) == len(names):
        raise SearchError("; ".join(errors))
    return results


def _searxng_search(ws, query, objective):
    """One query against SearXNG's JSON API."""
    url = str(ws["searxng_url"]).rstrip("/") + "/search"
    try:
        response = requests.get(
            url, timeout=ws["timeout"],
            params={"q": query, "format": "json",
                    "categories": ws["categories"]})
    except requests.exceptions.RequestException as e:
        raise SearchError(f"could not reach SearXNG at {url} ({e})") from e
    if response.status_code == 403:
        raise SearchError(
            "SearXNG refused the request (HTTP 403). Enable the 'json' "
            "format under search.formats in its settings.yml.")
    if response.status_code != 200:
        raise SearchError(f"SearXNG returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as e:
        raise SearchError("SearXNG did not return JSON") from e

    results = []
    for item in data.get("results") or []:
        link = item.get("url") if isinstance(item, dict) else None
        if not link or not str(link).startswith(("http://", "https://")):
            continue
        results.append({
            "title": _clean(item.get("title"))[:150],
            "url": str(link),
            "snippet": _clean(item.get("content"))[:int(ws["snippet_chars"])],
        })
        if len(results) >= int(ws["results_per_query"]):
            break
    return results


# ------------------------------------------------- Parallel (an MCP server)
#
# Parallel's search is a remote MCP server (Streamable HTTP, JSON-RPC 2.0):
# `initialize` opens a session (Mcp-Session-Id header), then `tools/call`
# runs its `web_search` tool with an objective and keyword queries. It works
# without a key at a free rate limit; a key (web_search.parallel_api_key or
# PARALLEL_API_KEY) raises it. Plain requests: no MCP SDK dependency.

_PARALLEL = {"session": None, "next_id": 1}


def _parallel_key(ws):
    """The Parallel key: 'env:VAR', a plain value, or PARALLEL_API_KEY. Never
    the LLM's key - that must not be sent to a search provider."""
    raw = str(ws.get("parallel_api_key") or "").strip()
    if raw.startswith("env:"):
        raw = os.environ.get(raw[4:].strip(), "")
    return raw or os.environ.get(PARALLEL_KEY_ENV, "")


def _parallel_rpc(ws, method, params=None, notify=False):
    """One JSON-RPC message to the Parallel MCP server; returns (response,
    the reply's message or None)."""
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    key = _parallel_key(ws)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if _PARALLEL["session"]:
        headers["Mcp-Session-Id"] = _PARALLEL["session"]
    body = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    if not notify:
        body["id"] = _PARALLEL["next_id"]
        _PARALLEL["next_id"] += 1
    url = str(ws.get("parallel_url") or "https://mcp.parallel.ai/mcp")
    try:
        response = requests.post(url, headers=headers, json=body,
                                 timeout=ws["timeout"])
    except requests.exceptions.RequestException as e:
        raise SearchError(f"could not reach Parallel at {url} ({e})") from e
    if notify or response.status_code >= 400:
        return response, None
    # An SSE stream without a charset: requests would decode it as Latin-1
    # and every degree sign and dash in an excerpt would come out mangled.
    text = response.content.decode("utf-8", "replace")
    for line in reversed(text.splitlines()):
        if line.startswith("data:"):
            text = line[5:].strip()
            break
    try:
        return response, json.loads(text)
    except ValueError as e:
        raise SearchError("Parallel did not return JSON-RPC") from e


def _parallel_open(ws):
    _PARALLEL["session"] = None
    response, reply = _parallel_rpc(ws, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "multi-agent-book-writer", "version": "1"}})
    if response.status_code >= 400 or not reply or "error" in reply:
        raise SearchError(f"Parallel refused the session "
                          f"(HTTP {response.status_code})")
    _PARALLEL["session"] = response.headers.get("mcp-session-id")
    _parallel_rpc(ws, "notifications/initialized", notify=True)


def _parallel_search(ws, query, objective):
    """Parallel's web_search tool; a stale session is reopened once."""
    for attempt in (1, 2):
        if attempt == 2 or not _PARALLEL["session"]:
            _parallel_open(ws)
        response, reply = _parallel_rpc(ws, "tools/call", {
            "name": "web_search",
            "arguments": {"objective": objective, "search_queries": [query]}})
        if response.status_code in (400, 404) and attempt == 1:
            continue                       # the session expired: open anew
        if response.status_code == 429:
            raise SearchError("Parallel's rate limit was reached (HTTP 429); "
                              "set web_search.parallel_api_key for a higher one")
        if response.status_code >= 400:
            raise SearchError(f"Parallel returned HTTP {response.status_code}")
        break
    if not reply or "error" in reply:
        message = ((reply or {}).get("error") or {}).get("message", "no reply")
        raise SearchError(f"Parallel search failed ({message})")
    result = reply.get("result") or {}
    if result.get("isError"):
        text = " ".join(c.get("text", "") for c in result.get("content") or [])
        raise SearchError(f"Parallel search failed ({_clean(text)[:200]})")
    data = result.get("structuredContent")
    if not isinstance(data, dict):
        try:
            data = json.loads((result.get("content") or [{}])[0].get("text", ""))
        except (ValueError, AttributeError, IndexError):
            data = {}
    words = set(_words(f"{objective} {query}"))
    out = []
    for item in data.get("results") or []:
        link = item.get("url") if isinstance(item, dict) else None
        if not link or not str(link).startswith(("http://", "https://")):
            continue
        out.append({"title": _clean(item.get("title"))[:150], "url": str(link),
                    "snippet": best_passages(item.get("excerpts") or [], words,
                                             int(ws["snippet_chars"]))})
    return out


def best_passages(excerpts, words, limit):
    """The excerpt passages that share the most words with the claim, in
    page order, up to `limit` characters. Parallel's excerpts run to
    thousands of characters across a page; cutting them at the start kept
    the page's opening ("The remainder of the county is rural...") and lost
    the sentence that answers the claim."""
    passages = []
    for excerpt in excerpts:
        for part in re.split(r"\n\s*\.\.\.\s*\n", str(excerpt)):
            part = _clean(part)
            if part:
                passages.append(part)
    if not passages:
        return ""
    score = [len(set(words) & set(_words(part))) for part in passages]
    # passages that share no word with the claim are filler: never let them
    # push the one that answers it past the limit (the page's opening is
    # used only when nothing matches at all)
    ranked = sorted((i for i in range(len(passages)) if score[i]),
                    key=lambda i: (-score[i], i)) or [0]
    chosen, used = [], 0
    for i in ranked:
        if chosen and used + len(passages[i]) > limit:
            break
        chosen.append(i)
        used += len(passages[i]) + 5
    return " ... ".join(passages[i] for i in sorted(chosen))[:limit]


# ------------------------------------------------------- TinyFish (REST)
#
# TinyFish Search: GET <tinyfish_url>?query=...&purpose=... with an X-API-Key
# header; it answers {"results": [{title, url, snippet, ...}]}. A key is
# required (web_search.tinyfish_api_key or TINYFISH_API_KEY).

TINYFISH_KEY_ENV = "TINYFISH_API_KEY"


def _tinyfish_key(ws):
    """'env:VAR', a plain value, or TINYFISH_API_KEY - never the LLM key."""
    raw = str(ws.get("tinyfish_api_key") or "").strip()
    if raw.startswith("env:"):
        raw = os.environ.get(raw[4:].strip(), "")
    return raw or os.environ.get(TINYFISH_KEY_ENV, "")


def _tinyfish_search(ws, query, objective):
    key = _tinyfish_key(ws)
    if not key:
        raise SearchError(f"TinyFish needs an API key: set {TINYFISH_KEY_ENV} "
                          "or web_search.tinyfish_api_key")
    url = str(ws.get("tinyfish_url") or "https://api.search.tinyfish.ai/")
    params = {"query": query}
    if objective and objective != query:
        params["purpose"] = objective            # what the query should find
    try:
        response = requests.get(url, timeout=ws["timeout"], params=params,
                                headers={"X-API-Key": key})
    except requests.exceptions.RequestException as e:
        raise SearchError(f"could not reach TinyFish at {url} ({e})") from e
    if response.status_code in (401, 403):
        raise SearchError(f"TinyFish refused the key (HTTP {response.status_code})")
    if response.status_code == 429:
        raise SearchError("TinyFish's rate limit was reached (HTTP 429)")
    if response.status_code != 200:
        raise SearchError(f"TinyFish returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as e:
        raise SearchError("TinyFish did not return JSON") from e
    out = []
    for item in data.get("results") or [] if isinstance(data, dict) else []:
        link = item.get("url") if isinstance(item, dict) else None
        if not link or not str(link).startswith(("http://", "https://")):
            continue
        out.append({"title": _clean(item.get("title"))[:150], "url": str(link),
                    "snippet": _clean(item.get("snippet"))[:int(ws["snippet_chars"])]})
    return out


_SEARCHES = {"searxng": _searxng_search, "parallel": _parallel_search,
             "tinyfish": _tinyfish_search}


def _pace(interval):
    """Wait until `interval` seconds have passed since the previous query.
    A chapter's queries went out back to back: Brave's Search API answered
    the second with 429 Too Many Requests and SearXNG suspended the engine
    for three minutes, so most queries never reached it."""
    if not PACING:
        return
    now = time.monotonic()
    if interval > 0 and _last_query[0] is not None:
        wait = _last_query[0] + interval - now
        if wait > 0:
            time.sleep(wait)
            now += wait
    _last_query[0] = now


def banned_terms():
    """The user's web_search.banned_terms as a clean list of strings."""
    raw = settings().get("banned_terms") or []
    if isinstance(raw, str):
        raw = [raw]
    return [" ".join(str(x).split()) for x in raw if x is not None and str(x).strip()]


def character_terms(bible):
    """Names of the bible's characters that must stay out of a query: each
    full name and each declared alias of TWO or more words ("Aria Vance").
    Single words are left out on purpose: a first name or nickname is often an
    ordinary word ("Rose", "Will"), and dropping every query that contains one
    would gut the fact-checking. List those in web_search.banned_terms."""
    terms = []
    for character in (bible or {}).get("characters") or []:
        name = re.sub(r"\(.*?\)", " ", str(character.get("name", "")))
        for term in [name, *(character.get("aliases") or [])]:
            term = " ".join(str(term).split())
            if len(term.split()) >= 2 and term.lower() not in (
                    t.lower() for t in terms):
                terms.append(term)
    return terms


def blocked_term(query, forbidden_terms):
    """The first banned term the query mentions (whole word, any case)."""
    low = query.lower()
    for term in forbidden_terms:
        if term and len(term) > 1 and re.search(
                rf"(?<!\w){re.escape(term.lower())}(?!\w)", low):
            return term
    return None


def safe_queries(raw, forbidden_terms=(), limit=3, on_drop=None):
    """Clean model-proposed queries before anything leaves the machine.

    Drops non-strings, over-long queries, duplicates, and any query that
    mentions a banned term (web_search.banned_terms), so the names you list
    are never sent to the search provider. Everything else is allowed.
    on_drop(query, term) is called for each query removed for a banned term.
    """
    out = []
    for q in raw if isinstance(raw, list) else []:
        if not isinstance(q, str):
            continue
        q = _clean(q).strip("\"'")
        if not q or len(q) > 120 or q.lower() in (x.lower() for x in out):
            continue
        term = blocked_term(q, forbidden_terms)
        if term:
            if on_drop:
                on_drop(q, term)
            continue
        out.append(q)
        if len(out) >= limit:
            break
    return out


_QUERY_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "with", "by", "from", "how", "what", "do", "does", "uk", "typical",
}
VERDICTS = ("supported", "contradicted", "unclear")


def _words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _stem(word):
    """Crude plural folding ('zones' == 'zone'; 'grass' stays 'grass')."""
    return word[:-1] if len(word) > 3 and word.endswith("s") \
        and not word.endswith("ss") else word


def similar_query(query, seen, threshold=0.6):
    """The earlier query in `seen` that `query` essentially repeats (word
    overlap above `threshold`), or None. Catches reworded repeats such as
    'pampas grass hardiness UK' vs 'pampas grass hardiness zone UK'."""
    mine = {_stem(w) for w in _words(query) if w not in _QUERY_STOPWORDS}
    if not mine:
        return None
    for old in seen:
        theirs = {_stem(w) for w in _words(old)
                  if w not in _QUERY_STOPWORDS}
        if theirs and len(mine & theirs) / len(mine | theirs) >= threshold:
            return old
    return None


def claim_overlap(claim, evidence):
    """(shared, total): how many of the claim's content words (plurals
    folded) also appear in the evidence quote."""
    terms = {_stem(w) for w in _words(claim) if w not in _QUERY_STOPWORDS
             and len(w) > 2}
    have = {_stem(w) for w in _words(evidence)}
    return len(terms & have), len(terms)


def addresses_claim(claim, evidence, query=""):
    """A quote that really exists can still be about something else. It must
    share at least two content words, and a third of them, with the claim
    (one word is enough for a one-word claim). Every result contains the
    words of the search query, so at least one shared word must ALSO be one
    the query did not contain: the part of the claim that is actually being
    checked ('triage ... patients', not just 'hospital wards')."""
    shared, total = claim_overlap(claim, evidence)
    if total <= 1:
        return shared >= 1
    if shared < 2 or shared / total < 0.3:
        return False
    claim_terms = {_stem(w) for w in _words(claim)
                   if w not in _QUERY_STOPWORDS and len(w) > 2}
    asked = {_stem(w) for w in _words(query)}
    specific = claim_terms - asked
    if not specific:
        return True
    return bool(specific & {_stem(w) for w in _words(evidence)})


def quote_in_text(quote, text, min_chars=15):
    """True when `quote` really appears in `text` (ignoring case, spacing
    and punctuation). A quote with '...' must have every piece present, each
    piece at least 6 characters, and at least `min_chars` overall (so a
    one-word 'quote' can't pass)."""
    pieces = [" ".join(_words(p))
              for p in re.split(r"\.{3}|\u2026", str(quote)) if p.strip()]
    haystack = " ".join(_words(text))
    return (bool(pieces) and sum(len(p) for p in pieces) >= min_chars
            and all(len(p) >= 6 and p in haystack for p in pieces))


def validate_verdicts(claims, raw_items):
    """Turn the model's verdicts into trustworthy ones.

    claims: [{"claim", "query", "results": [{n, title, url, snippet}]}]
    raw_items: parsed JSON from the verifier, [{claim_id, verdict, source,
    evidence, note}]. A 'supported'/'contradicted' verdict is kept only when
    its source is one of THAT claim's results and its quote is really in that
    result's text; otherwise it is downgraded to 'unclear'. Returns one
    verdict dict per claim, in order.
    """
    by_id = {}
    for item in raw_items if isinstance(raw_items, list) else []:
        if isinstance(item, dict):
            try:
                by_id[int(item.get("claim_id"))] = item
            except (TypeError, ValueError):
                continue
    out = []
    for i, claim in enumerate(claims, 1):
        item = by_id.get(i, {})
        verdict = str(item.get("verdict", "unclear")).strip().lower()
        if verdict not in VERDICTS:
            verdict = "unclear"
        result, reason = None, ""
        if verdict != "unclear":
            try:
                source = int(item.get("source"))
            except (TypeError, ValueError):
                source = None
            result = next((r for r in claim["results"] if r["n"] == source),
                          None)
            if result is None:
                verdict, reason = "unclear", "no valid source cited"
            elif not quote_in_text(item.get("evidence", ""),
                                   f"{result['title']} {result['snippet']}"):
                verdict, reason = "unclear", "quote not found in the source"
                result = None
            elif not addresses_claim(claim["claim"], item.get("evidence", ""),
                                     claim.get("query", "")):
                verdict, reason = "unclear", "quote does not address the claim"
                result = None
        out.append({
            "claim": claim["claim"], "query": claim["query"],
            "verdict": verdict,
            "evidence": _clean(item.get("evidence")) if result else "",
            "url": result["url"] if result else "",
            "note": _clean(item.get("note")) if verdict == "contradicted"
            else "",
            "reason": reason,
        })
    return out


def second_check_prompt(verdict):
    """Independent yes/no question about ONE verdict: does the quote, taken
    by itself, directly support or contradict the claim?"""
    word = "support" if verdict["verdict"] == "supported" else "contradict"
    # The answer key is "answer", not "agrees": for a CONTRADICTION verdict a
    # model reads {"agrees": true} as "the quote agrees with the claim" and
    # answers false to a quote that does contradict it, so correct
    # contradictions were being thrown away.
    return (f'Claim: {verdict["claim"]}\n'
            f'Quote from {verdict["url"]}: "{verdict["evidence"]}"\n\n'
            f"Question: does this quote, taken by itself, directly {word} "
            "the claim? Answer only from the quote.\n"
            'Reply with ONLY JSON: {"answer": true} if YES, '
            '{"answer": false} if NO.')


def format_claims_for_verifier(claims):
    """Claims with their numbered search results, for the verifier prompt."""
    lines = []
    for i, claim in enumerate(claims, 1):
        lines.append(f"CLAIM {i}: {claim['claim']}")
        lines.append("Results:")
        for r in claim["results"]:
            lines.append(f"[{r['n']}] {r['title']} - {r['url']}")
            if r["snippet"]:
                lines.append(f"    {r['snippet']}")
        if not claim["results"]:
            lines.append("(none)")
        lines.append("")
    return "\n".join(lines).strip()


def format_verified(verdicts):
    """Verified facts for the lore brief ('' when nothing was verified).
    Claims left 'unclear' are deliberately left out."""
    lines = []
    for v in verdicts:
        if v["verdict"] == "supported":
            lines.append(f'- SUPPORTED: {v["claim"]} - "{v["evidence"]}" '
                         f'[{v["url"]}]')
        elif v["verdict"] == "contradicted":
            fix = f" Correct fact: {v['note']}" if v["note"] else ""
            lines.append(f'- CONTRADICTED: {v["claim"]}.{fix} - '
                         f'"{v["evidence"]}" [{v["url"]}]')
    return "\n".join(lines)


def audit_markdown(chapter_number, claims, verdicts):
    """Everything searched and judged, for the interim directory."""
    lines = [f"# Web fact-check - Chapter {chapter_number}", ""]
    for claim, v in zip(claims, verdicts):
        lines += [f"## {claim['claim']}", "", f"- query: `{claim['query']}`",
                  f"- verdict: **{v['verdict']}**"
                  + (f" ({v['reason']})" if v["reason"] else "")]
        if v["evidence"]:
            lines.append(f'- evidence: "{v["evidence"]}" ({v["url"]})')
        if v["note"]:
            lines.append(f"- note: {v['note']}")
        lines.append("- results:")
        lines += ([f"  - [{r['title']}]({r['url']}) - {r['snippet']}"
                   for r in claim["results"]] or ["  - (no results)"])
        lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------- availability

def check(only=None):
    """(ok, message): can we run a search (on `only`, or any provider)?"""
    try:
        search("test", only=only)
        return True, ""
    except SearchError as e:
        return False, str(e)


def is_available():
    """Cached for the process, so a missing server is reported once."""
    global _status
    if _status is None:
        ok, message = check()
        if not ok:
            print(f"[SEARCH] Web search unavailable: {message}")
        _status = ok
    return _status


def reset_status():
    global _status
    _status = None


# ------------------------------------------------------------ docker setup

def _local_port(url):
    """Port of a local SearXNG URL, or None when the URL isn't local."""
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() not in _LOCAL_HOSTS:
        return None
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def config_dir():
    """Where the generated SearXNG settings live (survives output wipes)."""
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "multi-agent-book-writer" / "searxng"


def write_searxng_settings(directory=None):
    """Write a minimal settings.yml: defaults + JSON output enabled (the API
    needs it), limiter off (it would block API clients) and a fresh random
    secret key. An existing file is left alone."""
    directory = Path(directory or config_dir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "settings.yml"
    if not path.exists():
        path.write_text(
            "use_default_settings: true\n"
            "server:\n"
            f'  secret_key: "{secrets.token_hex(32)}"\n'
            "  limiter: false\n"
            "  image_proxy: false\n"
            "search:\n"
            "  formats:\n"
            "    - html\n"
            "    - json\n",
            encoding="utf-8")
    return path


def _docker(args, **kwargs):
    return subprocess.run(["docker", *args], text=True, **kwargs)


def _container_state():
    """'running', 'exited', ... or None when the container doesn't exist."""
    result = _docker(["ps", "-a", "--filter", f"name=^{CONTAINER_NAME}$",
                      "--format", "{{.State}}"],
                     capture_output=True, timeout=30)
    return result.stdout.strip() or None


def reachable(timeout=5):
    """True when something answers at the SearXNG URL. Used to wait for a
    container to come up: check() runs a real search, which SearXNG forwards
    to its upstream engines, and polling with it would send dozens of
    queries. Any HTTP answer below 500 counts (older versions have no
    /healthz and return 404)."""
    url = str(settings()["searxng_url"]).rstrip("/") + "/healthz"
    try:
        return requests.get(url, timeout=timeout).status_code < 500
    except requests.exceptions.RequestException:
        return False


def _wait_until_ready(seconds=90, attempts=3):
    """Wait for SearXNG to answer, then confirm with a real search (a few
    tries: the engines can need a moment after the web server is up)."""
    deadline = time.time() + seconds
    while not reachable():
        if time.time() >= deadline:
            return False
        time.sleep(3)
    for attempt in range(attempts):
        ok, message = check(only="searxng")
        if ok:
            return True
        if attempt + 1 < attempts:
            time.sleep(3)
    print(f"[SEARCH] SearXNG answers but cannot search: {message}")
    return False


def _ask(question):
    """y/N prompt; never blocks without a terminal."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _auto_start_mode(value):
    """'ask' | 'yes' | 'no'. YAML turns an unquoted yes/no into a boolean,
    so accept those too; anything unrecognised means 'ask' (the safe one)."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    value = str(value or "ask").strip().lower()
    return value if value in ("ask", "yes", "no") else "ask"


def offer_docker_setup():
    """Make web search available: SearXNG first when it is a provider
    (offering to start it in Docker), then any other provider. Returns True
    when a search works. Never raises."""
    global _status
    names = providers()
    if "searxng" in names and _offer_searxng():
        return True
    for name in names:
        if name == "searxng":
            continue
        ok, message = check(only=name)
        if ok:
            print(f"[SEARCH] Searching with {name}.")
            _status = True
            return True
        print(f"[SEARCH] {name} is not available: {message}")
    print("[SEARCH] No search provider works; the run continues without web "
          "search." + (" SearXNG elsewhere (a Docker network, another host)? "
                       f"Set web_search.searxng_url or {SEARXNG_URL_ENV}."
                       if "searxng" in names else ""))
    _status = False
    return False


def _offer_searxng():
    """Make SearXNG available, offering to start it in Docker if needed.

    Honors web_search.auto_start: "ask" (default; prompts on a terminal),
    "yes" (no prompt) or "no" (never touch Docker). Returns True when a
    search works afterwards. Never raises.
    """
    global _status
    ws = settings()
    ok, message = check(only="searxng")
    if ok:
        _status = True
        return True
    print(f"[SEARCH] SearXNG is not available: {message}")

    mode = _auto_start_mode(ws.get("auto_start"))
    port = _local_port(ws["searxng_url"])
    if mode == "no" or port is None:
        if port is None:
            print("[SEARCH] The URL isn't local, so it can't be started "
                  "automatically.")
        _status = False
        return False
    if not shutil.which("docker"):
        print("[SEARCH] Docker was not found; install it, or run SearXNG "
              "yourself and set web_search.searxng_url.")
        _status = False
        return False

    try:
        if _docker(["info"], capture_output=True, timeout=30).returncode:
            print("[SEARCH] Docker is installed but its daemon isn't "
                  "reachable (is it running, and are you in the docker "
                  "group?).")
            _status = False
            return False
        state = _container_state()
        image = ws["docker_image"]
        if state == "running":
            print(f"[SEARCH] Container '{CONTAINER_NAME}' is running; "
                  "waiting for it to answer...")
        elif state:
            if mode != "yes" and not _ask(
                    f"Start the existing '{CONTAINER_NAME}' container?"):
                _status = False
                return False
            _docker(["start", CONTAINER_NAME], check=True,
                    capture_output=True, timeout=120)
        else:
            if mode != "yes" and not _ask(
                    f"Download the '{image}' Docker image and start SearXNG "
                    f"on 127.0.0.1:{port}?"):
                print("[SEARCH] Not starting SearXNG.")
                _status = False
                return False
            settings_dir = config_dir()
            write_searxng_settings(settings_dir)
            print(f"[SEARCH] Starting {image} (the first run downloads the "
                  "image; this can take a few minutes)...")
            _docker(["run", "-d", "--name", CONTAINER_NAME,
                     "-p", f"127.0.0.1:{port}:8080",
                     "-v", f"{settings_dir}:/etc/searxng", image],
                    check=True, timeout=1800)
        print("[SEARCH] Waiting for SearXNG to come up...")
        _status = _wait_until_ready()
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[SEARCH] Could not start SearXNG: {e}")
        _status = False
    print("[SEARCH] SearXNG is ready." if _status else
          "[SEARCH] SearXNG did not come up.")
    return _status
