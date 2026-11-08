# Multi-Agent Book Writer: the pipeline CLI plus the optional web GUI.
#
#   docker compose up --build          # web GUI on http://127.0.0.1:8000
#   docker compose run --rm app uv run python main.py --demo   # the CLI
#
# config.yaml is mounted at run time (never baked in) because it holds the
# LLM endpoint and key. See docker-compose.yml for that and the mounts.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    BOOKWRITER_CONFIG=/app/config.yaml

WORKDIR /app

# dependencies first: editing the code then only rebuilds the last layers
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra web

COPY main.py make_epub.py ./
COPY agents/ agents/
COPY shared/ shared/
COPY seeds/ seeds/
COPY web/ web/

RUN mkdir -p /app/output

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
