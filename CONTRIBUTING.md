# Contributing to Multi-Agent-Book-Writer

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) (Python 3.11+).

```sh
uv sync            # create .venv and install dependencies (incl. pytest)
uv run pytest      # run the test suite
uv run ruff check .   # lint (bug-catching rules; CI runs this too)
uv run main.py --help
```

The tests need no LLM server, network access or Docker, so they run anywhere. A full pipeline run needs an OpenAI-compatible LLM endpoint; see the [README](README.md).

## Steps to contribute

- Comment on the issue you want to work on. Make sure it's not assigned to someone else.

### Making a PR

> - Make sure you have been assigned the issue to which you are making a PR.
> - If you make a PR before being assigned, it may be labeled `invalid` and closed without merging.

- Fork [the repo](https://github.com/psylsph/Multi-Agent-Book-Writer) and clone your fork.
- Add an `upstream` remote pointing at the main repo:

    ```sh
    git remote add upstream https://github.com/psylsph/Multi-Agent-Book-Writer.git
    ```

- Keep your clone up to date by pulling from upstream (this also avoids merge conflicts later):

    ```sh
    git pull upstream main
    ```

- Create your feature branch from `main`:

    ```sh
    git checkout -b <feature-name>
    ```

- Make your changes, then run the tests (`uv run pytest`) before committing.
- Stage and commit:

    ```sh
    git add <name-of-file>
    git commit -m "Meaningful commit message"
    ```

- Push your branch:

    ```sh
    git push origin <feature-name>
    ```

- Open a pull request against `main` on GitHub.

### Guidelines

- **Tests:** new behaviour needs tests in `tests/` (pytest). Tests must be offline and deterministic: stub the LLM (`generate_prose` / `generate_with_wait` and friends), HTTP, Docker and prompts, and use `tmp_path` for any files. A test must never write to the real `output/` directory.
- **Config options:** if you add or rename an option, update both `config.yaml` and `config.example.yaml` (`tests/test_config_files.py` checks they stay in sync) and the README's configuration section.
- **Dependencies:** add them with `uv add <package>` (or `uv add --dev <package>`) and commit the updated `pyproject.toml` and `uv.lock`. Prefer the standard library where it's enough.
- **Code style:** match the surrounding code. Comment the *why* of non-obvious logic, not the *what*.
- **Docs:** update the README / QUICKSTART when user-visible behaviour or command-line options change.
- **Commits:** keep them focused, with a message that says what changed and why. The git history is the changelog.

## Issue suggestions/Bug reporting

When you are creating an issue, make sure it's not already present. Furthermore, provide a proper description of the changes. If you are suggesting any code improvements, provide thorough details about the improvements.

**Great issue suggestions** tend to have:

- A quick summary of the changes.
- In case of a bug, steps to reproduce:
  - Be specific!
  - Give sample code or a minimal seed prompt if you can.
  - What you expected to happen
  - What actually happens
  - Your model and server (e.g. llama.cpp, LM Studio, Ollama) and the relevant part of `config.yaml` (remove any API key)
  - Notes (possibly including why you think this might be happening, or things you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE.txt).
