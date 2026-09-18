# Contributing

Thanks for taking the time. This is a demonstration stack for real-world-evidence
tooling over synthetic patient data, so the bar for a change is that it stays
runnable with local synthetic data and no licensed patient dataset. The default
workflow needs an external LLM API key, but no hosted warehouse infrastructure.

## Getting the stack running

Follow the [README quickstart](README.md#quickstart) for a fresh clone. Use the
[development guide](docs/development.md) for container iteration or running the
backend on your host. The default Compose file does not publish dependency
ports; the host workflow needs its explicit development overlay and settings.

Python is pinned to 3.12 (`>=3.12,<3.13`) and dependencies are managed with
[`uv`](https://docs.astral.sh/uv/).

## Before opening a pull request

From the repository root:

```bash
uv sync --frozen --extra dev
pre-commit install        # once; install pre-commit separately if needed
pre-commit run --all-files
PYTHONPATH=./src uv run --no-sync pytest tests/ -q
(cd services/medical-coder && uv sync --frozen --extra dev && uv run --no-sync pytest tests/ -q)
```

`PYTHONPATH=./src` is not optional — the packages live under `src/`, and without
it the imports fail in a way that looks like missing dependencies.

Install [pre-commit](https://pre-commit.com/#install) if it is not available in
your shell. The hooks use Ruff and the repository's per-project settings. If a
hook rewrites a file, stage the result and rerun the checks. CI gates changed
Python files, uses a type-error ratchet, and runs both test suites and Python
image builds. All-file checks can also report pre-existing lint/format debt.

## Working with the data

The synthetic datasets under `data/synthetic/` are generated, not hand-made, and
the same seed reproduces the same databases — see
[`data/synthetic/README.md`](data/synthetic/README.md) for where they come from.
If a change affects the data, regenerate rather than editing a database by hand,
and move the ground truth with it.

Scale is one choice, not three: `ground_truth_1k.json` describes the 1k
population, so pairing it with the 10k databases makes every assertion fail for
no real reason.

## What makes a change easy to accept

**Explain why, not what.** The diff already says what changed. A comment or
commit message earns its place by recording the reason — the constraint you hit,
the alternative you rejected, the failure that made this necessary.

**Keep the data stack self-contained.** Do not add a required licensed vocabulary,
private data endpoint, or hosted warehouse to the default setup. Synthetic data
and local services should remain sufficient apart from the documented LLM APIs.

**One concern per pull request.** A refactor and a behaviour change in the same
diff are hard to review and harder to revert.

**Say what you verified.** "Tests pass" is weaker than naming the case you
checked and what it produced. If something is untested, saying so is better than
leaving it to be discovered.

## Reporting problems

Open an issue with what you ran, what happened, and what you expected. For
anything data-related, include the scale (`1k` or `10k`) — most surprising counts
trace back to a ground truth and a database that disagree about which population
they describe.

Please do not report security issues in a public issue; contact the maintainers
listed in [`.github/CODEOWNERS`](.github/CODEOWNERS) directly.

## Licence

By contributing you agree that your contributions are licensed under the
BSD 3-Clause Licence in [`LICENSE`](LICENSE).
