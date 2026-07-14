# Contributing to shardcheck

Thanks for your interest in contributing. Issues, discussions, and pull
requests are all welcome.

## Development setup

```bash
git clone https://github.com/JaydenCJ/shardcheck
cd shardcheck
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks

```bash
pytest                 # the unit and CLI test suite (tests/)
bash scripts/smoke.sh  # end-to-end smoke: build fixtures, check, ls, explain
```

Both must pass before a pull request is reviewed; `scripts/smoke.sh` must
print `SMOKE OK`. The suite runs fully offline, needs no model files, and
finishes in under a second.

## Before you open a pull request

1. Format and lint if you have the tools (`ruff format` / `ruff check`);
   match the surrounding style either way.
2. `pytest` — must pass.
3. `bash scripts/smoke.sh` — must print `SMOKE OK`.
4. Add tests for behavior changes; keep logic in pure, unit-testable modules.

## Ground rules

- **No new runtime dependencies.** The package is standard-library only;
  that is the headline feature. Test-only dependencies belong in the `dev`
  extra.
- **Rule ids are a public contract.** Never change what an existing id
  means; add a new rule instead, and document it in `docs/rules.md` and
  `shardcheck explain` in the same pull request.
- **JSON output is a schema.** Anything that changes the meaning of an
  existing field must bump the `schema` number and update `docs/rules.md`.
- **No network calls, ever.** shardcheck reads local files and prints;
  nothing else.
- Code comments and doc comments are written in English.
- **Keep the three READMEs aligned.** `README.md`, `README.zh.md`, and
  `README.ja.md` share the same structure; update all three when you change
  one (English is the authoritative version).

## Reporting bugs

Please include `shardcheck --version`, the full text (or `--json`) output,
and — if at all possible — a way to reproduce the checkpoint shape
(`examples/make_fixture.py` shows how to build synthetic shards with the
standard library; a header-only repro is usually enough, tensor data never
matters to shardcheck).

## Security

Please do not report security issues in public GitHub issues. Use GitHub's
private vulnerability reporting on this repository instead.
