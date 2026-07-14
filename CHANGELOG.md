# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added

- Header-only safetensors parser (`shardcheck.safetensors`): reads
  `8 + header_size` bytes per shard, never tensor data, with a two-tier
  error model — container damage raises, per-entry damage is isolated so
  one bad tensor never hides the rest of the shard.
- Byte-level duplicate-key detection inside shard headers
  (`duplicate-name`), a corruption every JSON parser silently swallows.
- Strict `*.safetensors.index.json` loader with named structural errors,
  including rejection of shard paths that escape the checkpoint directory.
- 18 validation rules with stable ids across three layers: index
  cross-references (`missing-shard`, `orphan-shard`, `missing-tensor`,
  `wrong-shard`, `duplicate-tensor`, `unmapped-tensor`,
  `total-size-mismatch`, `index-unreadable`), shard containers
  (`header-invalid`, `bad-entry`, `duplicate-name`, `unknown-dtype`,
  `size-mismatch`, `empty-shard`), and payload layout (`shard-truncated`
  with exact missing byte counts, `trailing-bytes`, `offset-overlap`,
  `offset-gap`).
- Stale-index forensics: orphan shards are still parsed, so a tensor mapped
  to shard A that actually sits in unreferenced shard B is reported as
  `wrong-shard` pointing at B.
- Noise suppression: a missing or unreadable shard is one finding, not one
  per mapped tensor; `total-size-mismatch` is skipped when the sum would be
  partial.
- `shardcheck check` with per-shard grouped text reports, `--json`
  (versioned schema), `--strict`, `-v`, and CI exit codes (0 loadable /
  1 findings / 2 unusable target); accepts an index file, a single shard,
  or a checkpoint directory (with or without an index).
- `shardcheck ls`: per-shard tensor counts, claimed bytes, file bytes, and
  status (referenced / orphan / missing / unreadable / loose).
- `shardcheck explain`: the built-in rule catalog, per-rule documentation.
- Public Python API (`shardcheck.validate`, `check_shard`, `read_shard`,
  `load_index`) mirroring the CLI.
- Deterministic offline demo fixtures (`examples/make_fixture.py`,
  `examples/preflight.sh`) and rule reference (`docs/rules.md`).
- 90 pytest tests and `scripts/smoke.sh` (prints `SMOKE OK`).

### Notes

- The repository ships no CI workflow; verification is local —
  `pip install -e '.[dev]' && pytest && bash scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/shardcheck/releases/tag/v0.1.0
