# shardcheck examples

Both examples are standard-library only and fully offline.

| File | What it shows |
|---|---|
| [`make_fixture.py`](make_fixture.py) | Builds two small, deterministic demo checkpoints — a valid two-shard one and the same checkpoint after a botched re-upload (truncated shard, stale index, orphan shard) — without importing shardcheck, so the demos start from independent input. |
| [`preflight.sh`](preflight.sh) | The full workflow: build the fixture → `check` the clean copy (exit 0) → `check` the broken copy (exit 1, per-shard findings) → `ls` the shards → `explain shard-truncated`. Prints `EXAMPLE OK` at the end. |

Run from the repository root:

```bash
bash examples/preflight.sh
```

The generated files land in a temporary directory that is removed on exit;
nothing in the repository is modified.
