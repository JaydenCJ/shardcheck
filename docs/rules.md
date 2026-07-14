# shardcheck rules

Every finding shardcheck can produce, with its stable id and severity.
Ids are part of the public contract: they appear verbatim in text and JSON
output, `shardcheck explain <id>` documents them, and their meaning never
changes — new checks get new ids.

Severity is a property of the rule: **error** means the checkpoint will fail
or corrupt a load; **warning** means it will load, but something is off.
Errors drive the exit code (`1`); warnings do not, unless `--strict` is set.

## Index-level rules

Fired when a `*.safetensors.index.json` disagrees with the shards next to it.

| Rule | Severity | Fires when |
|---|---|---|
| `index-unreadable` | error | the index file is missing, not valid JSON, or structurally invalid (no `weight_map`, non-string shard names, paths escaping the directory) |
| `missing-shard` | error | `weight_map` references a shard file that is not on disk |
| `orphan-shard` | warning | a `*.safetensors` file next to the index is never referenced |
| `missing-tensor` | error | a mapped tensor is absent from every shard on disk |
| `wrong-shard` | error | a mapped tensor exists — but in a different shard than the index says |
| `duplicate-tensor` | error | the same tensor name is defined in more than one shard |
| `unmapped-tensor` | warning | a shard holds a tensor `weight_map` never mentions |
| `total-size-mismatch` | warning | `metadata.total_size` does not equal the bytes the shards actually claim |

Notes:

- A missing or unreadable shard produces exactly one finding; the tensors
  mapped to it are *not* additionally reported as missing — one actionable
  line beats forty noisy ones.
- Orphan shards are still parsed. A tensor the index maps to shard A that
  actually sits in an orphan B is reported as `wrong-shard` pointing at B —
  the exact signature of a stale index over renamed files.
- `total-size-mismatch` is only evaluated when every referenced shard was
  present and readable; a partial sum would always mismatch and add noise.

## Shard container rules

Fired per shard, from the header and the file size alone — tensor data is
never read, which is why checking a 100 GB checkpoint takes well under a second.

| Rule | Severity | Fires when |
|---|---|---|
| `header-invalid` | error | the 8-byte length prefix is corrupt, the header is truncated or over the 100 MB limit, or the header JSON is unusable |
| `bad-entry` | error | one tensor entry violates the schema (dtype/shape/data_offsets types) |
| `duplicate-name` | error | the same tensor name appears twice inside one header (JSON parsers hide this; shardcheck detects it at the byte level) |
| `unknown-dtype` | error | a dtype outside the fifteen the safetensors specification defines |
| `size-mismatch` | error | `data_offsets` spans a different byte count than `product(shape) × itemsize(dtype)` |
| `shard-truncated` | error | the file is shorter than `8 + header + declared tensor data` — the half-uploaded shard; the finding reports the exact missing byte count |
| `trailing-bytes` | warning | bytes exist past the last declared tensor range |
| `offset-overlap` | error | two tensors claim intersecting byte ranges |
| `offset-gap` | warning | payload bytes no tensor claims (valid files pack contiguously from offset 0) |
| `empty-shard` | warning | the header parses but defines zero tensors |

Notes:

- Zero-length tensors (a `0` in the shape) occupy no bytes and can neither
  overlap anything nor plug a gap; they are excluded from layout checks.
- On a `duplicate-name` collision the *first* definition wins for all
  further checks, so downstream findings stay deterministic.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | checkpoint is loadable — no errors (warnings allowed unless `--strict`) |
| `1` | findings block a load (or warnings exist and `--strict` was given) |
| `2` | the target could not be checked at all (bad path, ambiguous directory, unknown rule id) |

## JSON output schema

`shardcheck check --json` emits one document (`schema: 1`):

```json
{
  "schema": 1,
  "target": "checkpoints/my-model",
  "mode": "index",
  "index": "/abs/path/model.safetensors.index.json",
  "shards": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
  "tensors": 6,
  "errors": 1,
  "warnings": 0,
  "ok": false,
  "findings": [
    {
      "rule": "shard-truncated",
      "severity": "error",
      "file": "model-00002-of-00002.safetensors",
      "message": "file is 6,520 bytes but header + tensor data need 10,616 (4,096 bytes missing from the tail — half-uploaded?)"
    }
  ]
}
```

`findings[].tensor` is present when the finding is pinned to a specific
tensor. Findings against the index itself use the pseudo-file `(index)`.
Breaking changes to this document bump `schema`.
