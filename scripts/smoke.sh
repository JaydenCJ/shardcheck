#!/usr/bin/env bash
# Smoke test for shardcheck: build a clean and a broken demo checkpoint,
# preflight both, and assert on the findings, the JSON schema, the ls table,
# the rule catalog, and every exit code the CLI documents.
# Self-contained: pure stdlib, no network, idempotent (works from a clean tree).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

# The package has zero runtime dependencies, so running from src/ needs no install.
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/shardcheck-smoke.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

fail() { echo "SMOKE FAIL: $1" >&2; exit 1; }

echo "[smoke] python: $("$PYTHON" --version 2>&1)"

# 0. Build the demo fixture (independent of shardcheck's own code).
"$PYTHON" "$ROOT/examples/make_fixture.py" "$WORKDIR" >/dev/null \
  || fail "make_fixture.py exited non-zero"

# 1. A clean checkpoint passes with exit 0 and an OK verdict.
good_out="$("$PYTHON" -m shardcheck check "$WORKDIR/good")" \
  || fail "check on the clean checkpoint exited non-zero"
echo "$good_out" | sed 's/^/[good] /'
echo "$good_out" | grep -q "^OK: 2 shards, 6 tensors, no findings$" \
  || fail "clean checkpoint did not report OK"

# 2. The broken checkpoint fails with exit 1 and per-shard findings.
set +e
bad_out="$("$PYTHON" -m shardcheck check "$WORKDIR/broken")"
bad_rc=$?
set -e
echo "$bad_out" | sed 's/^/[broken] /'
[ "$bad_rc" -eq 1 ] || fail "broken checkpoint should exit 1, got $bad_rc"
echo "$bad_out" | grep -q "shard-truncated" || fail "truncated shard not detected"
echo "$bad_out" | grep -q "4,096 bytes missing" || fail "missing byte count not reported"
echo "$bad_out" | grep -q "wrong-shard" || fail "stale index mapping not detected"
echo "$bad_out" | grep -q "missing-tensor" || fail "vanished tensor not detected"
echo "$bad_out" | grep -q "orphan-shard" || fail "orphan shard not detected"
echo "$bad_out" | grep -q "^FAIL: 4 errors, 1 warning" || fail "verdict line wrong"

# 3. JSON output honors the documented schema.
json_out="$("$PYTHON" -m shardcheck check "$WORKDIR/broken" --json || true)"
echo "$json_out" | "$PYTHON" -c '
import json, sys
doc = json.load(sys.stdin)
assert doc["schema"] == 1, doc
assert doc["ok"] is False
assert doc["errors"] == 4 and doc["warnings"] == 1
rules = {f["rule"] for f in doc["findings"]}
assert "shard-truncated" in rules and "wrong-shard" in rules
print("json schema ok")
' | sed 's/^/[json] /' || fail "JSON output does not match the schema"

# 4. ls reports per-shard sizes and flags the orphan.
ls_out="$("$PYTHON" -m shardcheck ls "$WORKDIR/broken")"
echo "$ls_out" | sed 's/^/[ls] /'
echo "$ls_out" | grep -E 'model-00003-of-00003\.safetensors.*orphan' >/dev/null \
  || fail "ls did not flag the orphan shard"
echo "$ls_out" | grep -E 'model-00001-of-00002\.safetensors.*referenced' >/dev/null \
  || fail "ls did not list the referenced shard"

# 5. A single shard file can be checked directly.
"$PYTHON" -m shardcheck check "$WORKDIR/good/model-00001-of-00002.safetensors" >/dev/null \
  || fail "single-file check should pass on a valid shard"

# 6. --strict flips a warnings-only run into a failure. The only defect in
#    broken2 is an orphan shard holding a unique tensor name — a warning.
rm -rf "$WORKDIR/broken2"; cp -r "$WORKDIR/good" "$WORKDIR/broken2"
"$PYTHON" - "$WORKDIR/broken2/leftover.safetensors" <<'EOF'
import json, struct, sys
doc = {"spare.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
header = json.dumps(doc).encode()
header += b" " * ((-len(header)) % 8)
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(header)) + header + bytes(8))
EOF
set +e
"$PYTHON" -m shardcheck check "$WORKDIR/broken2" >/dev/null; default_rc=$?
"$PYTHON" -m shardcheck check "$WORKDIR/broken2" --strict >/dev/null; strict_rc=$?
set -e
[ "$default_rc" -eq 0 ] || fail "warnings alone should exit 0 by default (got $default_rc)"
[ "$strict_rc" -eq 1 ] || fail "--strict should exit 1 on warnings (got $strict_rc)"

# 7. Uncheckable targets exit 2.
set +e
"$PYTHON" -m shardcheck check "$WORKDIR/does-not-exist" 2>/dev/null
usage_rc=$?
set -e
[ "$usage_rc" -eq 2 ] || fail "missing target should exit 2, got $usage_rc"

# 8. explain documents the rules. (Capture first, then grep: grep -q on a
#    live pipe closes it early, which under pipefail would fail the step.)
explain_out="$("$PYTHON" -m shardcheck explain)" || fail "explain exited non-zero"
echo "$explain_out" | grep -q "shard-truncated" || fail "explain missing rule list"
detail_out="$("$PYTHON" -m shardcheck explain shard-truncated)" \
  || fail "explain shard-truncated exited non-zero"
echo "$detail_out" | grep -q "half-uploaded" \
  || fail "explain shard-truncated missing detail"

# 9. --version agrees with the package version.
version_out="$("$PYTHON" -m shardcheck --version)"
pkg_version="$("$PYTHON" -c 'import shardcheck; print(shardcheck.__version__)')"
[ "$version_out" = "shardcheck $pkg_version" ] \
  || fail "--version mismatch: '$version_out' vs package '$pkg_version'"

echo "SMOKE OK"
