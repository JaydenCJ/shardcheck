#!/usr/bin/env bash
# The full shardcheck workflow on a demo checkpoint pair: build the fixture,
# preflight the clean copy (exit 0), preflight the broken copy (exit 1, with
# per-shard findings), then inspect it with `ls` and read a rule with
# `explain`. Standard-library only, fully offline; prints EXAMPLE OK.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/shardcheck-example.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

"$PYTHON" "$ROOT/examples/make_fixture.py" "$WORKDIR"

echo
echo "== 1. preflight the clean checkpoint =="
"$PYTHON" -m shardcheck check "$WORKDIR/good"

echo
echo "== 2. preflight the broken checkpoint (exit code 1 expected) =="
if "$PYTHON" -m shardcheck check "$WORKDIR/broken"; then
  echo "expected findings but the check passed" >&2
  exit 1
fi

echo
echo "== 3. list the shards =="
"$PYTHON" -m shardcheck ls "$WORKDIR/broken"

echo
echo "== 4. read up on one rule =="
"$PYTHON" -m shardcheck explain shard-truncated

echo
echo "EXAMPLE OK"
