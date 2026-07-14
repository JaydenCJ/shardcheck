#!/usr/bin/env python3
"""Build two small demo checkpoints: one clean, one realistically broken.

Standard-library only, deterministic, and deliberately *not* using
shardcheck's own code, so the demo (and the smoke test built on it) starts
from independently constructed input.

Usage: python3 examples/make_fixture.py OUTPUT_DIR

Creates OUTPUT_DIR/good/ (a valid two-shard checkpoint) and
OUTPUT_DIR/broken/ (the same checkpoint after a botched re-upload: one
shard truncated mid-transfer, a stale index that maps one tensor to the
wrong shard and one tensor that no longer exists, plus a leftover shard
from an earlier save).
"""

from __future__ import annotations

import json
import os
import struct
import sys


def payload(n: int, seed: int = 0) -> bytes:
    """n deterministic bytes; different seeds give different content."""
    return bytes((i * 7 + seed * 13 + 3) % 256 for i in range(n))


def write_shard(path: str, tensors) -> None:
    """Write a valid safetensors file: contiguous offsets, padded header."""
    doc = {}
    offset = 0
    blob = b""
    for name, (dtype, shape, data) in tensors.items():
        doc[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blob += data
    header = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.write(blob)


def model_tensors():
    """A miniature transformer-shaped checkpoint, split 3 + 3."""
    return {
        "model.embed_tokens.weight": ("F32", [64, 32], payload(8192, seed=1)),
        "model.layers.0.attn.weight": ("F16", [32, 32], payload(2048, seed=2)),
        "model.layers.0.mlp.weight": ("BF16", [64, 32], payload(4096, seed=3)),
        "model.layers.1.attn.weight": ("F16", [32, 32], payload(2048, seed=4)),
        "model.norm.weight": ("F32", [32], payload(128, seed=5)),
        "lm_head.weight": ("F32", [64, 32], payload(8192, seed=6)),
    }


SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"


def build_good(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    tensors = model_tensors()
    names = list(tensors)
    write_shard(os.path.join(directory, SHARD_1), {n: tensors[n] for n in names[:3]})
    write_shard(os.path.join(directory, SHARD_2), {n: tensors[n] for n in names[3:]})
    index = {
        "metadata": {"total_size": sum(len(d) for _, _, d in tensors.values())},
        "weight_map": {
            n: (SHARD_1 if n in names[:3] else SHARD_2) for n in names
        },
    }
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")


def build_broken(directory: str) -> None:
    """The good checkpoint after a botched re-upload and a stale index."""
    build_good(directory)

    # 1. Shard 2's upload died mid-transfer: chop 4 KiB off its tail.
    shard_2 = os.path.join(directory, SHARD_2)
    with open(shard_2, "rb") as handle:
        raw = handle.read()
    with open(shard_2, "wb") as handle:
        handle.write(raw[: len(raw) - 4096])

    # 2. The index is stale: it predates a re-shard, so it maps
    #    'model.norm.weight' to shard 1 (it lives in shard 2) and still
    #    lists a tensor that was renamed away.
    index_path = os.path.join(directory, "model.safetensors.index.json")
    with open(index_path) as handle:
        index = json.load(handle)
    index["weight_map"]["model.norm.weight"] = SHARD_1
    index["weight_map"]["model.layers.2.attn.weight"] = SHARD_2
    with open(index_path, "w") as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")

    # 3. A shard from an earlier 3-way save was never cleaned up.
    write_shard(
        os.path.join(directory, "model-00003-of-00003.safetensors"),
        {"lm_head.weight": model_tensors()["lm_head.weight"]},
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    out = sys.argv[1]
    build_good(os.path.join(out, "good"))
    build_broken(os.path.join(out, "broken"))
    print("fixture ready: %s/good and %s/broken" % (out, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
