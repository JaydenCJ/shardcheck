"""Shared fixture factories: build real sharded checkpoints, deterministically.

Everything is written with the stdlib (struct + json) rather than through any
model library, so the tests exercise shardcheck against independently
constructed input. Payload bytes come from a fixed arithmetic pattern — no
randomness anywhere, every run byte-identical, fully offline.
"""

from __future__ import annotations

import json
import struct
from typing import Dict, Optional, Sequence, Tuple

import pytest

#: (dtype, shape, payload bytes) — the value type used by write_st.
Spec = Tuple[str, Sequence[int], bytes]


def payload(n: int, seed: int = 0) -> bytes:
    """n deterministic bytes; different seeds give different content."""
    return bytes((i * 7 + seed * 13 + 3) % 256 for i in range(n))


def write_st(
    path,
    tensors: Dict[str, Spec],
    metadata: Optional[Dict[str, str]] = None,
    header_override: Optional[bytes] = None,
    truncate_tail: int = 0,
    extra_payload: bytes = b"",
) -> None:
    """Write a safetensors file from named (dtype, shape, bytes) specs.

    Offsets are packed contiguously in dict order. ``header_override`` swaps
    in arbitrary header bytes to build corrupt files; ``truncate_tail`` chops
    bytes off the end (the half-uploaded shard); ``extra_payload`` appends
    slack bytes beyond the last tensor.
    """
    doc: Dict[str, object] = {}
    if metadata is not None:
        doc["__metadata__"] = metadata
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
    header = header_override
    if header is None:
        header = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        header += b" " * ((-len(header)) % 8)
    raw = struct.pack("<Q", len(header)) + header + blob + extra_payload
    if truncate_tail:
        raw = raw[:-truncate_tail]
    with open(path, "wb") as f:
        f.write(raw)


def write_raw_entries(path, entries: Dict[str, object], blob: bytes = b"") -> None:
    """Write a file whose header entries are taken verbatim (may be invalid)."""
    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(blob)


def model_tensors() -> Dict[str, Spec]:
    """A miniature transformer-shaped checkpoint split across two shards."""
    return {
        "model.embed_tokens.weight": ("F32", [4, 8], payload(128, seed=1)),
        "model.layers.0.attn.weight": ("F16", [4, 4], payload(32, seed=2)),
        "model.layers.0.mlp.weight": ("BF16", [8, 4], payload(64, seed=3)),
        "model.layers.1.attn.weight": ("F16", [4, 4], payload(32, seed=4)),
        "model.norm.weight": ("F32", [4], payload(16, seed=5)),
        "lm_head.weight": ("F32", [8, 4], payload(128, seed=6)),
    }


SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"


def split_tensors():
    """The model tensors split as the fixtures shard them: 3 + 3."""
    tensors = model_tensors()
    names = list(tensors)
    return (
        {name: tensors[name] for name in names[:3]},
        {name: tensors[name] for name in names[3:]},
    )


def write_index(directory, weight_map: Dict[str, str], total_size: Optional[int]) -> str:
    """Write an *.index.json next to the shards; returns its path."""
    doc: Dict[str, object] = {"weight_map": weight_map}
    if total_size is not None:
        doc["metadata"] = {"total_size": total_size}
    path = directory / "model.safetensors.index.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return str(path)


def make_checkpoint(directory, total_size: Optional[int] = None) -> str:
    """A valid two-shard checkpoint with index; returns the index path.

    ``total_size`` defaults to the true byte total so the default fixture is
    completely clean; pass a wrong value to provoke total-size-mismatch.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shard_a, shard_b = split_tensors()
    write_st(directory / SHARD_1, shard_a, metadata={"format": "pt"})
    write_st(directory / SHARD_2, shard_b)
    if total_size is None:
        total_size = sum(len(data) for _, _, data in model_tensors().values())
    weight_map = {
        **{name: SHARD_1 for name in shard_a},
        **{name: SHARD_2 for name in shard_b},
    }
    return write_index(directory, weight_map, total_size)


@pytest.fixture
def checkpoint(tmp_path):
    """Index path of a fully valid two-shard checkpoint."""
    return make_checkpoint(tmp_path / "ckpt")


@pytest.fixture
def single_shard(tmp_path):
    """One valid stand-alone shard (no index)."""
    path = tmp_path / "model.safetensors"
    write_st(path, model_tensors(), metadata={"format": "pt"})
    return path
