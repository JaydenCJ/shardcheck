"""Read safetensors headers without touching tensor data.

A safetensors file is an 8-byte little-endian unsigned header length, a JSON
object describing every tensor (dtype, shape, byte range relative to the end
of the header), then one flat payload buffer. Everything shardcheck needs —
which tensors a shard holds, how many bytes each claims, how large the file
must therefore be — lives in that header, so a multi-gigabyte shard is
checked by reading a few kilobytes.

Parsing is deliberately two-tiered. Damage that makes the header unusable
(bad length prefix, truncated or oversized header, invalid JSON, non-object
root) raises :class:`~shardcheck.errors.FormatError`. Damage scoped to one
tensor entry (wrong field types, negative offsets, a name defined twice) is
recorded on the returned :class:`Shard` instead, so a single bad entry never
hides the rest of the file from the cross-shard checks.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .errors import FormatError

#: Reserved header key holding the free-form string-to-string metadata map.
METADATA_KEY = "__metadata__"

#: The reference implementation refuses headers above 100 MB; we match it.
HEADER_LIMIT = 100 * 1024 * 1024

#: Bytes per element for every dtype in the safetensors specification.
DTYPE_SIZES: Dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    """One tensor as its shard header describes it."""

    name: str
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        """Bytes this tensor claims in the payload buffer."""
        return self.end - self.start

    @property
    def numel(self) -> int:
        """Number of elements the shape implies (1 for a scalar)."""
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def expected_nbytes(self) -> Optional[int]:
        """Bytes the dtype and shape imply, or None for an unknown dtype."""
        itemsize = DTYPE_SIZES.get(self.dtype)
        if itemsize is None:
            return None
        return self.numel * itemsize


@dataclass
class Shard:
    """A parsed shard: its tensors plus everything needed to validate it."""

    path: str
    file_size: int
    header_size: int
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    metadata: Optional[Dict[str, str]] = None
    #: (tensor name, reason) for entries that were structurally unusable.
    bad_entries: List[Tuple[str, str]] = field(default_factory=list)
    #: Tensor names defined more than once inside this single header.
    duplicate_names: List[str] = field(default_factory=list)

    @property
    def data_size(self) -> int:
        """Payload bytes actually present after the header."""
        return self.file_size - 8 - self.header_size

    @property
    def declared_end(self) -> int:
        """Highest byte offset any tensor claims (0 for an empty shard)."""
        return max((t.end for t in self.tensors.values()), default=0)

    @property
    def tensor_bytes(self) -> int:
        """Total payload bytes claimed by all tensors."""
        return sum(t.nbytes for t in self.tensors.values())


def _pairs_keeping_duplicates(pairs):
    """``object_pairs_hook`` that records duplicate keys instead of merging.

    ``json.loads`` silently keeps the last value for a repeated key — which
    is exactly how a duplicated tensor name slips past every loader. We keep
    the *first* definition and remember the collision.
    """
    out: Dict[str, object] = {}
    dupes: List[str] = []
    for key, value in pairs:
        if key in out:
            dupes.append(key)
        else:
            out[key] = value
    if dupes:
        out["\x00duplicates"] = dupes
    return out


def parse_header(blob: bytes, file_size: int, path: str = "<memory>") -> Shard:
    """Parse the raw start of a safetensors file into a :class:`Shard`.

    ``blob`` must contain at least the 8-byte length prefix and the full
    JSON header; ``file_size`` is the size of the whole file on disk, which
    the truncation checks compare against the declared tensor ranges.
    """
    if len(blob) < 8:
        raise FormatError(
            "file is %d bytes; a safetensors file starts with an 8-byte header length"
            % len(blob)
        )
    (header_size,) = struct.unpack("<Q", blob[:8])
    if header_size == 0:
        raise FormatError("header length prefix is 0")
    if header_size > HEADER_LIMIT:
        raise FormatError(
            "header length prefix claims %d bytes (limit %d); the prefix is corrupt"
            % (header_size, HEADER_LIMIT)
        )
    if 8 + header_size > file_size:
        raise FormatError(
            "header needs %d bytes but the file has only %d; the header itself is truncated"
            % (8 + header_size, file_size)
        )
    raw = blob[8 : 8 + header_size]
    if len(raw) < header_size:
        raise FormatError(
            "expected %d header bytes, could read only %d" % (header_size, len(raw))
        )
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_keeping_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError("header is not valid JSON: %s" % exc) from None
    if not isinstance(doc, dict):
        raise FormatError("header JSON is %s, expected an object" % type(doc).__name__)

    shard = Shard(path=path, file_size=file_size, header_size=header_size)
    shard.duplicate_names = sorted(doc.pop("\x00duplicates", []))

    metadata = doc.pop(METADATA_KEY, None)
    if metadata is not None:
        if isinstance(metadata, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
        ):
            shard.metadata = metadata
        else:
            shard.bad_entries.append(
                (METADATA_KEY, "__metadata__ must be a string-to-string object")
            )

    for name, entry in doc.items():
        reason = _entry_problem(entry)
        if reason is not None:
            shard.bad_entries.append((name, reason))
            continue
        shard.tensors[name] = TensorInfo(
            name=name,
            dtype=entry["dtype"],
            shape=tuple(entry["shape"]),
            start=entry["data_offsets"][0],
            end=entry["data_offsets"][1],
        )
    return shard


def _entry_problem(entry: object) -> Optional[str]:
    """Return why a header entry is structurally unusable, or None if fine."""
    if not isinstance(entry, dict):
        return "entry is %s, expected an object" % type(entry).__name__
    dtype = entry.get("dtype")
    if not isinstance(dtype, str):
        return "'dtype' is missing or not a string"
    shape = entry.get("shape")
    if not isinstance(shape, list) or any(
        not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape
    ):
        return "'shape' must be a list of non-negative integers"
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(o, int) or isinstance(o, bool) for o in offsets)
    ):
        return "'data_offsets' must be a pair of integers"
    start, end = offsets
    if start < 0 or end < start:
        return "'data_offsets' [%d, %d] is negative or reversed" % (start, end)
    return None


def read_shard(path: str) -> Shard:
    """Read and parse just the header of the safetensors file at ``path``.

    Only ``8 + header_size`` bytes are read, so this is fast even on
    multi-gigabyte shards. Raises :class:`FormatError` on container-level
    damage and ``OSError`` if the file cannot be opened at all.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            return parse_header(prefix, file_size, path=path)  # raises with a clear message
        (header_size,) = struct.unpack("<Q", prefix)
        to_read = min(header_size, HEADER_LIMIT + 1)
        blob = prefix + handle.read(to_read)
    return parse_header(blob, file_size, path=path)
