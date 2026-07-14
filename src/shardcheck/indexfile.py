"""Load and validate the structure of ``*.safetensors.index.json`` files.

The index format (as written by ``transformers``' ``save_pretrained``) is a
JSON object with a ``weight_map`` mapping every tensor name to the shard
file that holds it, and an optional ``metadata.total_size`` giving the sum
of all tensor byte sizes. Structural problems raise
:class:`~shardcheck.errors.IndexFileError`; semantic problems (a shard that
does not exist, a tensor that is not really there) are the validator's job.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .errors import IndexFileError


@dataclass
class IndexFile:
    """A parsed shard index: tensor-to-shard map plus declared totals."""

    path: str
    #: tensor name -> shard file name (as written, usually a bare basename).
    weight_map: Dict[str, str] = field(default_factory=dict)
    #: ``metadata.total_size`` if present and a non-negative integer.
    total_size: Optional[int] = None

    @property
    def directory(self) -> str:
        """The directory shard file names are resolved against."""
        return os.path.dirname(os.path.abspath(self.path))

    @property
    def shard_names(self) -> List[str]:
        """Unique shard file names, in first-appearance order."""
        seen: Dict[str, None] = {}
        for shard in self.weight_map.values():
            seen.setdefault(shard, None)
        return list(seen)

    def tensors_for(self, shard_name: str) -> List[str]:
        """Tensor names the index maps to ``shard_name``, in map order."""
        return [t for t, s in self.weight_map.items() if s == shard_name]

    def shard_path(self, shard_name: str) -> str:
        """Absolute path of a shard, resolved next to the index file."""
        return os.path.join(self.directory, shard_name)


def load_index(path: str) -> IndexFile:
    """Parse the index file at ``path``, validating structure strictly.

    Every structural rule an index must satisfy is enforced here with a
    message naming the offending key, because "the index is malformed" is
    useless at 3 a.m. when a deploy is blocked on it.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except OSError as exc:
        raise IndexFileError("cannot read index: %s" % exc) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IndexFileError("index is not valid JSON: %s" % exc) from None

    if not isinstance(doc, dict):
        raise IndexFileError("index root is %s, expected an object" % type(doc).__name__)

    weight_map = doc.get("weight_map")
    if weight_map is None:
        raise IndexFileError("index has no 'weight_map' key")
    if not isinstance(weight_map, dict):
        raise IndexFileError(
            "'weight_map' is %s, expected an object" % type(weight_map).__name__
        )
    if not weight_map:
        raise IndexFileError("'weight_map' is empty; the index maps no tensors")

    clean: Dict[str, str] = {}
    for tensor, shard in weight_map.items():
        if not isinstance(shard, str) or not shard:
            raise IndexFileError(
                "weight_map[%r] is %r, expected a shard file name" % (tensor, shard)
            )
        if os.path.isabs(shard) or ".." in shard.replace("\\", "/").split("/"):
            raise IndexFileError(
                "weight_map[%r] points outside the checkpoint directory: %r"
                % (tensor, shard)
            )
        clean[tensor] = shard

    total_size: Optional[int] = None
    metadata = doc.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise IndexFileError(
                "'metadata' is %s, expected an object" % type(metadata).__name__
            )
        raw_total = metadata.get("total_size")
        if raw_total is not None:
            if not isinstance(raw_total, int) or isinstance(raw_total, bool) or raw_total < 0:
                raise IndexFileError(
                    "metadata.total_size is %r, expected a non-negative integer" % (raw_total,)
                )
            total_size = raw_total

    return IndexFile(path=path, weight_map=clean, total_size=total_size)
