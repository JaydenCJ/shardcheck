"""Work out what to check from the path the user gave.

``shardcheck check`` accepts an index file, a single shard, or a checkpoint
directory. Directories are resolved here: exactly one ``*.index.json`` means
index-driven validation; none means every ``*.safetensors`` file is checked
as a set; more than one is ambiguous and the user must pick. All listing is
sorted so results are deterministic across filesystems.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .errors import TargetError

MODE_INDEX = "index"
MODE_DIRECTORY = "directory"
MODE_FILE = "file"


@dataclass(frozen=True)
class Target:
    """A resolved check target: what mode to run and against which files."""

    mode: str
    #: The path the user gave, as given (used for display).
    given: str
    #: Absolute index path in index mode, else None.
    index_path: Optional[str] = None
    #: Absolute shard paths in directory/file mode, else empty.
    shard_paths: tuple = ()


def list_indexes(directory: str) -> List[str]:
    """Sorted ``*.index.json`` basenames in ``directory``."""
    return sorted(
        name
        for name in os.listdir(directory)
        if name.endswith(".index.json") and os.path.isfile(os.path.join(directory, name))
    )


def list_shards(directory: str) -> List[str]:
    """Sorted ``*.safetensors`` basenames in ``directory``."""
    return sorted(
        name
        for name in os.listdir(directory)
        if name.endswith(".safetensors") and os.path.isfile(os.path.join(directory, name))
    )


def resolve(path: str) -> Target:
    """Turn a user-supplied path into a concrete :class:`Target`.

    Raises :class:`TargetError` (exit code 2 territory) when the path does
    not exist, is not checkable, or is ambiguous.
    """
    if not os.path.exists(path):
        raise TargetError("no such file or directory: %s" % path)

    if os.path.isfile(path):
        if path.endswith(".index.json"):
            return Target(mode=MODE_INDEX, given=path, index_path=os.path.abspath(path))
        if path.endswith(".safetensors"):
            return Target(
                mode=MODE_FILE, given=path, shard_paths=(os.path.abspath(path),)
            )
        raise TargetError(
            "%s is neither an *.index.json nor a *.safetensors file" % path
        )

    indexes = list_indexes(path)
    if len(indexes) > 1:
        raise TargetError(
            "%s contains %d index files (%s); pass one explicitly"
            % (path, len(indexes), ", ".join(indexes))
        )
    if len(indexes) == 1:
        return Target(
            mode=MODE_INDEX,
            given=path,
            index_path=os.path.abspath(os.path.join(path, indexes[0])),
        )

    shards = list_shards(path)
    if not shards:
        raise TargetError(
            "%s contains no *.index.json and no *.safetensors files" % path
        )
    return Target(
        mode=MODE_DIRECTORY,
        given=path,
        shard_paths=tuple(os.path.abspath(os.path.join(path, name)) for name in shards),
    )
