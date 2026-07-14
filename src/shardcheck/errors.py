"""Exception hierarchy for shardcheck.

The split matters for exit codes: a :class:`FormatError` on a shard is a
*finding* (the file is broken, which is exactly what shardcheck exists to
report — exit 1), while a :class:`TargetError` means shardcheck was pointed
at something it cannot even begin to check (exit 2, usage error).
"""

from __future__ import annotations


class ShardcheckError(Exception):
    """Base class for every error shardcheck raises on purpose."""


class FormatError(ShardcheckError):
    """A shard file violates the safetensors container format.

    Raised only for damage that makes the header unusable (bad length
    prefix, truncated header, invalid JSON, non-object root). Per-tensor
    problems are collected on the parsed shard instead, so one bad entry
    never hides the rest of the file.
    """


class IndexFileError(ShardcheckError):
    """An ``*.index.json`` file is unreadable or structurally invalid."""


class TargetError(ShardcheckError):
    """The path given to shardcheck is not something it can check."""
