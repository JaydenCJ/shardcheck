"""shardcheck — preflight validation for sharded safetensors checkpoints.

Validates a ``*.safetensors.index.json`` against the shard files actually on
disk: missing, duplicated, misplaced or unmapped tensors, truncated or
overlapping payloads, stale totals — everything that otherwise fails minutes
into a model load — from headers alone, in seconds, with zero dependencies.

Programmatic use mirrors the CLI:

    from shardcheck import validate

    report = validate("checkpoints/my-model")
    if not report.ok:
        for finding in report.findings:
            print(finding.rule, finding.file, finding.message)
"""

from .errors import FormatError, IndexFileError, ShardcheckError, TargetError
from .findings import Finding, Report
from .indexfile import IndexFile, load_index
from .locate import Target, resolve
from .rules import RULES, Rule
from .safetensors import DTYPE_SIZES, Shard, TensorInfo, read_shard
from .validate import check_shard, validate, validate_index, validate_target

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DTYPE_SIZES",
    "Finding",
    "FormatError",
    "IndexFile",
    "IndexFileError",
    "Report",
    "Rule",
    "RULES",
    "Shard",
    "ShardcheckError",
    "Target",
    "TargetError",
    "TensorInfo",
    "check_shard",
    "load_index",
    "read_shard",
    "resolve",
    "validate",
    "validate_index",
    "validate_target",
]
