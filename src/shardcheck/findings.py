"""Findings and the report that groups them per shard.

A :class:`Finding` is one concrete violation of one rule, pinned to the file
(and, where it makes sense, the tensor) it was found in. A :class:`Report`
is everything one ``shardcheck check`` run produced: the findings, plus the
context needed to render them usefully — which shards were checked, how many
tensors were seen, and whether an index drove the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .rules import ERROR, RULES, WARNING

#: The sentinel "file" findings against the index itself are attached to.
INDEX_FILE = "(index)"

#: Report schema version emitted in JSON output; bump on breaking changes.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Finding:
    """One violation: rule id, the file it is in, and a concrete message."""

    rule: str
    file: str
    message: str
    tensor: Optional[str] = None

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError("unknown rule id: %r" % self.rule)

    @property
    def severity(self) -> str:
        return RULES[self.rule].severity

    def to_dict(self) -> Dict[str, object]:
        doc: Dict[str, object] = {
            "rule": self.rule,
            "severity": self.severity,
            "file": self.file,
            "message": self.message,
        }
        if self.tensor is not None:
            doc["tensor"] = self.tensor
        return doc


@dataclass
class Report:
    """The full result of checking one checkpoint."""

    target: str
    mode: str  # "index" | "directory" | "file"
    index_path: Optional[str] = None
    #: Shard file names in check order (referenced and orphaned alike).
    shards: List[str] = field(default_factory=list)
    tensor_count: int = 0
    findings: List[Finding] = field(default_factory=list)

    def add(self, rule: str, file: str, message: str, tensor: Optional[str] = None) -> None:
        self.findings.append(Finding(rule=rule, file=file, message=message, tensor=tensor))

    @property
    def errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == ERROR)

    @property
    def warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == WARNING)

    @property
    def ok(self) -> bool:
        """True when no *errors* were found (warnings do not fail a check)."""
        return self.errors == 0

    def files_with_findings(self) -> List[str]:
        """Files that have at least one finding, in report order."""
        seen: Dict[str, None] = {}
        for finding in self.findings:
            seen.setdefault(finding.file, None)
        return list(seen)

    def findings_for(self, file: str) -> List[Finding]:
        return [f for f in self.findings if f.file == file]

    def to_dict(self) -> Dict[str, object]:
        """JSON-ready form, deterministic for a given checkpoint state."""
        return {
            "schema": SCHEMA_VERSION,
            "target": self.target,
            "mode": self.mode,
            "index": self.index_path,
            "shards": list(self.shards),
            "tensors": self.tensor_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
        }
