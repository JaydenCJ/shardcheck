"""Render a report as human-readable text or machine-readable JSON.

The text format is built for the terminal at failure time: a one-line
header with the checked target, findings grouped per shard so "which file
do I re-upload?" is answered by the grouping itself, and a final verdict
line (``OK`` / ``FAIL``) that is greppable in CI logs. JSON output is the
same information via :meth:`Report.to_dict`, sorted keys, one document.
"""

from __future__ import annotations

import json
from typing import List

from .findings import INDEX_FILE, Report


def render_text(report: Report, verbose: bool = False) -> str:
    """The full text report, without a trailing newline."""
    lines: List[str] = []
    lines.append("shardcheck: %s" % report.target)
    shard_word = "shard" if len(report.shards) == 1 else "shards"
    context = "mode: %s   %d %s   %d tensors" % (
        report.mode,
        len(report.shards),
        shard_word,
        report.tensor_count,
    )
    lines.append(context)
    lines.append("")

    files = report.files_with_findings()
    # The index pseudo-file renders last: shard problems are actionable
    # per file, index-level bookkeeping reads best as a footer.
    ordered = [f for f in files if f != INDEX_FILE] + (
        [INDEX_FILE] if INDEX_FILE in files else []
    )
    for file in ordered:
        lines.append(file)
        for finding in report.findings_for(file):
            lines.append(
                "  %-7s %-20s %s" % (finding.severity, finding.rule, finding.message)
            )
        lines.append("")

    if verbose and not files:
        for shard in report.shards:
            lines.append("%s: clean" % shard)
        if report.shards:
            lines.append("")

    lines.append(_verdict(report))
    return "\n".join(lines)


def _verdict(report: Report) -> str:
    """The final OK/FAIL line, with counts that match the exit code logic."""
    if not report.findings:
        return "OK: %d %s, %d tensors, no findings" % (
            len(report.shards),
            "shard" if len(report.shards) == 1 else "shards",
            report.tensor_count,
        )
    files = report.files_with_findings()
    shard_files = [f for f in files if f != INDEX_FILE]
    if shard_files:
        scope = "%d of %d shards" % (len(shard_files), len(report.shards))
        if INDEX_FILE in files:
            scope += " + the index"
    else:
        scope = "the index"
    counts = "%d %s, %d %s" % (
        report.errors,
        "error" if report.errors == 1 else "errors",
        report.warnings,
        "warning" if report.warnings == 1 else "warnings",
    )
    verdict = "FAIL" if report.errors else "WARN"
    return "%s: %s in %s" % (verdict, counts, scope)


def render_json(report: Report) -> str:
    """One JSON document with sorted keys; stable for a given tree state."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


def render_listing_text(report_rows: List[dict]) -> str:
    """Tabular ``shardcheck ls`` output from pre-computed row dicts."""
    headers = ["shard", "tensors", "tensor bytes", "file bytes", "status"]
    rows = [
        [
            row["shard"],
            str(row["tensors"]) if row["tensors"] is not None else "-",
            format(row["tensor_bytes"], ",") if row["tensor_bytes"] is not None else "-",
            format(row["file_bytes"], ",") if row["file_bytes"] is not None else "-",
            row["status"],
        ]
        for row in report_rows
    ]
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    for row in rows:
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(out)


__all__ = ["render_text", "render_json", "render_listing_text"]
