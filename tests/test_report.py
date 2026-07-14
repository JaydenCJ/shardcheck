"""Rendering: the text report people read and the JSON machines read.

Text assertions target the load-bearing pieces (grouping, verdict line,
alignment tokens) rather than full-screen golden output, so cosmetic tweaks
do not shatter the suite; JSON assertions are exact because it is a schema.
"""

from __future__ import annotations

import json

from shardcheck.findings import INDEX_FILE, Finding, Report
from shardcheck.report import render_json, render_listing_text, render_text

import pytest


def make_report(**kwargs) -> Report:
    defaults = dict(
        target="ckpt",
        mode="index",
        index_path="/x/model.safetensors.index.json",
        shards=["s1.safetensors", "s2.safetensors"],
        tensor_count=6,
    )
    defaults.update(kwargs)
    return Report(**defaults)


def test_clean_report_renders_ok_verdict():
    text = render_text(make_report())
    assert text.splitlines()[0] == "shardcheck: ckpt"
    assert text.splitlines()[-1] == "OK: 2 shards, 6 tensors, no findings"
    # Verbose mode names every clean shard so "it checked nothing" is ruled out.
    verbose = render_text(make_report(), verbose=True)
    assert "s1.safetensors: clean" in verbose
    assert "s2.safetensors: clean" in verbose


def test_findings_are_grouped_under_their_file():
    report = make_report()
    report.add("shard-truncated", "s2.safetensors", "file is short")
    report.add("missing-tensor", "s2.safetensors", "'w' is gone", tensor="w")
    lines = render_text(report).splitlines()
    header_at = lines.index("s2.safetensors")
    assert "shard-truncated" in lines[header_at + 1]
    assert "missing-tensor" in lines[header_at + 2]


def test_index_findings_render_last_as_a_footer():
    report = make_report()
    report.add("total-size-mismatch", INDEX_FILE, "totals disagree")
    report.add("shard-truncated", "s1.safetensors", "short")
    lines = render_text(report).splitlines()
    assert lines.index("s1.safetensors") < lines.index(INDEX_FILE)


def test_fail_verdict_counts_errors_and_scoped_shards():
    report = make_report()
    report.add("shard-truncated", "s2.safetensors", "short")
    report.add("orphan-shard", "s2.safetensors", "orphan")
    assert render_text(report).splitlines()[-1] == (
        "FAIL: 1 error, 1 warning in 1 of 2 shards"
    )


def test_warnings_only_verdict_says_warn_not_fail():
    report = make_report()
    report.add("orphan-shard", "s1.safetensors", "orphan")
    verdict = render_text(report).splitlines()[-1]
    assert verdict.startswith("WARN:")
    assert report.ok  # warnings never flip ok


def test_index_only_findings_scope_reads_the_index():
    report = make_report()
    report.add("index-unreadable", INDEX_FILE, "bad json")
    assert render_text(report).splitlines()[-1] == "FAIL: 1 error, 0 warnings in the index"


def test_json_report_is_a_stable_schema():
    report = make_report()
    report.add("wrong-shard", "s1.safetensors", "'w' lives elsewhere", tensor="w")
    doc = json.loads(render_json(report))
    assert doc["schema"] == 1
    assert doc["ok"] is False
    assert doc["errors"] == 1 and doc["warnings"] == 0
    assert doc["findings"] == [
        {
            "rule": "wrong-shard",
            "severity": "error",
            "file": "s1.safetensors",
            "message": "'w' lives elsewhere",
            "tensor": "w",
        }
    ]


def test_unknown_rule_id_is_rejected_at_construction():
    # A typo in a rule id must explode in tests, never ship silently.
    with pytest.raises(ValueError, match="unknown rule id"):
        Finding(rule="not-a-rule", file="x", message="m")


def test_listing_renders_aligned_columns_with_placeholders():
    rows = [
        {
            "shard": "model-00001.safetensors",
            "tensors": 3,
            "tensor_bytes": 224,
            "file_bytes": 512,
            "status": "referenced",
        },
        {
            "shard": "gone.safetensors",
            "tensors": None,
            "tensor_bytes": None,
            "file_bytes": None,
            "status": "missing",
        },
    ]
    lines = render_listing_text(rows).splitlines()
    assert lines[0].split() == ["shard", "tensors", "tensor", "bytes", "file", "bytes", "status"]
    assert "224" in lines[1] and "referenced" in lines[1]
    assert lines[2].split() == ["gone.safetensors", "-", "-", "-", "missing"]
