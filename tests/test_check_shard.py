"""Container-level rules: everything check_shard can find in one file.

Each test corrupts a valid shard in exactly one way and asserts that
exactly the expected rule fires — no more, no less. That "no more" half is
what keeps the report readable on real damage.
"""

from __future__ import annotations

import pytest

from shardcheck.safetensors import read_shard
from shardcheck.validate import check_shard

from conftest import model_tensors, payload, write_raw_entries, write_st


def findings_for(path):
    return check_shard(read_shard(str(path)))


def rules(findings):
    return sorted(f.rule for f in findings)


def test_valid_shard_yields_zero_findings(single_shard):
    assert findings_for(single_shard) == []


def test_truncated_shard_reports_missing_byte_count(tmp_path):
    # The signature failure mode: header intact, payload tail gone. The
    # finding carries the shard basename and the exact missing byte count.
    path = tmp_path / "cut.safetensors"
    write_st(path, model_tensors(), truncate_tail=100)
    found = findings_for(path)
    assert rules(found) == ["shard-truncated"]
    assert "100 bytes missing" in found[0].message
    assert found[0].file == "cut.safetensors"
    # Even a single missing byte must be caught — no tolerance window.
    one = tmp_path / "cut1.safetensors"
    write_st(one, model_tensors(), truncate_tail=1)
    assert rules(findings_for(one)) == ["shard-truncated"]


def test_trailing_bytes_after_payload_warn(tmp_path):
    path = tmp_path / "slack.safetensors"
    write_st(path, model_tensors(), extra_payload=payload(64))
    found = findings_for(path)
    assert rules(found) == ["trailing-bytes"]
    assert "64" in found[0].message
    assert found[0].severity == "warning"


def test_unknown_dtype_is_an_error_and_skips_size_check(tmp_path):
    path = tmp_path / "dtype.safetensors"
    write_st(path, {"w": ("F13", [4], payload(16))})
    found = findings_for(path)
    assert rules(found) == ["unknown-dtype"]
    assert "'F13'" in found[0].message


def test_size_mismatch_names_dtype_shape_and_span(tmp_path):
    # F32[4] needs 16 bytes; the range spans only 12.
    write_raw_entries(
        tmp_path / "size.safetensors",
        {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 12]}},
        blob=payload(12),
    )
    found = findings_for(tmp_path / "size.safetensors")
    assert rules(found) == ["size-mismatch"]
    assert "16 bytes" in found[0].message and "span 12" in found[0].message


def test_overlapping_offsets_are_an_error(tmp_path):
    write_raw_entries(
        tmp_path / "overlap.safetensors",
        {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        },
        blob=payload(24),
    )
    found = findings_for(tmp_path / "overlap.safetensors")
    assert rules(found) == ["offset-overlap"]
    assert "'b'" in found[0].message and "'a'" in found[0].message


def test_gap_between_tensors_warns_with_range(tmp_path):
    write_raw_entries(
        tmp_path / "gap.safetensors",
        {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [24, 40]},
        },
        blob=payload(40),
    )
    found = findings_for(tmp_path / "gap.safetensors")
    assert rules(found) == ["offset-gap"]
    assert "[16, 24)" in found[0].message


def test_payload_not_starting_at_zero_is_a_gap(tmp_path):
    write_raw_entries(
        tmp_path / "lead.safetensors",
        {"a": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]}},
        blob=payload(24),
    )
    assert rules(findings_for(tmp_path / "lead.safetensors")) == ["offset-gap"]


def test_zero_length_tensor_inside_a_range_is_not_an_overlap(tmp_path):
    # An empty tensor occupies no bytes; sitting "inside" another tensor's
    # range must not be flagged as sharing bytes with it.
    write_raw_entries(
        tmp_path / "empty.safetensors",
        {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "e": {"dtype": "F32", "shape": [0], "data_offsets": [8, 8]},
        },
        blob=payload(16),
    )
    assert findings_for(tmp_path / "empty.safetensors") == []


def test_duplicate_name_within_one_header_is_an_error(tmp_path):
    import struct

    body = (
        b'{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    )
    path = tmp_path / "dup.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body + payload(8))
    found = findings_for(path)
    assert rules(found) == ["duplicate-name"]


def test_bad_entry_reports_but_other_tensors_still_checked(tmp_path):
    # One malformed entry plus one truncation: both must surface.
    write_raw_entries(
        tmp_path / "mixed.safetensors",
        {
            "ok": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "broken": {"dtype": "F32", "shape": "x", "data_offsets": [16, 32]},
        },
        blob=payload(8),  # 8 < 16 -> the good tensor is truncated too
    )
    assert rules(findings_for(tmp_path / "mixed.safetensors")) == [
        "bad-entry",
        "shard-truncated",
    ]


def test_empty_shard_warns(tmp_path):
    write_raw_entries(tmp_path / "none.safetensors", {})
    found = findings_for(tmp_path / "none.safetensors")
    assert rules(found) == ["empty-shard"]
    assert found[0].severity == "warning"


def test_multiple_independent_problems_all_fire(tmp_path):
    write_raw_entries(
        tmp_path / "multi.safetensors",
        {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "WAT", "shape": [4], "data_offsets": [24, 40]},
        },
        blob=payload(40),
    )
    assert rules(findings_for(tmp_path / "multi.safetensors")) == [
        "offset-gap",
        "unknown-dtype",
    ]
