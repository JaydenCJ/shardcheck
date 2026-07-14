"""Index-level validation: the cross-references that make shardcheck useful.

Fixtures start from a fully valid two-shard checkpoint and break one thing
per test; assertions check both the rule that fires and the file the
finding is pinned to, because per-shard attribution is the product.
"""

from __future__ import annotations

import json
import os

from shardcheck.validate import validate, validate_index

from conftest import (
    SHARD_1,
    SHARD_2,
    make_checkpoint,
    model_tensors,
    payload,
    split_tensors,
    write_index,
    write_st,
)


def rules(report):
    return sorted(f.rule for f in report.findings)


def rewrite_index(index_path, mutate):
    doc = json.loads(open(index_path, encoding="utf-8").read())
    mutate(doc)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)


def test_valid_checkpoint_is_clean(checkpoint):
    report = validate_index(checkpoint)
    assert report.findings == []
    assert report.ok
    assert report.tensor_count == 6
    assert report.shards == [SHARD_1, SHARD_2]
    # validate() on the containing directory resolves to the same index.
    via_dir = validate(os.path.dirname(checkpoint))
    assert via_dir.mode == "index"
    assert via_dir.ok


def test_unreadable_index_is_a_single_finding(tmp_path):
    path = tmp_path / "model.safetensors.index.json"
    path.write_text("{broken", encoding="utf-8")
    report = validate_index(str(path))
    assert rules(report) == ["index-unreadable"]
    assert not report.ok


def test_missing_shard_is_one_finding_with_mapped_tensor_count(checkpoint):
    # The 3 tensors mapped to the absent shard must NOT also each produce
    # a missing-tensor finding; one actionable line beats four.
    os.remove(os.path.join(os.path.dirname(checkpoint), SHARD_2))
    report = validate_index(checkpoint)
    assert rules(report) == ["missing-shard"]
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.file == SHARD_2
    assert "3 tensors" in finding.message


def test_orphan_shard_warns_and_is_listed(checkpoint):
    directory = os.path.dirname(checkpoint)
    write_st(os.path.join(directory, "leftover.safetensors"), {"x": ("F32", [2], payload(8))})
    report = validate_index(checkpoint)
    assert rules(report) == ["orphan-shard"]
    assert report.findings[0].severity == "warning"
    assert "leftover.safetensors" in report.shards


def test_missing_tensor_fires_when_no_shard_has_it(checkpoint):
    rewrite_index(
        checkpoint, lambda doc: doc["weight_map"].update({"ghost.weight": SHARD_1})
    )
    report = validate_index(checkpoint)
    assert rules(report) == ["missing-tensor"]
    assert report.findings[0].tensor == "ghost.weight"
    assert report.findings[0].file == SHARD_1


def test_wrong_shard_points_at_the_real_location(checkpoint):
    # Map a shard-2 tensor to shard 1: the stale-index signature.
    rewrite_index(
        checkpoint, lambda doc: doc["weight_map"].update({"lm_head.weight": SHARD_1})
    )
    report = validate_index(checkpoint)
    assert rules(report) == ["wrong-shard"]
    assert SHARD_2 in report.findings[0].message
    assert report.findings[0].file == SHARD_1


def test_wrong_shard_found_in_an_orphan(checkpoint):
    # Tensor moved into a file the index never mentions: both findings fire
    # and wrong-shard names the orphan, so the fix is obvious.
    directory = os.path.dirname(checkpoint)
    rewrite_index(
        checkpoint,
        lambda doc: doc["weight_map"].update({"renamed.weight": SHARD_1}),
    )
    write_st(
        os.path.join(directory, "extra.safetensors"),
        {"renamed.weight": ("F32", [2], payload(8))},
    )
    report = validate_index(checkpoint)
    assert rules(report) == ["orphan-shard", "wrong-shard"]
    wrong = [f for f in report.findings if f.rule == "wrong-shard"][0]
    assert "extra.safetensors" in wrong.message


def test_unmapped_tensor_warns_per_referenced_shard(tmp_path):
    directory = tmp_path / "ckpt"
    index_path = make_checkpoint(directory)
    rewrite_index(index_path, lambda doc: doc["weight_map"].pop("model.norm.weight"))
    report = validate_index(index_path)
    assert rules(report) == ["unmapped-tensor"]
    unmapped = report.findings[0]
    assert unmapped.tensor == "model.norm.weight"
    assert unmapped.file == SHARD_2
    assert unmapped.severity == "warning"


def test_duplicate_tensor_across_shards_lists_all_holders(tmp_path):
    directory = tmp_path / "ckpt"
    directory.mkdir()
    shard_a, shard_b = split_tensors()
    shard_b["model.embed_tokens.weight"] = shard_a["model.embed_tokens.weight"]
    write_st(directory / SHARD_1, shard_a)
    write_st(directory / SHARD_2, shard_b)
    weight_map = {name: SHARD_1 for name in shard_a}
    weight_map.update({name: SHARD_2 for name in shard_b if name not in weight_map})
    total = sum(len(d) for _, _, d in shard_a.values()) + sum(
        len(d) for _, _, d in shard_b.values()
    )
    index_path = write_index(directory, weight_map, total)
    report = validate_index(index_path)
    dup = [f for f in report.findings if f.rule == "duplicate-tensor"]
    assert len(dup) == 1
    assert SHARD_1 in dup[0].message and SHARD_2 in dup[0].message


def test_total_size_mismatch_reports_both_numbers(tmp_path):
    index_path = make_checkpoint(tmp_path / "ckpt", total_size=999999)
    report = validate_index(index_path)
    assert rules(report) == ["total-size-mismatch"]
    finding = report.findings[0]
    assert "999,999" in finding.message and "400" in finding.message
    assert finding.severity == "warning"


def test_total_size_check_is_skipped_when_a_shard_is_missing(checkpoint):
    # A partial sum would always mismatch; that noise must be suppressed.
    os.remove(os.path.join(os.path.dirname(checkpoint), SHARD_1))
    report = validate_index(checkpoint)
    assert "total-size-mismatch" not in rules(report)


def test_unreadable_shard_becomes_header_invalid_without_tensor_noise(checkpoint):
    shard_path = os.path.join(os.path.dirname(checkpoint), SHARD_1)
    with open(shard_path, "wb") as handle:
        handle.write(b"\x00" * 4)  # not even a length prefix
    report = validate_index(checkpoint)
    assert rules(report) == ["header-invalid"]
    assert report.findings[0].file == SHARD_1


def test_truncated_shard_and_correct_shard_are_attributed_separately(checkpoint):
    directory = os.path.dirname(checkpoint)
    shard_a, _ = split_tensors()
    write_st(os.path.join(directory, SHARD_1), shard_a, truncate_tail=50)
    report = validate_index(checkpoint)
    assert rules(report) == ["shard-truncated"]
    assert report.findings[0].file == SHARD_1
    assert report.files_with_findings() == [SHARD_1]


def test_loose_directory_checks_shards_and_duplicates(tmp_path):
    directory = tmp_path / "loose"
    directory.mkdir()
    write_st(directory / "a.safetensors", {"w": ("F32", [2], payload(8))})
    write_st(directory / "b.safetensors", {"w": ("F32", [2], payload(8))})
    report = validate(str(directory))
    assert report.mode == "directory"
    assert rules(report) == ["duplicate-tensor"]
    assert report.tensor_count == 1


def test_single_file_mode_runs_container_checks_only(tmp_path):
    path = tmp_path / "model.safetensors"
    write_st(path, model_tensors(), truncate_tail=10)
    report = validate(str(path))
    assert report.mode == "file"
    assert rules(report) == ["shard-truncated"]
    assert report.shards == ["model.safetensors"]
