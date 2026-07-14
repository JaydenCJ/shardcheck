"""Index parsing: strict structure, forgiving nothing, naming everything.

Every IndexFileError message must point at the offending key — these errors
are what a blocked deploy sees first.
"""

from __future__ import annotations

import json

import pytest

from shardcheck.errors import IndexFileError
from shardcheck.indexfile import load_index


def write_index(tmp_path, doc) -> str:
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_valid_index_round_trips(tmp_path):
    path = write_index(
        tmp_path,
        {
            "metadata": {"total_size": 400},
            "weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"},
        },
    )
    index = load_index(path)
    assert index.total_size == 400
    assert index.weight_map == {"a": "s1.safetensors", "b": "s2.safetensors"}
    assert index.shard_names == ["s1.safetensors", "s2.safetensors"]


def test_shard_names_preserve_first_appearance_order(tmp_path):
    path = write_index(
        tmp_path,
        {"weight_map": {"a": "z.safetensors", "b": "a.safetensors", "c": "z.safetensors"}},
    )
    assert load_index(path).shard_names == ["z.safetensors", "a.safetensors"]


def test_tensors_for_filters_by_shard(tmp_path):
    path = write_index(
        tmp_path,
        {"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors", "c": "s1.safetensors"}},
    )
    assert load_index(path).tensors_for("s1.safetensors") == ["a", "c"]


def test_shard_path_resolves_next_to_the_index(tmp_path):
    path = write_index(tmp_path, {"weight_map": {"a": "s1.safetensors"}})
    resolved = load_index(path).shard_path("s1.safetensors")
    assert resolved == str(tmp_path / "s1.safetensors")


def test_missing_file_raises(tmp_path):
    with pytest.raises(IndexFileError, match="cannot read index"):
        load_index(str(tmp_path / "absent.index.json"))


def test_unparseable_documents_raise(tmp_path):
    bad = tmp_path / "bad.index.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexFileError, match="not valid JSON"):
        load_index(str(bad))
    path = write_index(tmp_path, ["weight_map"])
    with pytest.raises(IndexFileError, match="expected an object"):
        load_index(path)


def test_missing_or_empty_weight_map_raises(tmp_path):
    path = write_index(tmp_path, {"metadata": {"total_size": 1}})
    with pytest.raises(IndexFileError, match="no 'weight_map'"):
        load_index(path)
    path = write_index(tmp_path, {"weight_map": {}})
    with pytest.raises(IndexFileError, match="maps no tensors"):
        load_index(path)


def test_non_string_shard_value_names_the_tensor(tmp_path):
    path = write_index(tmp_path, {"weight_map": {"layer.weight": 3}})
    with pytest.raises(IndexFileError, match="layer.weight"):
        load_index(path)


def test_paths_escaping_the_directory_are_rejected(tmp_path):
    # An index must never make shardcheck read outside the checkpoint dir.
    for evil in ("/etc/hosts.safetensors", "../up.safetensors", "a/../../up.safetensors"):
        path = write_index(tmp_path, {"weight_map": {"a": evil}})
        with pytest.raises(IndexFileError, match="outside the checkpoint directory"):
            load_index(path)


def test_bad_total_size_raises(tmp_path):
    # Negative, non-numeric, and the classic JSON true (a bool is an int in
    # Python, so it must be excluded explicitly).
    for bad_total in (-1, "big", True):
        path = write_index(
            tmp_path,
            {"metadata": {"total_size": bad_total}, "weight_map": {"a": "s.safetensors"}},
        )
        with pytest.raises(IndexFileError, match="total_size"):
            load_index(path)


def test_total_size_is_optional(tmp_path):
    path = write_index(tmp_path, {"weight_map": {"a": "s.safetensors"}})
    assert load_index(path).total_size is None
