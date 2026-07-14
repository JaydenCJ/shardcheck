"""Target resolution: index vs directory vs single file, and the failures.

Everything the CLI turns into exit code 2 originates here, so ambiguity and
non-checkable paths must raise TargetError with a message naming the path.
"""

from __future__ import annotations

import pytest

from shardcheck.errors import TargetError
from shardcheck.locate import resolve

from conftest import make_checkpoint, payload, write_st


def test_index_file_resolves_to_index_mode(checkpoint):
    target = resolve(checkpoint)
    assert target.mode == "index"
    assert target.index_path == checkpoint


def test_directory_with_one_index_resolves_to_it(tmp_path):
    index_path = make_checkpoint(tmp_path / "ckpt")
    target = resolve(str(tmp_path / "ckpt"))
    assert target.mode == "index"
    assert target.index_path == index_path


def test_shard_file_resolves_to_file_mode(single_shard):
    target = resolve(str(single_shard))
    assert target.mode == "file"
    assert target.shard_paths == (str(single_shard),)


def test_directory_without_index_lists_shards_sorted(tmp_path):
    for name in ("b.safetensors", "a.safetensors"):
        write_st(tmp_path / name, {"w": ("F32", [2], payload(8))})
    target = resolve(str(tmp_path))
    assert target.mode == "directory"
    assert [p.rsplit("/", 1)[-1] for p in target.shard_paths] == [
        "a.safetensors",
        "b.safetensors",
    ]


def test_uncheckable_paths_raise(tmp_path):
    with pytest.raises(TargetError, match="no such file"):
        resolve(str(tmp_path / "nope"))
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(TargetError, match="neither"):
        resolve(str(config))


def test_empty_directory_raises(tmp_path):
    with pytest.raises(TargetError, match="no \\*\\.index\\.json"):
        resolve(str(tmp_path))


def test_two_indexes_is_ambiguous_and_names_both(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / "other.safetensors.index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TargetError) as excinfo:
        resolve(str(tmp_path))
    assert "model.safetensors.index.json" in str(excinfo.value)
    assert "other.safetensors.index.json" in str(excinfo.value)


def test_index_in_subdirectory_is_not_picked_up(tmp_path):
    # Resolution is deliberately non-recursive: checking a parent directory
    # must not silently validate a nested, unrelated checkpoint.
    make_checkpoint(tmp_path / "nested")
    with pytest.raises(TargetError):
        resolve(str(tmp_path))
