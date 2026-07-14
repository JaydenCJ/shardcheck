"""Header parsing: the two-tier contract of shardcheck.safetensors.

Container-level damage must raise FormatError with a message a human can
act on; entry-level damage must land in Shard.bad_entries/duplicate_names
so the rest of the shard stays checkable.
"""

from __future__ import annotations

import json
import struct

import pytest

from shardcheck.errors import FormatError
from shardcheck.safetensors import DTYPE_SIZES, parse_header, read_shard

from conftest import model_tensors, payload, write_st


def test_read_shard_parses_all_tensors(single_shard):
    shard = read_shard(str(single_shard))
    assert set(shard.tensors) == set(model_tensors())
    assert shard.bad_entries == []
    assert shard.duplicate_names == []


def test_tensor_info_carries_dtype_shape_and_range(single_shard):
    shard = read_shard(str(single_shard))
    embed = shard.tensors["model.embed_tokens.weight"]
    assert embed.dtype == "F32"
    assert embed.shape == (4, 8)
    assert embed.nbytes == 128
    assert embed.numel == 32
    assert embed.expected_nbytes() == 128


def test_metadata_is_preserved_and_absence_stays_none(single_shard, tmp_path):
    assert read_shard(str(single_shard)).metadata == {"format": "pt"}
    # No __metadata__ key at all is distinct from an empty one.
    bare = tmp_path / "nometa.safetensors"
    write_st(bare, {"w": ("F32", [2], payload(8))})
    assert read_shard(str(bare)).metadata is None


def test_data_size_and_declared_end_agree_on_a_valid_file(single_shard):
    shard = read_shard(str(single_shard))
    # A well-formed contiguous file: payload exactly covers declared ranges.
    assert shard.data_size == shard.declared_end == shard.tensor_bytes


def test_degenerate_shapes_scalar_and_zero_dim(tmp_path):
    # Empty shape = scalar (one element); a 0 in the shape = zero bytes.
    path = tmp_path / "degenerate.safetensors"
    write_st(path, {"scalar": ("F32", [], payload(4)), "empty": ("F32", [0, 8], b"")})
    shard = read_shard(str(path))
    assert shard.tensors["scalar"].numel == 1
    assert shard.tensors["scalar"].expected_nbytes() == 4
    assert shard.tensors["empty"].numel == 0
    assert shard.tensors["empty"].nbytes == 0


def test_corrupt_length_prefixes_raise(tmp_path):
    # Shorter than the prefix itself, and a prefix of literal zero.
    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(b"\x01\x02\x03")
    with pytest.raises(FormatError, match="8-byte header length"):
        read_shard(str(tiny))
    zero = tmp_path / "zero.safetensors"
    zero.write_bytes(struct.pack("<Q", 0))
    with pytest.raises(FormatError, match="prefix is 0"):
        read_shard(str(zero))


def test_header_length_over_limit_raises():
    blob = struct.pack("<Q", 101 * 1024 * 1024) + b"{}"
    with pytest.raises(FormatError, match="limit"):
        parse_header(blob, file_size=10**9)


def test_truncated_header_raises_with_byte_counts(tmp_path):
    # The prefix promises 500 header bytes but the file ends after 10.
    path = tmp_path / "cut.safetensors"
    path.write_bytes(struct.pack("<Q", 500) + b"{" + b" " * 9)
    with pytest.raises(FormatError, match="header itself is truncated"):
        read_shard(str(path))


def test_unusable_header_json_raises(tmp_path):
    garbage = tmp_path / "garbage.safetensors"
    body = b"not json at all!"
    garbage.write_bytes(struct.pack("<Q", len(body)) + body)
    with pytest.raises(FormatError, match="not valid JSON"):
        read_shard(str(garbage))
    # Valid JSON that is not an object is equally unusable.
    array = tmp_path / "array.safetensors"
    array.write_bytes(struct.pack("<Q", 7) + b"[1,2,3]")
    with pytest.raises(FormatError, match="expected an object"):
        read_shard(str(array))


def test_duplicate_key_in_header_is_detected_not_merged(tmp_path):
    # json.loads silently keeps the last duplicate; shardcheck must not.
    body = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}'
    path = tmp_path / "dup.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body + payload(8))
    shard = read_shard(str(path))
    assert shard.duplicate_names == ["w"]
    # The first definition wins, so downstream checks see a stable view.
    assert shard.tensors["w"].start == 0


def test_bad_entry_is_isolated_from_good_entries(tmp_path):
    body = json.dumps(
        {
            "good": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "bad": {"dtype": 7, "shape": [2], "data_offsets": [8, 16]},
        }
    ).encode()
    path = tmp_path / "mixed.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body + payload(16))
    shard = read_shard(str(path))
    assert "good" in shard.tensors
    assert "bad" not in shard.tensors
    assert shard.bad_entries == [("bad", "'dtype' is missing or not a string")]


def test_every_entry_schema_violation_is_named(tmp_path):
    from conftest import write_raw_entries

    cases = [
        (["not", "an", "object"], "expected an object"),
        ({"shape": [1], "data_offsets": [0, 4]}, "dtype"),
        ({"dtype": "F32", "shape": "nope", "data_offsets": [0, 4]}, "shape"),
        ({"dtype": "F32", "shape": [1, -2], "data_offsets": [0, 4]}, "non-negative"),
        ({"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}, "non-negative"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [0]}, "pair of integers"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [0, 1.5]}, "pair of integers"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}, "reversed"),
        ({"dtype": "F32", "shape": [1], "data_offsets": [-4, 0]}, "negative"),
    ]
    for i, (entry, reason_fragment) in enumerate(cases):
        path = tmp_path / ("entry%d.safetensors" % i)
        write_raw_entries(path, {"t": entry}, blob=payload(8))
        shard = read_shard(str(path))
        assert shard.tensors == {}, entry
        assert len(shard.bad_entries) == 1, entry
        assert reason_fragment in shard.bad_entries[0][1], entry


def test_non_string_metadata_becomes_bad_entry(tmp_path):
    from conftest import write_raw_entries

    path = tmp_path / "meta.safetensors"
    write_raw_entries(path, {"__metadata__": {"steps": 100}})
    shard = read_shard(str(path))
    assert shard.metadata is None
    assert shard.bad_entries[0][0] == "__metadata__"


def test_dtype_table_matches_the_specification():
    # Fifteen dtypes, and the itemsizes that anchor every size check.
    assert len(DTYPE_SIZES) == 15
    assert DTYPE_SIZES["BF16"] == 2
    assert DTYPE_SIZES["F8_E4M3"] == 1
    assert DTYPE_SIZES["F64"] == 8


def test_only_header_bytes_are_read_from_large_payload(tmp_path):
    # A shard whose payload is far larger than its header: read_shard must
    # still be O(header). We prove it stays correct; speed follows from the
    # implementation reading exactly 8 + header_size bytes.
    path = tmp_path / "big.safetensors"
    write_st(path, {"blob": ("U8", [1 << 16], payload(1 << 16))})
    shard = read_shard(str(path))
    assert shard.tensors["blob"].nbytes == 1 << 16
    assert shard.file_size == path.stat().st_size
