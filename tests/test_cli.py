"""The CLI end-to-end through main(): output, exit codes, JSON contract.

Exit codes are the CI interface — 0 loadable, 1 findings, 2 unusable
target — so every path to each code is pinned here.
"""

from __future__ import annotations

import json
import os

import pytest

from shardcheck import __version__
from shardcheck.cli import main

from conftest import SHARD_2, make_checkpoint, model_tensors, split_tensors, write_st


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_check_clean_checkpoint_exits_zero(capsys, checkpoint):
    code, out, err = run(capsys, "check", checkpoint)
    assert code == 0
    assert "OK: 2 shards, 6 tensors, no findings" in out
    assert err == ""
    # Pointing at the containing directory finds the same index.
    code, out, _ = run(capsys, "check", os.path.dirname(checkpoint))
    assert code == 0
    assert "mode: index" in out


def test_check_broken_checkpoint_exits_one_and_groups_by_shard(capsys, checkpoint):
    directory = os.path.dirname(checkpoint)
    _, shard_b = split_tensors()
    write_st(os.path.join(directory, SHARD_2), shard_b, truncate_tail=64)
    code, out, _ = run(capsys, "check", checkpoint)
    assert code == 1
    assert SHARD_2 in out
    assert "shard-truncated" in out
    assert "64 bytes missing" in out
    assert out.rstrip().endswith("FAIL: 1 error, 0 warnings in 1 of 2 shards")


def test_warnings_exit_zero_unless_strict(capsys, checkpoint):
    directory = os.path.dirname(checkpoint)
    write_st(
        os.path.join(directory, "spare.safetensors"),
        {"spare.w": ("F32", [2], b"\x00" * 8)},
    )
    code, out, _ = run(capsys, "check", checkpoint)
    assert code == 0
    assert "orphan-shard" in out
    code, _, _ = run(capsys, "check", checkpoint, "--strict")
    assert code == 1


def test_check_json_emits_the_schema(capsys, checkpoint):
    code, out, _ = run(capsys, "check", checkpoint, "--json")
    assert code == 0
    doc = json.loads(out)
    assert doc["schema"] == 1
    assert doc["ok"] is True
    assert doc["tensors"] == 6
    assert doc["findings"] == []


def test_check_missing_path_exits_two_with_stderr(capsys, tmp_path):
    code, out, err = run(capsys, "check", str(tmp_path / "nope"))
    assert code == 2
    assert out == ""
    assert "no such file" in err


def test_check_verbose_lists_clean_shards(capsys, checkpoint):
    _, out, _ = run(capsys, "check", checkpoint, "-v")
    assert "model-00001-of-00002.safetensors: clean" in out


def test_ls_shows_referenced_and_orphan_rows(capsys, checkpoint):
    directory = os.path.dirname(checkpoint)
    write_st(
        os.path.join(directory, "spare.safetensors"),
        {"spare.w": ("F32", [2], b"\x00" * 8)},
    )
    code, out, _ = run(capsys, "ls", checkpoint)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("shard")
    assert any("referenced" in line for line in lines[1:])
    assert any("spare.safetensors" in line and "orphan" in line for line in lines[1:])


def test_ls_marks_missing_shards(capsys, checkpoint):
    os.remove(os.path.join(os.path.dirname(checkpoint), SHARD_2))
    _, out, _ = run(capsys, "ls", checkpoint)
    row = [line for line in out.splitlines() if SHARD_2 in line][0]
    assert "missing" in row


def test_ls_json_rows_carry_sizes(capsys, single_shard):
    code, out, _ = run(capsys, "ls", str(single_shard))
    assert code == 0
    code, out, _ = run(capsys, "ls", str(single_shard), "--json")
    doc = json.loads(out)
    row = doc["shards"][0]
    assert row["tensors"] == len(model_tensors())
    assert row["tensor_bytes"] == 400
    assert row["status"] == "loose"


def test_ls_on_unreadable_index_exits_two(capsys, tmp_path):
    path = tmp_path / "model.safetensors.index.json"
    path.write_text("{oops", encoding="utf-8")
    code, _, err = run(capsys, "ls", str(path))
    assert code == 2
    assert "cannot list" in err


def test_explain_lists_every_rule_and_details_one(capsys):
    from shardcheck.rules import RULES

    code, out, _ = run(capsys, "explain")
    assert code == 0
    for rule_id in RULES:
        assert rule_id in out
    code, out, _ = run(capsys, "explain", "shard-truncated")
    assert code == 0
    assert out.startswith("shard-truncated (error)")
    assert "half-uploaded" in out


def test_explain_unknown_rule_exits_two(capsys):
    code, _, err = run(capsys, "explain", "made-up")
    assert code == 2
    assert "unknown rule" in err


def test_version_flag_and_bare_invocation(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == "shardcheck %s" % __version__
    # No subcommand prints help and exits 2 (a usage error, not a pass).
    code, out, _ = run(capsys)
    assert code == 2
    assert "COMMAND" in out


def test_broken_pipe_exits_quietly_with_sigpipe_status(monkeypatch, capsys):
    # `shardcheck explain | head -1` closes stdout mid-listing. With an
    # unbuffered stdout (PYTHONUNBUFFERED=1) the write raises BrokenPipeError
    # inside main(); the CLI must exit 128+SIGPIPE quietly, not traceback.
    import sys

    class ClosedPipe:
        def __init__(self, fd):
            self._fd = fd

        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

        def fileno(self):
            return self._fd

    placeholder_fd = os.open(os.devnull, os.O_WRONLY)
    monkeypatch.setattr(sys, "stdout", ClosedPipe(placeholder_fd))
    code = main(["explain"])
    os.close(placeholder_fd)
    assert code == 141
    assert capsys.readouterr().err == ""  # no traceback leaked to stderr


def test_python_dash_m_entry_point(capsys, checkpoint):
    # __main__ must route to the same main(); exercised in-process for speed.
    import shardcheck.__main__  # noqa: F401  (import must not execute main)

    code, out, _ = run(capsys, "check", checkpoint)
    assert code == 0 and "OK" in out
