"""The ``shardcheck`` command-line interface.

Three subcommands:

* ``check`` — the preflight itself. Exit code 0 when the checkpoint is
  loadable (warnings allowed unless ``--strict``), 1 when findings block a
  load, 2 when the target could not be checked at all.
* ``ls`` — one row per shard: tensor count, claimed bytes, file bytes, and
  a status column that survives missing or unreadable files.
* ``explain`` — the rule catalog, or the full documentation for one rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from . import __version__
from .errors import FormatError, IndexFileError, TargetError
from .indexfile import load_index
from .locate import MODE_INDEX, Target, list_shards, resolve
from .report import render_json, render_listing_text, render_text
from .rules import RULES
from .safetensors import read_shard
from .validate import validate_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shardcheck",
        description=(
            "Validate sharded safetensors checkpoints before load time: "
            "missing, duplicated, truncated or overlapping tensors across shards."
        ),
    )
    parser.add_argument(
        "--version", action="version", version="shardcheck %s" % __version__
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    check = sub.add_parser(
        "check",
        help="validate a checkpoint (index file, shard file, or directory)",
        description=(
            "Run every check against the target and print a per-shard report. "
            "Exit 0 = loadable, 1 = findings, 2 = target not checkable."
        ),
    )
    check.add_argument("path", help="an *.index.json, a *.safetensors, or a directory")
    check.add_argument("--json", action="store_true", help="emit one JSON document")
    check.add_argument(
        "--strict", action="store_true", help="treat warnings as failures (exit 1)"
    )
    check.add_argument(
        "-v", "--verbose", action="store_true", help="also list clean shards"
    )
    check.set_defaults(func=cmd_check)

    ls = sub.add_parser(
        "ls",
        help="list shards with tensor counts and sizes",
        description="One row per shard, including missing and orphaned files.",
    )
    ls.add_argument("path", help="an *.index.json, a *.safetensors, or a directory")
    ls.add_argument("--json", action="store_true", help="emit one JSON document")
    ls.set_defaults(func=cmd_ls)

    explain = sub.add_parser(
        "explain",
        help="document the rules (all, or one by id)",
        description="Without an argument, list every rule; with one, print its full documentation.",
    )
    explain.add_argument("rule", nargs="?", help="a rule id, e.g. shard-truncated")
    explain.set_defaults(func=cmd_explain)

    return parser


def cmd_check(args: argparse.Namespace) -> int:
    target = resolve(args.path)
    report = validate_target(target)
    if args.json:
        print(render_json(report))
    else:
        print(render_text(report, verbose=args.verbose))
    if report.errors:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


def _shard_row(path: str, status: str) -> Dict[str, object]:
    """One ``ls`` row; degrades gracefully when the file is damaged."""
    row: Dict[str, object] = {
        "shard": os.path.basename(path),
        "tensors": None,
        "tensor_bytes": None,
        "file_bytes": None,
        "status": status,
    }
    if not os.path.isfile(path):
        row["status"] = "missing"
        return row
    row["file_bytes"] = os.path.getsize(path)
    try:
        shard = read_shard(path)
    except (FormatError, OSError):
        row["status"] = "unreadable"
        return row
    row["tensors"] = len(shard.tensors)
    row["tensor_bytes"] = shard.tensor_bytes
    return row


def _listing_rows(target: Target) -> List[Dict[str, object]]:
    if target.mode == MODE_INDEX:
        index = load_index(target.index_path)
        rows = [
            _shard_row(index.shard_path(name), "referenced")
            for name in index.shard_names
        ]
        referenced = set(index.shard_names)
        for name in list_shards(index.directory):
            if name not in referenced:
                rows.append(_shard_row(index.shard_path(name), "orphan"))
        return rows
    return [_shard_row(path, "loose") for path in target.shard_paths]


def cmd_ls(args: argparse.Namespace) -> int:
    target = resolve(args.path)
    try:
        rows = _listing_rows(target)
    except IndexFileError as exc:
        raise TargetError("cannot list %s: %s" % (args.path, exc)) from None
    if args.json:
        print(json.dumps({"schema": 1, "shards": rows}, indent=2, sort_keys=True))
    else:
        print(render_listing_text(rows))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    if args.rule is None:
        width = max(len(rule_id) for rule_id in RULES)
        for rule in RULES.values():
            print("%-*s  %-7s  %s" % (width, rule.id, rule.severity, rule.title))
        return 0
    rule = RULES.get(args.rule)
    if rule is None:
        raise TargetError(
            "unknown rule %r; run 'shardcheck explain' for the full list" % args.rule
        )
    print("%s (%s)" % (rule.id, rule.severity))
    print(rule.title)
    print()
    print(rule.detail)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except TargetError as exc:
        print("shardcheck: %s" % exc, file=sys.stderr)
        return 2
    except BrokenPipeError:
        # The consumer (e.g. ``| head``) closed our stdout mid-write. Point
        # the fd at devnull so the interpreter's exit-time flush cannot raise
        # again, and exit with the conventional 128 + SIGPIPE status.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 141


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py
    sys.exit(main())
