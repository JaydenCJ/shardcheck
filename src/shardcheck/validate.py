"""The validator: turn a resolved target into a report full of findings.

Three layers of checks, from the inside out:

1. **Shard container checks** (:func:`check_shard`) — everything knowable
   from one header plus the file size: schema violations, unknown dtypes,
   dtype/shape/byte-range disagreements, overlapping or gapped offsets, and
   the flagship case, a payload shorter than the header promises (the
   half-uploaded shard).
2. **Cross-shard checks** — the same tensor name defined in two shards.
3. **Index checks** — the ``weight_map`` cross-referenced against reality:
   shards that are missing or orphaned, tensors that are absent, live in
   the wrong shard, or are never mapped, and a stale ``total_size``.

Orphan shards (on disk but unreferenced) are still parsed: a tensor the
index maps to shard A but which actually sits in an orphan B is reported as
``wrong-shard`` pointing at B — the exact signature of a stale index.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .errors import FormatError, IndexFileError
from .findings import INDEX_FILE, Finding, Report
from .indexfile import IndexFile, load_index
from .locate import MODE_INDEX, Target, list_shards, resolve
from .safetensors import DTYPE_SIZES, Shard, read_shard


def _n(value: int) -> str:
    """Human-readable integer: 12345 -> '12,345'."""
    return format(value, ",")


def check_shard(shard: Shard) -> List[Finding]:
    """All container-level findings for one parsed shard."""
    findings: List[Finding] = []
    name = os.path.basename(shard.path)

    def add(rule: str, message: str, tensor: Optional[str] = None) -> None:
        findings.append(Finding(rule=rule, file=name, message=message, tensor=tensor))

    for entry_name, reason in shard.bad_entries:
        add("bad-entry", "entry %r is malformed: %s" % (entry_name, reason), entry_name)
    for dup in shard.duplicate_names:
        add(
            "duplicate-name",
            "tensor %r is defined more than once in this header; JSON parsers "
            "silently pick one definition" % dup,
            dup,
        )
    if not shard.tensors and not shard.bad_entries:
        add("empty-shard", "header parses but defines no tensors")

    for tensor in shard.tensors.values():
        if tensor.dtype not in DTYPE_SIZES:
            add(
                "unknown-dtype",
                "%r declares dtype %r, which is not in the safetensors specification"
                % (tensor.name, tensor.dtype),
                tensor.name,
            )
            continue
        expected = tensor.expected_nbytes()
        if expected != tensor.nbytes:
            add(
                "size-mismatch",
                "%r is %s%s = %s bytes, but data_offsets [%s, %s) span %s"
                % (
                    tensor.name,
                    tensor.dtype,
                    list(tensor.shape),
                    _n(expected if expected is not None else 0),
                    _n(tensor.start),
                    _n(tensor.end),
                    _n(tensor.nbytes),
                ),
                tensor.name,
            )

    findings.extend(_layout_findings(shard, name))

    needed = 8 + shard.header_size + shard.declared_end
    if shard.file_size < needed:
        add(
            "shard-truncated",
            "file is %s bytes but header + tensor data need %s (%s bytes missing "
            "from the tail — half-uploaded?)"
            % (_n(shard.file_size), _n(needed), _n(needed - shard.file_size)),
        )
    elif shard.file_size > needed:
        add(
            "trailing-bytes",
            "%s unaccounted bytes after the last tensor (file is %s, header "
            "accounts for %s)"
            % (_n(shard.file_size - needed), _n(shard.file_size), _n(needed)),
        )
    return findings


def _layout_findings(shard: Shard, file_name: str) -> List[Finding]:
    """Overlap and gap findings from the shard's data_offsets intervals.

    Zero-length ranges (tensors with a 0 in their shape) occupy no bytes,
    so they can neither overlap anything nor plug a gap; they are skipped.
    """
    findings: List[Finding] = []
    ranges: List[Tuple[int, int, str]] = sorted(
        (t.start, t.end, t.name) for t in shard.tensors.values() if t.end > t.start
    )
    covered = 0  # everything below this offset is claimed by some tensor
    prev_name = None
    for start, end, tensor_name in ranges:
        if start < covered:
            findings.append(
                Finding(
                    rule="offset-overlap",
                    file=file_name,
                    tensor=tensor_name,
                    message="%r [%s, %s) overlaps %r (bytes below %s are already claimed)"
                    % (tensor_name, _n(start), _n(end), prev_name, _n(covered)),
                )
            )
        elif start > covered:
            findings.append(
                Finding(
                    rule="offset-gap",
                    file=file_name,
                    tensor=tensor_name,
                    message="%s unclaimed bytes at [%s, %s) before %r"
                    % (_n(start - covered), _n(covered), _n(start), tensor_name),
                )
            )
        if end > covered:
            covered = end
            prev_name = tensor_name
    return findings


def _read_or_report(path: str, report: Report) -> Optional[Shard]:
    """Parse a shard; on container-level damage, file a finding instead."""
    name = os.path.basename(path)
    try:
        shard = read_shard(path)
    except FormatError as exc:
        report.add("header-invalid", name, str(exc))
        return None
    except OSError as exc:
        report.add("header-invalid", name, "cannot read file: %s" % exc)
        return None
    report.findings.extend(check_shard(shard))
    return shard


def _duplicate_findings(shards: Dict[str, Shard]) -> List[Finding]:
    """Cross-shard ``duplicate-tensor`` findings, one per duplicated name."""
    holders: Dict[str, List[str]] = {}
    for shard_name, shard in shards.items():
        for tensor_name in shard.tensors:
            holders.setdefault(tensor_name, []).append(shard_name)
    findings = []
    for tensor_name in sorted(holders):
        files = holders[tensor_name]
        if len(files) > 1:
            findings.append(
                Finding(
                    rule="duplicate-tensor",
                    file=files[0],
                    tensor=tensor_name,
                    message="%r is defined in %d shards: %s"
                    % (tensor_name, len(files), ", ".join(files)),
                )
            )
    return findings


def validate_index(index_path: str, given: Optional[str] = None) -> Report:
    """Validate a sharded checkpoint through its ``*.index.json``."""
    report = Report(
        target=given or index_path, mode=MODE_INDEX, index_path=index_path
    )
    try:
        index = load_index(index_path)
    except IndexFileError as exc:
        report.add("index-unreadable", INDEX_FILE, str(exc))
        return report

    report.tensor_count = len(index.weight_map)
    referenced = index.shard_names
    on_disk = list_shards(index.directory)
    orphans = [s for s in on_disk if s not in set(referenced)]
    report.shards = referenced + orphans

    parsed: Dict[str, Shard] = {}
    missing: List[str] = []
    for shard_name in referenced:
        path = index.shard_path(shard_name)
        if not os.path.isfile(path):
            missing.append(shard_name)
            mapped = len(index.tensors_for(shard_name))
            report.add(
                "missing-shard",
                shard_name,
                "index maps %d %s here but the file is not on disk"
                % (mapped, "tensor" if mapped == 1 else "tensors"),
            )
            continue
        shard = _read_or_report(path, report)
        if shard is not None:
            parsed[shard_name] = shard

    for shard_name in orphans:
        report.add(
            "orphan-shard",
            shard_name,
            "present next to the index but never referenced by weight_map",
        )
        shard = _read_or_report(index.shard_path(shard_name), report)
        if shard is not None:
            parsed[shard_name] = shard

    report.findings.extend(_duplicate_findings(parsed))
    _check_weight_map(index, parsed, missing, report)
    _check_total_size(index, parsed, referenced, missing, report)
    return report


def _check_weight_map(
    index: IndexFile,
    parsed: Dict[str, Shard],
    missing: List[str],
    report: Report,
) -> None:
    """Per-tensor index checks: missing-tensor, wrong-shard, unmapped-tensor."""
    holders: Dict[str, List[str]] = {}
    for shard_name, shard in parsed.items():
        for tensor_name in shard.tensors:
            holders.setdefault(tensor_name, []).append(shard_name)

    for tensor_name, mapped_shard in index.weight_map.items():
        if mapped_shard in missing or (
            mapped_shard in index.shard_names and mapped_shard not in parsed
        ):
            # The mapped shard is absent or unreadable; its own finding
            # already explains everything a per-tensor line could add.
            continue
        holder_names = holders.get(tensor_name, [])
        if mapped_shard in holder_names:
            continue
        if holder_names:
            report.add(
                "wrong-shard",
                mapped_shard,
                "%r is mapped here but actually lives in %s"
                % (tensor_name, ", ".join(holder_names)),
                tensor=tensor_name,
            )
        else:
            report.add(
                "missing-tensor",
                mapped_shard,
                "%r is mapped here but no shard on disk defines it" % tensor_name,
                tensor=tensor_name,
            )

    # Orphan shards are skipped here on purpose: every tensor they hold is
    # unmapped by definition, and their own orphan-shard finding says so.
    for shard_name in index.shard_names:
        shard = parsed.get(shard_name)
        if shard is None:
            continue
        for tensor_name in shard.tensors:
            if tensor_name not in index.weight_map:
                report.add(
                    "unmapped-tensor",
                    shard_name,
                    "%r exists in this shard but weight_map never mentions it"
                    % tensor_name,
                    tensor=tensor_name,
                )


def _check_total_size(
    index: IndexFile,
    parsed: Dict[str, Shard],
    referenced: List[str],
    missing: List[str],
    report: Report,
) -> None:
    """Compare ``metadata.total_size`` with the bytes shards actually claim.

    Only meaningful when every referenced shard was present and readable;
    otherwise the sum is partial and the mismatch would be pure noise on
    top of the missing-shard/header-invalid findings already filed.
    """
    if index.total_size is None:
        return
    if missing or any(name not in parsed for name in referenced):
        return
    actual = sum(parsed[name].tensor_bytes for name in referenced)
    if actual != index.total_size:
        report.add(
            "total-size-mismatch",
            INDEX_FILE,
            "metadata.total_size says %s bytes, tensors in referenced shards "
            "sum to %s (off by %s)"
            % (
                _n(index.total_size),
                _n(actual),
                _n(abs(actual - index.total_size)),
            ),
        )


def validate_shard_set(paths, mode: str, given: str) -> Report:
    """Validate loose shards (no index): container checks + duplicates."""
    report = Report(target=given, mode=mode)
    parsed: Dict[str, Shard] = {}
    names: List[str] = []
    seen: set = set()
    for path in paths:
        name = os.path.basename(path)
        names.append(name)
        shard = _read_or_report(path, report)
        if shard is not None:
            parsed[name] = shard
            seen.update(shard.tensors)
    report.shards = names
    report.tensor_count = len(seen)
    report.findings.extend(_duplicate_findings(parsed))
    return report


def validate_target(target: Target) -> Report:
    """Validate a resolved :class:`~shardcheck.locate.Target`."""
    if target.mode == MODE_INDEX:
        return validate_index(target.index_path, given=target.given)
    return validate_shard_set(target.shard_paths, target.mode, target.given)


def validate(path: str) -> Report:
    """Resolve ``path`` (index file, shard, or directory) and validate it.

    Raises :class:`~shardcheck.errors.TargetError` when the path is not
    checkable at all; every other problem becomes a finding in the report.
    """
    return validate_target(resolve(path))
