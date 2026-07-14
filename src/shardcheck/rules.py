"""The rule registry: every finding shardcheck can produce, with stable ids.

Rule ids are part of the public contract — they appear in text and JSON
output, ``shardcheck explain`` documents them, and scripts are expected to
grep for them — so ids never change meaning and are only ever added.
Severity is a property of the rule, not the individual finding: ``error``
means the checkpoint will fail or corrupt a load, ``warning`` means it will
load but something is off (wasted bytes, stale bookkeeping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Rule:
    """One class of finding: stable id, severity, and documentation."""

    id: str
    severity: str
    title: str
    detail: str


_RULES = [
    # --- index-level ---------------------------------------------------
    Rule(
        "index-unreadable",
        ERROR,
        "the index file cannot be parsed",
        "The *.index.json file is missing, unreadable, not valid JSON, or "
        "structurally invalid (no weight_map, non-string shard names, paths "
        "escaping the checkpoint directory). Nothing can be cross-checked "
        "until the index itself is fixed.",
    ),
    Rule(
        "missing-shard",
        ERROR,
        "the index references a shard file that is not on disk",
        "weight_map maps one or more tensors to a shard file that does not "
        "exist next to the index. Classic causes: an interrupted upload or "
        "download, or an index regenerated after shards were renamed.",
    ),
    Rule(
        "orphan-shard",
        WARNING,
        "a shard file on disk is not referenced by the index",
        "A *.safetensors file sits next to the index but no weight_map entry "
        "points at it. Loaders ignore it, so it is dead weight — usually a "
        "leftover from an earlier save with a different shard count.",
    ),
    Rule(
        "missing-tensor",
        ERROR,
        "a mapped tensor is absent from its shard's header",
        "The index maps a tensor to a shard, but the shard's header does not "
        "contain that name and no other shard does either. The load will "
        "fail with a KeyError deep inside the loader.",
    ),
    Rule(
        "wrong-shard",
        ERROR,
        "a tensor lives in a different shard than the index says",
        "The tensor exists, but in another shard than the one weight_map "
        "names. A strict loader fails; a lazy one may read the wrong file. "
        "This is the signature of a stale index over re-sharded files.",
    ),
    Rule(
        "duplicate-tensor",
        ERROR,
        "the same tensor name appears in more than one shard",
        "Two or more shard headers define the same tensor name. Which copy "
        "wins depends on load order, so the checkpoint is ambiguous. Often "
        "caused by mixing shards from two different saves in one directory.",
    ),
    Rule(
        "unmapped-tensor",
        WARNING,
        "a shard holds a tensor the index does not map",
        "A tensor exists in a shard header but has no weight_map entry. "
        "Index-driven loaders will silently never load it — if it matters, "
        "the model is quietly missing a weight.",
    ),
    Rule(
        "total-size-mismatch",
        WARNING,
        "metadata.total_size disagrees with the shards",
        "The index's metadata.total_size does not equal the sum of tensor "
        "byte sizes across all shards. Harmless to most loaders but a "
        "reliable smell that index and shards come from different saves.",
    ),
    # --- shard container -----------------------------------------------
    Rule(
        "header-invalid",
        ERROR,
        "a shard's safetensors header is unusable",
        "The 8-byte length prefix is corrupt, the header is truncated or "
        "over the 100 MB limit, or the header JSON does not parse into an "
        "object. Nothing inside this shard can be checked or loaded.",
    ),
    Rule(
        "bad-entry",
        ERROR,
        "a tensor entry in a shard header is malformed",
        "One entry violates the safetensors schema: dtype not a string, "
        "shape not a list of non-negative integers, data_offsets not a "
        "non-negative ordered pair. Other entries in the shard are still "
        "checked.",
    ),
    Rule(
        "duplicate-name",
        ERROR,
        "a tensor name is defined twice inside one header",
        "The header JSON contains the same key twice. Every JSON parser "
        "silently keeps one of the two definitions, so this corruption is "
        "invisible to loaders — shardcheck detects it at the byte level.",
    ),
    Rule(
        "unknown-dtype",
        ERROR,
        "a tensor declares a dtype outside the specification",
        "The dtype string is not one of the fifteen types the safetensors "
        "specification defines, so element size — and therefore the layout "
        "of every byte after it — cannot be trusted.",
    ),
    Rule(
        "size-mismatch",
        ERROR,
        "a tensor's byte range disagrees with its dtype and shape",
        "data_offsets spans a different number of bytes than "
        "product(shape) * itemsize(dtype). One of the three is lying; a "
        "loader will either crash or reinterpret garbage.",
    ),
    Rule(
        "shard-truncated",
        ERROR,
        "the file is shorter than its header promises",
        "The header declares tensor data extending past the end of the "
        "file. This is the half-uploaded / half-downloaded shard: the "
        "header (written first) is intact, the payload tail is gone. The "
        "finding reports exactly how many bytes are missing.",
    ),
    Rule(
        "trailing-bytes",
        WARNING,
        "the file is longer than its header accounts for",
        "Bytes exist after the last declared tensor range. Loaders ignore "
        "them, but they usually mean a broken resume appended data or the "
        "header was rewritten in place over a larger payload.",
    ),
    Rule(
        "offset-overlap",
        ERROR,
        "two tensors in one shard claim overlapping bytes",
        "Two data_offsets ranges intersect, so the tensors share bytes. "
        "The reference implementation refuses such files; anything that "
        "does load them aliases memory between two weights.",
    ),
    Rule(
        "offset-gap",
        WARNING,
        "unclaimed bytes sit between tensor ranges",
        "The payload has byte ranges no tensor claims. Valid files pack "
        "tensors contiguously from offset 0, so gaps indicate a hand-edited "
        "header or a buggy writer — and wasted disk on every copy.",
    ),
    Rule(
        "empty-shard",
        WARNING,
        "a shard contains no tensors",
        "The header parses but defines zero tensors. Legal per the spec, "
        "pointless in a checkpoint — check whether this shard's contents "
        "went missing, or whether it should simply be deleted.",
    ),
]

#: Public registry, keyed by rule id, in documentation order.
RULES: Dict[str, Rule] = {rule.id: rule for rule in _RULES}


def severity_of(rule_id: str) -> str:
    """Severity for a rule id (raises ``KeyError`` for unknown ids)."""
    return RULES[rule_id].severity
