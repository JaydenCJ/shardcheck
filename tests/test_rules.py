"""The rule registry is a public contract; these tests freeze it."""

from __future__ import annotations

from shardcheck.rules import ERROR, RULES, WARNING, severity_of


def test_registry_has_the_documented_eighteen_rules():
    assert len(RULES) == 18


def test_every_rule_id_is_lowercase_kebab_case():
    for rule_id in RULES:
        assert rule_id == rule_id.lower()
        assert " " not in rule_id and "_" not in rule_id


def test_every_rule_has_severity_title_and_detail():
    for rule in RULES.values():
        assert rule.severity in (ERROR, WARNING)
        assert rule.title and not rule.title.endswith(".")
        assert len(rule.detail) > 40  # detail must actually explain, not restate


def test_severity_split_matches_the_readme_table():
    errors = {rid for rid, r in RULES.items() if r.severity == ERROR}
    warnings = {rid for rid, r in RULES.items() if r.severity == WARNING}
    assert severity_of("shard-truncated") == ERROR
    assert severity_of("orphan-shard") == WARNING
    assert errors == {
        "index-unreadable",
        "missing-shard",
        "missing-tensor",
        "wrong-shard",
        "duplicate-tensor",
        "header-invalid",
        "bad-entry",
        "duplicate-name",
        "unknown-dtype",
        "size-mismatch",
        "shard-truncated",
        "offset-overlap",
    }
    assert warnings == {
        "orphan-shard",
        "unmapped-tensor",
        "total-size-mismatch",
        "trailing-bytes",
        "offset-gap",
        "empty-shard",
    }
