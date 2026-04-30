from __future__ import annotations

from answer_engineering.config.engine_defaults import MatchDefaults
from answer_engineering.rules.parse.ast import MatchOptionsAST
from answer_engineering.rules.parse.match_options import resolve_match_options


def test_resolve_match_options_item_overrides_all() -> None:
    resolved = resolve_match_options(
        item=MatchOptionsAST(casefold=False, word=True),
        section=MatchOptionsAST(casefold=True, word=False),
        rule=MatchOptionsAST(casefold=True, word=False),
        defaults=MatchDefaults(casefold=True, word=False),
    )
    assert resolved.casefold is False
    assert resolved.word is True


def test_resolve_match_options_section_overrides_rule_and_default() -> None:
    resolved = resolve_match_options(
        item=MatchOptionsAST(),
        section=MatchOptionsAST(casefold=False, word=True),
        rule=MatchOptionsAST(casefold=True, word=False),
        defaults=MatchDefaults(casefold=True, word=False),
    )
    assert resolved.casefold is False
    assert resolved.word is True


def test_resolve_match_options_rule_overrides_default() -> None:
    resolved = resolve_match_options(
        item=MatchOptionsAST(),
        section=MatchOptionsAST(),
        rule=MatchOptionsAST(casefold=False, word=True),
        defaults=MatchDefaults(casefold=True, word=False),
    )
    assert resolved.casefold is False
    assert resolved.word is True


def test_resolve_match_options_uses_defaults_when_all_unspecified() -> None:
    defaults = MatchDefaults(casefold=False, word=True)
    resolved = resolve_match_options(
        item=MatchOptionsAST(),
        section=MatchOptionsAST(),
        rule=MatchOptionsAST(),
        defaults=defaults,
    )
    assert resolved.casefold is False
    assert resolved.word is True
