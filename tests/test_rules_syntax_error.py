from __future__ import annotations

import pytest

from answer_engineering import CompiledRules
from answer_engineering.rules.parse.errors import (
    RulesSyntaxError,
)


def test_parse_rules_invalid_syntax_raises_with_location() -> None:
    bad_rules = "## Replace (once): bad\n\nWith\n\n* missing colon\n"
    with pytest.raises(RulesSyntaxError) as exc:
        CompiledRules(bad_rules)

    err = exc.value
    assert err.line >= 1
    assert err.column >= 1
    assert err.snippet
