from collections import Counter

from instruction_duplication.audit import _highlight_stem_coverage
from instruction_duplication.lexical import LexicalReference


def test_lexical_display_never_highlights_stopwords_and_marks_common_context() -> None:
    reference = LexicalReference(
        document_count=1000,
        document_frequency={"sacral": 10, "sulcus": 20},
        high_idf_threshold=6.0,
        idf_cap=10.0,
    )
    rendered = _highlight_stem_coverage(
        "does not radiate; sacral sulcus",
        Counter({"not": 1, "sacral": 1}),
        Counter({"sulcus": 1}),
        reference,
    )
    assert '<mark class="lex"' in rendered
    assert ">sacral</mark>" in rendered
    assert '<span class="lex-common"' in rendered
    assert ">sulcus</span>" in rendered
    assert ">not</mark>" not in rendered
    assert ">not</span>" not in rendered
