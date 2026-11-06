import pytest
from wsm.document_loader import load_from_texts
from wsm.engine import SearchEngine
from wsm.normalizer import Normalizer
from wsm.query_processor import QuerySyntaxError, difference, intersect, union
from wsm.tokenizer import tokenize
from wsm.wildcard import PermutermIndex

DOCS = [
    ("d0", "The quick brown fox jumps over the lazy dog"),
    ("d1", "Computers compute; computing is fun. Café culture!"),
    ("d2", "Information retrieval and web search: inverted index, postings"),
    ("d3", "A lazy index of the quick information"),
]


@pytest.fixture
def engine():
    return SearchEngine(load_from_texts(DOCS))


def ids(engine, q):
    return [h.doc_id for h in engine.search(q).hits]


def test_tokenizer():
    assert tokenize("Don't stop-me, now_42!") == ["Don't", "stop", "me", "now", "42"]


def test_normalizer_case_and_accents():
    n = Normalizer()
    assert [t for _, t in n.process("Café DOG")] == ["cafe", "dog"]


def test_stopword_positions_preserved():
    n = Normalizer(remove_stopwords=True)
    assert n.process("the quick fox") == [(1, "quick"), (2, "fox")]


def test_merge_algorithms():
    assert intersect([1, 3, 5], [3, 4, 5]) == [3, 5]
    assert union([1, 3], [2, 3, 9]) == [1, 2, 3, 9]
    assert difference([1, 2, 3, 4], [2, 4]) == [1, 3]


def test_single_term_case_insensitive(engine):
    assert ids(engine, "QUICK") == [0, 3]


def test_boolean_ops(engine):
    assert ids(engine, "quick AND lazy") == [0, 3]
    assert ids(engine, "quick lazy") == [0, 3]          # implicit AND
    assert ids(engine, "fox OR postings") == [0, 2]
    assert ids(engine, "quick AND NOT fox") == [3]
    assert ids(engine, "NOT quick") == [1, 2]
    assert ids(engine, "(fox OR postings) AND NOT retrieval") == [0]


def test_phrase(engine):
    assert ids(engine, '"inverted index"') == [2]
    assert ids(engine, '"index inverted"') == []
    assert ids(engine, '"quick brown"') == [0]


def test_wildcards(engine):
    assert ids(engine, "comput*") == [1]
    assert ids(engine, "*ing") == [1]
    assert ids(engine, "q*k") == [0, 3]
    assert ids(engine, "i*x") == [2, 3]
    assert "computers" in engine.search("comput*").expansions["comput*"]


def test_permuterm_multi_star():
    p = PermutermIndex(["compute", "computing", "complete", "cat"])
    assert p.match("co*t*") == ["complete", "compute", "computing"]
    assert p.match("c*e") == ["complete", "compute"]


def test_accent_query(engine):
    assert ids(engine, "café") == [1]
    assert ids(engine, "CAFE") == [1]


@pytest.mark.parametrize("bad", ["", "   ", "(fox", "fox AND", "AND fox", '"unclosed', "fox )"])
def test_syntax_errors(engine, bad):
    with pytest.raises(QuerySyntaxError):
        engine.search(bad)


def test_trace_steps_present(engine):
    r = engine.search('comput* AND "quick brown"')
    titles = [t for t, _ in r.steps]
    assert titles[0].startswith("Step 1") and titles[-1].startswith("Step 4")
    assert any("Wildcard term" in t for t in titles)
    assert any("Phrase" in t for t in titles)
    assert any("AND" in t for t in titles)
    assert "Permuterm rotation" in r.steps_text()


def test_wildcard_explain():
    p = PermutermIndex(["hello", "help", "halo"])
    info = p.explain("hel*o")
    assert info["rotated"] == "o$hel*" and info["prefix"] == "o$hel"
    assert info["matches"] == ["hello"]
