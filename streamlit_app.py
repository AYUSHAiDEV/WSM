"""Module 7 - Frontend GUI (Streamlit)."""
from pathlib import Path

import streamlit as st

from wsm.document_loader import load_directory
from wsm.engine import SearchEngine
from wsm.query_processor import QuerySyntaxError

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="WSM Search Engine", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Building inverted index...")
def get_engine(corpus: tuple, remove_stopwords: bool) -> SearchEngine:
    from wsm.document_loader import load_from_texts
    return SearchEngine(load_from_texts(corpus), remove_stopwords)


# ---------------- sidebar: corpus + index options ----------------
with st.sidebar:
    st.header("Corpus & index")
    uploads = st.file_uploader("Upload .txt documents (optional)", type=["txt", "md"], accept_multiple_files=True)
    remove_sw = st.checkbox("Remove stop words", value=False)
    if uploads:
        corpus = tuple((f.name.rsplit(".", 1)[0], f.getvalue().decode("utf-8", errors="ignore")) for f in uploads)
        st.caption(f"Using {len(corpus)} uploaded document(s).")
    else:
        corpus = tuple((d.title, d.text) for d in load_directory(DATA_DIR))
        st.caption(f"Using the {len(corpus)} bundled sample documents.")

engine = get_engine(corpus, remove_sw)

with st.sidebar:
    stats = engine.index.stats()
    c1, c2 = st.columns(2)
    c1.metric("Documents", stats["documents"])
    c2.metric("Vocabulary", stats["vocabulary"])
    c1.metric("Tokens", stats["tokens"])
    c2.metric("Postings", stats["postings"])

# ---------------- main page ----------------
st.title("🔎 WSM Search Engine")
st.caption("Positional inverted index · Boolean & phrase queries · Permuterm wildcard search")

with st.expander("Query syntax"):
    st.markdown(
        """
- `retrieval` — single term (case-insensitive)
- `index AND postings`, `crawler OR pagerank`, `NOT boolean` — Boolean operators (**UPPERCASE**)
- `inverted index` — adjacent terms are combined with implicit AND
- `"inverted index"` — exact phrase
- `comp*`, `*tion`, `r*val` — wildcards (permuterm index)
- `(boolean OR wildcard) AND NOT crawler` — parentheses for grouping
"""
    )

with st.form("search_form"):
    query = st.text_input("Search query", placeholder='e.g.  comp* AND "relevance judgments"')
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    try:
        st.session_state["result_text"] = engine.search(query).to_text()
    except QuerySyntaxError as err:
        st.session_state["result_text"] = f"Query error: {err}"

st.text_area(
    "Results",
    value=st.session_state.get("result_text", "Enter a query and press Search."),
    height=420,
    disabled=True,  # read-only result area
)
