"""Module 7 - Frontend GUI (Streamlit)."""
import time
from pathlib import Path
 
import streamlit as st
 
from wsm.document_loader import load_directory, load_from_texts
from wsm.engine import SearchEngine
from wsm.query_processor import QuerySyntaxError
 
DATA_DIR = Path(__file__).parent / "data"
 
st.set_page_config(page_title="WSM Search Engine", page_icon="🔎", layout="wide")
 
 
@st.cache_resource(show_spinner="Building inverted index...")
def get_engine(corpus: tuple, remove_stopwords: bool) -> SearchEngine:
    return SearchEngine(load_from_texts(corpus), remove_stopwords)
 
 
def render_steps(steps, animate: bool) -> None:
    with st.status("Retrieval steps", expanded=True, state="running") as status:
        for title, lines in steps:
            st.markdown(f"**{title}**")
            st.code("\n".join(lines), language=None)
            if animate:
                time.sleep(0.6)
        status.update(label="Retrieval steps (complete)", state="complete", expanded=True)
 
 
# ---------------- sidebar: corpus + options ----------------
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
    st.header("Display")
    show_steps = st.checkbox("Show retrieval steps", value=True)
    animate = st.checkbox("Animate steps (demo mode)", value=False)
 
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
        response = engine.search(query)
        st.session_state["result_text"] = response.to_text()
        st.session_state["steps"] = response.steps
    except QuerySyntaxError as err:
        st.session_state["result_text"] = f"Query error: {err}"
        st.session_state["steps"] = None
 
if show_steps and st.session_state.get("steps"):
    render_steps(st.session_state["steps"], animate and submitted)
 
st.text_area(
    "Results",
    value=st.session_state.get("result_text", "Enter a query and press Search."),
    height=420,
    disabled=True,  # read-only result area
)
 
