# WSM Search Engine
**WORKING_LINK:**https://uhfcwwrkwg4v3qcbvhns6p.streamlit.app/
**Course:** Web Search Mining (WSM) — Assignment
**Topic:** Boolean, phrase and wildcard retrieval over a positional inverted index

## 1. Aim
Design and implement, from first principles and without IR libraries, a small search engine that
(a) builds an inverted index over a document collection, and (b) answers Boolean, phrase and
wildcard queries through a web GUI.

## 2. System architecture
```
 documents ──► document_loader ──► tokenizer ──► normalizer ──► inverted_index ─┐
                                                                                 ├─► query_processor ──► engine ──► streamlit_app (GUI)
                                         vocabulary ──► wildcard (permuterm) ────┘
```

| # | Module | File | Technique | Cost |
|---|--------|------|-----------|------|
| 1 | Document loading | `wsm/document_loader.py` | Recursive file read → `Document(id, title, text)` | O(total size) |
| 2 | Tokenization | `wsm/tokenizer.py` | Regex over letter/digit runs, internal apostrophes kept | O(n) |
| 3 | Normalization | `wsm/normalizer.py` | Case folding, accent stripping, punctuation removal, optional stop words | O(n) |
| 4 | Inverted index | `wsm/inverted_index.py` | Positional: term → {doc_id → [positions]} | O(total tokens) |
| 5 | Query processing | `wsm/query_processor.py` | Recursive-descent parser (NOT > AND > OR), two-pointer merges, phrase check via positions | O(sum of postings lengths) |
| 6 | Wildcard processing | `wsm/wildcard.py` | Permuterm index + binary search; regex post-filter for multi-`*` | O(log N + k) |
| 7 | Frontend GUI | `streamlit_app.py` | Search box, Search button, read-only result area | — |

`wsm/engine.py` is a thin facade that wires the modules together and formats results.

## 3. Design notes
* **Same pipeline for documents and queries.** Anything else breaks matching (see IIR §2.2).
* **Positions come from the original token stream**, so removing stop words leaves gaps rather
  than shifting positions; phrase queries stay correct.
* **Permuterm rotation.** Each term `t` is stored under every rotation of `t$`. The query `hel*o`
  becomes the prefix lookup `o$hel`. Cost: index size grows roughly by the average term length.
* **Operators are case-sensitive** (`AND`, `OR`, `NOT`), so the ordinary words *and/or/not*
  remain searchable.

## 4. Query syntax
| Query | Meaning |
|-------|---------|
| `retrieval` | single term, case-insensitive |
| `index AND postings` / `index postings` | conjunction (implicit AND) |
| `crawler OR pagerank` | disjunction |
| `NOT boolean` | complement w.r.t. the collection |
| `"inverted index"` | exact phrase |
| `comp*`, `*tion`, `r*val`, `co*t*` | wildcards |
| `(boolean OR wildcard) AND NOT crawler` | grouping |

## 5. Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Put your own `.txt` / `.md` files in `data/`, or upload them from the sidebar.

## 6. Test
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## 7. Deployment
Streamlit needs a long-running Python server with WebSockets, so it **cannot run on Vercel**
(serverless functions). Deploy on **Streamlit Community Cloud**:
1. Push this folder to a public GitHub repository.
2. Go to share.streamlit.io → *New app* → pick the repo, branch `main`, main file `streamlit_app.py`.
3. Deploy. You get a public `https://<name>.streamlit.app` link.

## 8. Limitations and future work
Boolean retrieval returns unranked sets. Natural extensions: TF-IDF / BM25 ranking, stemming or
lemmatization, k-gram index with spelling correction, skip pointers, index compression.

## 9. References
Manning, Raghavan, Schütze — *Introduction to Information Retrieval* (CUP, 2008): Ch. 1 (Boolean
retrieval), Ch. 2 (dictionary & postings), Ch. 3.2 (wildcard queries, permuterm index).
