"""
server.py
Flask backend for the Legal Search HTML.

Provides two endpoints:
  POST /api/search     – embed query with BGE-M3, return Platinum-tier semantic hits
  GET  /api/fulltext   – return cleaned assembled text for a given parent_id

Expected folder layout (dataset/ and corpus/ are NOT in git — download from Google Drive):
  Visualizing-the-EU-s-Digital-Legal-Landscape/
  ├── search.html
  ├── server.py
  ├── dataset/
  │   ├── 260422-legislation-shortlist.csv
  │   ├── 260422-cases-shortlist.csv
  │   ├── legislation-eurlex-enriched.csv
  │   └── 260418-cases-merged.csv
  └── corpus/
      ├── corpus_full.parquet
      ├── dense_embeddings.npy
      ├── corpus_meta.parquet
      └── sparse_weights.json

Run:
    python server.py
    python server.py --threshold 0.40   # adjust similarity cutoff
    python server.py --port 5001
    python server.py --html /other/path/search.html

The HTML is served at http://localhost:5001/
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)   # allow fetch from HTML file opened locally

# Paths
BASE_DIR      = Path(__file__).parent
CORPUS_DIR    = BASE_DIR / "corpus"
DATASET_DIR   = BASE_DIR / "dataset"
# Shortlist CSVs — loaded fully into memory at startup (fast primary lookup)
LAW_CSV_SHORT  = DATASET_DIR / "260422-legislation-shortlist.csv"
CASE_CSV_SHORT = DATASET_DIR / "260422-cases-shortlist.csv"
# Full CSVs — queried on-demand for misses (too large to load entirely)
LAW_CSV_FULL   = DATASET_DIR / "legislation-eurlex-enriched.csv"
CASE_CSV_FULL  = DATASET_DIR / "260418-cases-merged.csv"
HTML_PATH      = None   # set from --html arg at startup (default below)

DEFAULT_THRESHOLD = 0.50
TOP_N             = 100     # max platinum results returned

# Global state (loaded once at startup)
_model           = None
_dense_vecs      = None
_corpus          = None   # corpus_full.parquet DataFrame
_docs            = None   # one row per parent_id, with chunk_indices
_law_lookup      = None   # {celex:       text}  — shortlist, in memory
_case_lookup     = None   # {ecli:        text}  — shortlist, in memory
_case_num_lookup = None   # {case_number: text}  — shortlist, in memory
_fallback_cache  = {}     # {parent_id: text}   — full-CSV hits cached after first lookup

BOILERPLATE_HEADERS = [
    "MAIN DOCUMENT", "BACKGROUND", "FROM WHEN DOES",
    "last update", "RELATED ACTS",
]


# ── Text helpers (mirrors preprocess_v2.py) ───────────────────────────────────
def _safe(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def clean_legal_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").strip()
    for marker in BOILERPLATE_HEADERS:
        pattern = re.compile(rf"(^|\n)#+\s*{re.escape(marker)}.*", re.IGNORECASE | re.DOTALL)
        text = pattern.sub("", text)
        idx = text.lower().find(marker.lower())
        if idx != -1:
            text = text[:idx]
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bby\s+\.', '', text)
    text = re.sub(r'\bof\s+on\b', 'of', text)
    text = re.sub(r'\bfrom\s+,', '', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text.strip()


def _assemble_law(row) -> str:
    parts = [
        _safe(row.get("md_content_1")),
        _safe(row.get("md_content_2_overflow_column")),
        _safe(row.get("md_content_3_overflow_column")),
    ]
    return "\n\n".join(p for p in parts if p)


def _assemble_case(row) -> str:
    return _safe(row.get("Markdown Content [by_DB]"))


# Startup: load everything once 
def load_data(threshold: float):
    global _model, _dense_vecs, _corpus, _docs, _law_lookup, _case_lookup, _case_num_lookup

    print("Loading BGE-M3 model...")
    from FlagEmbedding import BGEM3FlagModel
    _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    print("  Model ready.")

    print("Loading corpus embeddings...")
    _corpus     = pd.read_parquet(CORPUS_DIR / "corpus_full.parquet")
    _dense_vecs = np.load(CORPUS_DIR / "dense_embeddings.npy")
    assert len(_corpus) == len(_dense_vecs), "Corpus/embedding row mismatch"
    print(f"  {len(_corpus):,} chunks loaded.")

    # Build doc-level index: one entry per parent_id with chunk indices
    _corpus["_idx"] = range(len(_corpus))
    records = []
    for parent_id, group in _corpus.groupby("parent_id", sort=False):
        meta     = group.iloc[0]
        shared   = meta.get("Shared Base Case Number [by_DB]", "")
        title    = (meta.get("title_en") or
                    meta.get("Case Name (EN) [usualNameML]") or "")
        records.append({
            "parent_id":     parent_id,
            "source":        meta["source"],
            "shared_num":    str(shared).strip() if pd.notna(shared) and shared != "" else "",
            "title":         str(title).strip() if pd.notna(title) else "",
            "chunk_indices": group["_idx"].tolist(),
        })
    _docs = pd.DataFrame(records).reset_index(drop=True)
    print(f"  {len(_docs):,} documents indexed.")

    print("Loading shortlist CSVs into memory...")
    df_law  = pd.read_csv(LAW_CSV_SHORT,  sep=";", low_memory=False)
    df_case = pd.read_csv(CASE_CSV_SHORT, sep=";", low_memory=False)

    _law_lookup = {
        str(row["celex"]).strip(): clean_legal_text(_assemble_law(row))
        for _, row in df_law.iterrows()
        if pd.notna(row.get("celex"))
    }
    _case_lookup = {}
    _case_num_lookup = {}
    for _, row in df_case.iterrows():
        cleaned = clean_legal_text(_assemble_case(row))
        if pd.notna(row.get("ECLI [ecli]")):
            _case_lookup[str(row["ECLI [ecli]"]).strip()] = cleaned
        if pd.notna(row.get("Case Number [publishedId]")):
            _case_num_lookup[str(row["Case Number [publishedId]"]).strip()] = cleaned
        if pd.notna(row.get("Shared Base Case Number [by_DB]")):
            _case_num_lookup.setdefault(str(row["Shared Base Case Number [by_DB]"]).strip(), cleaned)
    print(f"  {len(_law_lookup):,} laws  |  {len(_case_lookup):,} cases (ECLI)  |  {len(_case_num_lookup):,} cases (number) in lookup.")
    print(f"\nReady — threshold={threshold}  top_n={TOP_N}")


# Routes
@app.route("/")
def index():
    try:
        html = HTML_PATH.read_text(encoding="utf-8")
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    except FileNotFoundError:
        return f"<p>search.html not found at: {HTML_PATH}</p>", 404


@app.route("/api/search", methods=["POST"])
def api_search():
    data      = request.get_json(force=True)
    query     = (data.get("query") or "").strip()
    threshold = float(data.get("threshold", app.config["THRESHOLD"]))

    if not query:
        return jsonify({"results": []})

    # Embed query
    q_vec = _model.encode(
        [query],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]

    # Per-chunk cosine similarity → max per doc
    chunk_scores = cosine_similarity(q_vec, _dense_vecs)[0]

    results = []
    for i, row in _docs.iterrows():
        idxs        = row["chunk_indices"]
        best_pos    = int(np.argmax(chunk_scores[idxs]))
        best_idx    = idxs[best_pos]
        best_score  = float(chunk_scores[best_idx])
        if best_score < threshold:
            continue
        best_chunk  = str(_corpus.iloc[best_idx]["text"])
        results.append({
            "parent_id":       row["parent_id"],
            "source":          row["source"],
            "shared_num":      row["shared_num"],
            "title":           row["title"],
            "score":           round(best_score, 4),
            "best_chunk_text": best_chunk,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return jsonify({"results": results[:TOP_N]})


def _fulltext_fallback(parent_id: str, source: str) -> str:
    """Scan the full CSV on-demand for documents not in the shortlist.
    Results are cached in _fallback_cache so repeated clicks are instant."""
    if parent_id in _fallback_cache:
        return _fallback_cache[parent_id]

    text = ""
    if source in ("legislation", ""):
        law_cols = ["celex", "md_content_1",
                    "md_content_2_overflow_column", "md_content_3_overflow_column"]
        for chunk in pd.read_csv(LAW_CSV_FULL, sep=";", low_memory=False,
                                 usecols=law_cols, chunksize=500):
            rows = chunk[chunk["celex"].astype(str).str.strip() == parent_id]
            if not rows.empty:
                text = clean_legal_text(_assemble_law(rows.iloc[0]))
                break

    if not text and source in ("caselaw", ""):
        case_cols = ["ECLI [ecli]", "Case Number [publishedId]",
                     "Shared Base Case Number [by_DB]", "Markdown Content [by_DB]"]
        for chunk in pd.read_csv(CASE_CSV_FULL, sep=";", low_memory=False,
                                 usecols=case_cols, chunksize=500):
            rows = chunk[
                (chunk["ECLI [ecli]"].astype(str).str.strip() == parent_id) |
                (chunk["Case Number [publishedId]"].astype(str).str.strip() == parent_id) |
                (chunk["Shared Base Case Number [by_DB]"].astype(str).str.strip() == parent_id)
            ]
            if not rows.empty:
                text = clean_legal_text(_assemble_case(rows.iloc[0]))
                break

    if text:
        _fallback_cache[parent_id] = text
        print(f"  [fallback] found {parent_id} in full CSV")
    return text


@app.route("/api/fulltext")
def api_fulltext():
    parent_id = request.args.get("parent_id", "").strip()
    source    = request.args.get("source", "").strip()

    def _case_get(key):
        return ((_case_lookup    or {}).get(key)
                or (_case_num_lookup or {}).get(key)
                or "")

    # 1. lookup shortlist
    if source == "legislation":
        text = (_law_lookup or {}).get(parent_id, "")
    elif source == "caselaw":
        text = _case_get(parent_id)
    else:
        text = (_law_lookup or {}).get(parent_id) or _case_get(parent_id) or ""

    # 2. Fallback: scan full CSV (slow first time, cached after)
    if not text:
        text = _fulltext_fallback(parent_id, source)

    if not text:
        return jsonify({"text": "", "error": f"No text found for {parent_id}"}), 404

    return jsonify({"text": text, "parent_id": parent_id, "source": source})


# Entry point 
_DEFAULT_HTML = BASE_DIR / "search.html"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--html", type=Path, default=_DEFAULT_HTML,
                        help="Path to search.html (default: sibling repo folder)")
    args = parser.parse_args()

    HTML_PATH = args.html
    app.config["THRESHOLD"] = args.threshold
    load_data(args.threshold)
    app.run(port=args.port, debug=False)
