"""
server.py
Flask backend for the Legal Search HTML.

Provides two endpoints:
  POST /api/search     – embed query with BGE-M3, return Platinum-tier semantic hits
  GET  /api/fulltext   – return cleaned assembled text for a given parent_id

Expected folder layout (Data/ and corpus/ are NOT in git — download from Google Drive):
  Visualizing-the-EU-s-Digital-Legal-Landscape/
  ├── search.html
  ├── server.py
  ├── Data/
  │   ├── laws.csv
  │   ├── cases.csv
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
import json
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
DATASET_DIR   = BASE_DIR / "Data"
# Shortlist CSVs — loaded fully into memory at startup (fast primary lookup)
LAW_CSV_SHORT  = DATASET_DIR / "laws.csv"
CASE_CSV_SHORT = DATASET_DIR / "cases.csv"
# Full CSVs — same files; fallback scans them chunk-by-chunk for ids not in the in-memory lookup
LAW_CSV_FULL   = DATASET_DIR / "laws.csv"
CASE_CSV_FULL  = DATASET_DIR / "cases.csv"
HTML_PATH      = None   # set from --html arg at startup (default below)

DEFAULT_THRESHOLD = 0.50
TOP_N             = 100     # max platinum results returned

# Global state (loaded once at startup)
_model            = None
_dense_vecs       = None
_corpus           = None    # corpus_full.parquet DataFrame
_docs             = None    # one row per parent_id, with chunk_indices
_law_lookup       = None    # {celex:       text}  — shortlist, in memory
_case_lookup      = None    # {ecli:        text}  — shortlist, in memory
_case_num_lookup  = None    # {case_number: text}  — shortlist, in memory
_fallback_cache   = {}      # {parent_id: text}   — full-CSV hits cached after first lookup
_implicit_pairs   = []      # [{source, target, score}] law-law semantic pairs
_implicit_nodes   = []      # [{id, label, name, group, …}] law nodes
_implicit_case_pairs = []   # [{source (case), target (law), score}] case-law semantic pairs
_implicit_case_nodes = []   # [{id, label, name, is_case:True, …}] case nodes
_explicit_pair_set = set()  # {(min_id, max_id)} explicit pairs to exclude

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
        if not cleaned:
            continue
        # Key by every possible identifier so any parent_id from the corpus finds the text
        for _col, _lut in [
            ("ECLI [ecli]",                        _case_lookup),
            ("CELEX [celex]",                      _case_num_lookup),
            ("Case Number [publishedId]",           _case_num_lookup),
            ("Shared Base Case Number [by_DB]",     _case_num_lookup),
        ]:
            _val = str(row.get(_col, "") or "").strip()
            if _val and _val != "nan":
                _lut.setdefault(_val, cleaned)
    print(f"  {len(_law_lookup):,} laws  |  {len(_case_lookup):,} cases (ECLI)  |  {len(_case_num_lookup):,} cases (CELEX/number) in lookup.")
    try:
        _build_implicit_network()
    except Exception as _e:
        print(f"[WARNING] implicit network build failed — server still starts: {_e}")
    print(f"\nReady — threshold={threshold}  top_n={TOP_N}")


def _build_implicit_network():
    global _implicit_pairs, _implicit_nodes, _explicit_pair_set
    global _implicit_case_pairs, _implicit_case_nodes

    if _dense_vecs is None or _docs is None:
        return

    # Load explicit pairs blacklist + node metadata from explicit_network_data.json
    explicit_node_map      = {}
    explicit_case_node_map = {}
    for candidate in [
        BASE_DIR / "networks" / "explicit_network_layout.json",   # prefer layout (has x,y)
        BASE_DIR / "networks" / "explicit_network_data.json",
        BASE_DIR / "deliverables" / "explicit_network_layout.json",
        BASE_DIR / "deliverables" / "explicit_network_data.json",
    ]:
        if candidate.exists():
            ex = json.loads(candidate.read_text(encoding="utf-8"))
            for e in ex.get("law_law_edges", []):
                a, b = e["source"], e["target"]
                _explicit_pair_set.add((min(a, b), max(a, b)))
            explicit_node_map      = {n["id"]: n for n in ex.get("law_nodes",  [])}
            explicit_case_node_map = {n["id"]: n for n in ex.get("case_nodes", [])}
            print(f"  [implicit] {len(_explicit_pair_set)} explicit pairs blacklisted, "
                  f"{len(explicit_node_map)} law nodes, {len(explicit_case_node_map)} case nodes loaded")
            break
    else:
        print("  [implicit] explicit_network_data.json not found — blacklist skipped")

    # ── Law-law semantic similarity ───────────────────────────────────────
    law_docs = _docs[_docs["source"] == "legislation"].reset_index(drop=True)
    n = len(law_docs)
    print(f"  [implicit] computing similarity for {n} laws…")

    dim = _dense_vecs.shape[1]
    law_vecs = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        idxs = law_docs.iloc[i]["chunk_indices"]
        if idxs:
            law_vecs[i] = _dense_vecs[idxs].mean(axis=0)

    norms = np.linalg.norm(law_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    law_vecs /= norms

    sim = (law_vecs @ law_vecs.T).astype(np.float32)

    parent_ids = law_docs["parent_id"].tolist()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim[i, j])
            if score < 0.5:
                continue
            a, b = parent_ids[i], parent_ids[j]
            if (min(a, b), max(a, b)) in _explicit_pair_set:
                continue
            pairs.append({"source": a, "target": b, "score": round(score, 4)})

    _implicit_pairs = pairs
    print(f"  [implicit] {len(pairs)} law-law semantic pairs (≥0.5, explicit excluded)")

    nodes = []
    for i, pid in enumerate(parent_ids):
        if pid in explicit_node_map:
            nodes.append(explicit_node_map[pid])
        else:
            row = law_docs.iloc[i]
            nodes.append({
                "id": pid, "label": pid,
                "name": str(row.get("title", pid) or pid),
                "group": 19, "subject_matter": [], "eurovoc": [], "directory": [],
            })
    _implicit_nodes = nodes
    print(f"  [implicit] {len(nodes)} law nodes ready")

    # ── Case-law semantic similarity ──────────────────────────────────────
    # Build a flexible ID→node lookup (CELEX, case number, shared number, label)
    case_id_to_explicit = {}
    for nid, node in explicit_case_node_map.items():
        case_id_to_explicit[nid] = node
        for alias in [node.get("label", ""), node.get("name", "")]:
            if alias and alias != nid:
                case_id_to_explicit[alias] = node

    # Build subject_matter/descriptor lookup from cases CSV (covers all cases,
    # not just those already in the explicit network)
    case_csv_meta: dict = {}
    if CASE_CSV_SHORT.exists():
        try:
            _dcm = pd.read_csv(CASE_CSV_SHORT, sep=";", dtype=str,
                               keep_default_na=False, encoding="utf-8-sig")
            _dcm.columns = [c.strip() for c in _dcm.columns]
            for _, _r in _dcm.iterrows():
                sm = _r.get("Subject Matter [matCodeML]", "").strip()
                de = _r.get("Descriptor [descriptorML]", "").strip()
                cn = _r.get("Case Name (EN) [usualNameML]", "").strip()
                entry = {"subject_matter": sm, "descriptor": de, "case_name": cn}
                for _k in [_r.get("CELEX [celex]", ""),
                            _r.get("Case Number [publishedId]", ""),
                            _r.get("Shared Base Case Number [by_DB]", "")]:
                    _k = str(_k).strip()
                    if _k and _k != "nan":
                        case_csv_meta[_k] = entry
            print(f"  [implicit] {len(case_csv_meta)} case meta entries loaded from CSV")
        except Exception as _e:
            print(f"  [implicit] case meta CSV load failed: {_e}")

    # Include ALL caselaw documents in corpus (not restricted to explicit network)
    case_docs_all = _docs[_docs["source"] == "caselaw"].reset_index(drop=True)
    MAX_CASES = 400
    case_docs = (case_docs_all.head(MAX_CASES) if len(case_docs_all) > MAX_CASES
                 else case_docs_all).reset_index(drop=True)
    n_cases = len(case_docs)

    if n_cases > 0:
        print(f"  [implicit] computing case-law similarity for {n_cases} cases…")
        case_vecs = np.zeros((n_cases, dim), dtype=np.float32)
        for i in range(n_cases):
            idxs = case_docs.iloc[i]["chunk_indices"]
            if idxs:
                case_vecs[i] = _dense_vecs[idxs].mean(axis=0)

        norms_c = np.linalg.norm(case_vecs, axis=1, keepdims=True)
        norms_c[norms_c == 0] = 1.0
        case_vecs /= norms_c

        case_law_sim = (case_vecs @ law_vecs.T).astype(np.float32)  # (n_cases, n_laws)

        # Diagnostic: show similarity distribution so we can tune the threshold
        flat = case_law_sim.flatten()
        print(f"  [implicit] case-law sim  mean={flat.mean():.3f}  "
              f"max={flat.max():.3f}  p50={float(np.percentile(flat, 50)):.3f}  "
              f"p75={float(np.percentile(flat, 75)):.3f}")

        case_parent_ids = case_docs["parent_id"].tolist()
        TOP_CASE_K = 3
        # Use lower threshold (0.35) for cases — case-law BGE-M3 sims tend to be lower
        # than law-law sims due to different text styles
        CASE_SCORE_FLOOR = 0.35
        case_pairs = []
        for i in range(n_cases):
            top_k = np.argpartition(-case_law_sim[i], min(TOP_CASE_K, n - 1))[:TOP_CASE_K]
            for j in top_k:
                score = float(case_law_sim[i, j])
                if score >= CASE_SCORE_FLOOR:
                    case_pairs.append({
                        "source":       case_parent_ids[i],
                        "target":       parent_ids[j],
                        "score":        round(score, 4),
                        "is_case_edge": True,
                    })

        _implicit_case_pairs = case_pairs

        # Build case node list — use explicit network metadata where IDs match
        case_nodes = []
        for i in range(n_cases):
            row = case_docs.iloc[i]
            pid    = row["parent_id"]
            shared = str(row.get("shared_num", "") or "")
            title  = str(row.get("title", "") or "")
            # Try to match to explicit network node via any alias
            matched = case_id_to_explicit.get(pid) or case_id_to_explicit.get(shared)
            csv_meta = case_csv_meta.get(pid) or case_csv_meta.get(shared) or {}
            if matched:
                node = dict(matched)
                node["is_case"]   = True
                node["case_name"] = csv_meta.get("case_name", "")
                # Fill metadata from CSV if explicit network node had none
                if not node.get("subject_matter"):
                    node["subject_matter"] = csv_meta.get("subject_matter", "")
                if not node.get("descriptor"):
                    node["descriptor"] = csv_meta.get("descriptor", "")
                case_nodes.append(node)
            else:
                label = shared if shared and shared != "nan" else pid
                case_nodes.append({
                    "id":             pid,
                    "label":          label,
                    "name":           title if title and title != "nan" else label,
                    "case_name":      csv_meta.get("case_name", ""),
                    "is_case":        True,
                    "group":          19,
                    "subject_matter": csv_meta.get("subject_matter", ""),
                    "descriptor":     csv_meta.get("descriptor", ""),
                })

        _implicit_case_nodes = case_nodes
        print(f"  [implicit] {len(case_pairs)} case-law semantic pairs, {len(case_nodes)} case nodes ready")
    else:
        _implicit_case_pairs = []
        _implicit_case_nodes = []
        print("  [implicit] no caselaw documents found in corpus")


# Routes
@app.route("/")
def index():
    try:
        html = HTML_PATH.read_text(encoding="utf-8")
        return html, 200, {"Content-Type": "text/html; charset=utf-8"}
    except FileNotFoundError:
        return f"<p>search.html not found at: {HTML_PATH}</p>", 404


@app.route("/explicit_network_data.json")
def serve_explicit_json():
    # Prefer layout file (has baked-in x,y positions); fall back to data file
    for candidate in [
        BASE_DIR / "networks"     / "explicit_network_layout.json",
        BASE_DIR / "networks"     / "explicit_network_data.json",
        BASE_DIR / "deliverables" / "explicit_network_layout.json",
        BASE_DIR / "deliverables" / "explicit_network_data.json",
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8"), 200, {
                "Content-Type": "application/json; charset=utf-8"
            }
    return jsonify({"error": "explicit_network_data.json not found"}), 404


@app.route("/api/implicit_network")
def api_implicit_network():
    threshold = float(request.args.get("threshold", 0.5))
    threshold = max(0.3, min(1.0, threshold))
    law_edges  = [p for p in _implicit_pairs      if p["score"] >= threshold]
    case_edges = [p for p in _implicit_case_pairs if p["score"] >= threshold]
    return jsonify({
        "nodes":      _implicit_nodes,
        "cases":      _implicit_case_nodes,
        "edges":      law_edges,
        "case_edges": case_edges,
        "meta": {
            "n_laws":   len(_implicit_nodes),
            "n_cases":  len(_implicit_case_nodes),
            "n_edges":  len(law_edges) + len(case_edges),
            "threshold": threshold,
        }
    })


@app.route("/api/best_chunk")
def api_best_chunk():
    law_a = request.args.get("law_a", "").strip()
    law_b = request.args.get("law_b", "").strip()
    if not law_a or not law_b:
        return jsonify({"error": "need law_a and law_b"}), 400

    row_a = _docs[_docs["parent_id"] == law_a]
    row_b = _docs[_docs["parent_id"] == law_b]
    if row_a.empty or row_b.empty:
        return jsonify({"error": "law not found"}), 404

    # Mean embedding of law_a
    idxs_a = row_a.iloc[0]["chunk_indices"]
    vec_a  = _dense_vecs[idxs_a].mean(axis=0)
    norm   = np.linalg.norm(vec_a)
    vec_a  = vec_a / (norm if norm else 1.0)

    # Best chunk of law_b relative to law_a
    idxs_b   = row_b.iloc[0]["chunk_indices"]
    vecs_b   = _dense_vecs[idxs_b].astype(np.float32)
    norms_b  = np.linalg.norm(vecs_b, axis=1, keepdims=True)
    norms_b[norms_b == 0] = 1.0
    scores   = (vecs_b / norms_b) @ vec_a
    best_i   = int(np.argmax(scores))
    best_idx = idxs_b[best_i]

    return jsonify({
        "best_chunk": str(_corpus.iloc[best_idx]["text"]),
        "score":      round(float(scores[best_i]), 4),
    })


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
        case_cols = ["CELEX [celex]", "ECLI [ecli]", "Case Number [publishedId]",
                     "Shared Base Case Number [by_DB]", "Markdown Content [by_DB]"]
        for chunk in pd.read_csv(CASE_CSV_FULL, sep=";", low_memory=False,
                                 usecols=case_cols, chunksize=500):
            rows = chunk[
                (chunk["CELEX [celex]"].astype(str).str.strip() == parent_id) |
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
_DEFAULT_HTML = BASE_DIR / "networks" / "network.html"

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
