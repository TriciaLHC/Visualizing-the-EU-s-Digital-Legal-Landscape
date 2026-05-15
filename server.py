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
import csv
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)   # allow fetch from HTML file opened locally

# CSV access log — saved to Google Drive if running in Colab, otherwise next to server.py
_GDRIVE = Path("/content/drive/MyDrive/CPDP_2026_thesis")
LOG_PATH = (_GDRIVE / "eu_legal_access.csv") if _GDRIVE.exists() else (Path(__file__).parent / "access.csv")

if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow(["timestamp", "ip", "method", "path", "status_code", "search_query", "search_threshold", "document_id"])


@app.after_request
def _log_and_patch(response):
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    # Extract which document/law/case was accessed per endpoint
    p = request.path
    if p == "/api/fulltext":
        doc_id = request.args.get("parent_id", "")
    elif p == "/api/related":
        doc_id = request.args.get("doc_id", "")
    elif p == "/api/best_chunk":
        law_a = request.args.get("law_a", "")
        law_b = request.args.get("law_b", "")
        doc_id = f"{law_a} <> {law_b}" if law_a and law_b else ""
    elif p == "/api/matching_chunks":
        doc_id = f"{request.args.get('query_id','')} <> {request.args.get('doc_id','')}"
    else:
        doc_id = ""

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow([
            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ip,
            request.method,
            request.path,
            response.status_code,
            g.get("search_query", ""),
            g.get("search_threshold", ""),
            doc_id,
        ])
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# Paths
BASE_DIR      = Path(__file__).parent
CORPUS_DIR    = BASE_DIR / "corpus"
DATASET_DIR   = BASE_DIR / "Data"
# CSV files — loaded fully into memory at startup; also scanned chunk-by-chunk as fallback
LAW_CSV  = DATASET_DIR / "laws.csv"
CASE_CSV = DATASET_DIR / "cases.csv"
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
_dense_normed     = None    # (n_chunks, dim) — all corpus chunk vectors, L2-normalised

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
        plain = re.search(rf'(?:^|\n)\s*{re.escape(marker)}\s*(?:\n|$)', text, re.IGNORECASE)
        if plain:
            text = text[:plain.start()]
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
    df_law  = pd.read_csv(LAW_CSV,  sep=";", low_memory=False)
    df_case = pd.read_csv(CASE_CSV, sep=";", low_memory=False)

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
    global _dense_normed

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
            # Also blacklist explicit case-law citations so they don't appear as
            # implicit semantic edges too (same connection shown in two views).
            _explicit_case_law_pairs = set()
            for e in ex.get("case_law_edges", []):
                _explicit_case_law_pairs.add((str(e["source"]), str(e["target"])))
            explicit_node_map      = {n["id"]: n for n in ex.get("law_nodes",  [])}
            explicit_case_node_map = {n["id"]: n for n in ex.get("case_nodes", [])}
            print(f"  [implicit] {len(_explicit_pair_set)} explicit law-law pairs blacklisted, "
                  f"{len(_explicit_case_law_pairs)} explicit case-law pairs blacklisted, "
                  f"{len(explicit_node_map)} law nodes, {len(explicit_case_node_map)} case nodes loaded")
            break
    else:
        _explicit_case_law_pairs = set()
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

    # Normalize all corpus chunk vectors once — reused by both law-law and case-law scoring
    _norms_all = np.linalg.norm(_dense_vecs, axis=1, keepdims=True)
    _norms_all[_norms_all == 0] = 1.0
    dense_normed = (_dense_vecs / _norms_all).astype(np.float32)
    _dense_normed = dense_normed   # persist for /api/related

    # ── Chunk-level law-law scoring ───────────────────────────────────────────
    # For each (law_i, law_j): score = max over chunks of law_i of (chunk · law_j_mean),
    # then symmetrise by taking max of both directions.
    parent_ids = law_docs["parent_id"].tolist()
    LAW_SCORE_FLOOR = 0.60

    # Collect all law chunks into one matrix and track which law owns each
    law_chunk_idx = []
    law_chunk_owner = []
    for i in range(n):
        idxs = law_docs.iloc[i]["chunk_indices"]
        law_chunk_idx.extend(idxs)
        law_chunk_owner.extend([i] * len(idxs))

    law_chunk_owner_arr = np.array(law_chunk_owner, dtype=np.int32)
    all_law_chunks = dense_normed[law_chunk_idx]          # (total_law_chunks, dim)
    chunk_law_sim  = (all_law_chunks @ law_vecs.T).astype(np.float32)  # (total_chunks, n_laws)

    # Per-law max: law_law_max[i, j] = max sim of law_i's chunks vs law_j's mean
    law_law_max = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        mask = law_chunk_owner_arr == i
        if mask.any():
            law_law_max[i] = chunk_law_sim[mask].max(axis=0)

    sym_sim = np.maximum(law_law_max, law_law_max.T)  # symmetric

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sym_sim[i, j])
            if score < LAW_SCORE_FLOOR:
                continue
            a, b = parent_ids[i], parent_ids[j]
            if (min(a, b), max(a, b)) in _explicit_pair_set:
                continue
            pairs.append({"source": a, "target": b, "score": round(score, 4)})

    _implicit_pairs = pairs
    print(f"  [implicit] {len(pairs)} law-law chunk-level pairs (floor≥{LAW_SCORE_FLOOR}, explicit excluded)")

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
    if CASE_CSV.exists():
        try:
            _dcm = pd.read_csv(CASE_CSV, sep=";", dtype=str,
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
    case_docs = _docs[_docs["source"] == "caselaw"].reset_index(drop=True)
    n_cases = len(case_docs)

    if n_cases > 0:
        print(f"  [implicit] computing case-law similarity for {n_cases} cases…")
        case_parent_ids = case_docs["parent_id"].tolist()
        # Chunk-level case-law scoring: for each case, score each law by
        # max(case_chunk · law_mean) — same method as the notebook.
        # This is more discriminative than mean-to-mean.
        CASE_SCORE_FLOOR = 0.60  # matches slider minimum (displayed 0.2 → internal 0.60)
        seen_pairs = set()
        case_pairs = []

        for i in range(n_cases):
            idxs = case_docs.iloc[i]["chunk_indices"]
            if not idxs:
                continue
            chunk_vecs = dense_normed[idxs]              # (n_chunks, dim)
            # scores[j] = max cosine sim of any case chunk vs law j mean
            scores = (chunk_vecs @ law_vecs.T).max(axis=0)  # (n_laws,)
            for j in np.where(scores >= CASE_SCORE_FLOOR)[0]:
                score = float(scores[j])
                key = (case_parent_ids[i], parent_ids[j])
                if key in seen_pairs:
                    continue
                # Skip pairs already represented by an explicit citation
                if key in _explicit_case_law_pairs:
                    continue
                seen_pairs.add(key)
                case_pairs.append({
                    "source":       case_parent_ids[i],
                    "target":       parent_ids[j],
                    "score":        round(score, 4),
                    "is_case_edge": True,
                })
        print(f"  [implicit] {len(case_pairs)} case-law chunk-level pairs (floor≥{CASE_SCORE_FLOOR}, explicit excluded)")

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
                node["corpus_id"] = pid   # actual parent_id in _docs (may differ from node id)
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
                    "corpus_id":      pid,   # same as id for unmatched cases
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

    def _resolve(doc_id: str):
        """Find _docs row trying parent_id, shared_num, and title in order."""
        rows = _docs[_docs["parent_id"] == doc_id]
        if not rows.empty:
            return rows
        rows = _docs[_docs["shared_num"] == doc_id]
        if not rows.empty:
            return rows
        # Last resort: title match (covers case name lookups)
        rows = _docs[_docs["title"] == doc_id]
        return rows

    row_a = _resolve(law_a)
    row_b = _resolve(law_b)
    if row_a.empty or row_b.empty:
        missing = []
        if row_a.empty: missing.append(f"law_a={law_a!r}")
        if row_b.empty: missing.append(f"law_b={law_b!r}")
        print(f"  [best_chunk] not found in corpus: {', '.join(missing)}")
        return jsonify({"error": "document not found in corpus"}), 404

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


@app.route("/api/matching_chunks")
def api_matching_chunks():
    """Return all chunks of doc_id that score >= threshold against query_id's mean embedding."""
    query_id  = request.args.get("query_id",  "").strip()
    doc_id    = request.args.get("doc_id",    "").strip()
    threshold = float(request.args.get("threshold", 0.5))
    if not query_id or not doc_id:
        return jsonify({"error": "need query_id and doc_id"}), 400

    def _resolve(pid):
        rows = _docs[_docs["parent_id"] == pid]
        if not rows.empty: return rows
        rows = _docs[_docs["shared_num"] == pid]
        if not rows.empty: return rows
        return _docs[_docs["title"] == pid]

    q_rows = _resolve(query_id)
    d_rows = _resolve(doc_id)
    if q_rows.empty or d_rows.empty:
        return jsonify({"chunks": []}), 404

    # Mean embedding of query document
    q_idxs = q_rows.iloc[0]["chunk_indices"]
    q_vec  = _dense_vecs[q_idxs].astype(np.float32).mean(axis=0)
    norm   = np.linalg.norm(q_vec)
    q_vec  = q_vec / (norm if norm else 1.0)

    # Score every chunk of the target document against the query mean
    d_idxs   = d_rows.iloc[0]["chunk_indices"]
    d_vecs   = _dense_vecs[d_idxs].astype(np.float32)
    d_norms  = np.linalg.norm(d_vecs, axis=1, keepdims=True)
    d_norms[d_norms == 0] = 1.0
    scores   = (d_vecs / d_norms) @ q_vec          # shape (n_chunks,)

    results = [
        {"score": round(float(scores[k]), 4), "text": str(_corpus.iloc[idx]["text"])}
        for k, idx in enumerate(d_idxs)
        if float(scores[k]) >= threshold
    ]
    results.sort(key=lambda x: -x["score"])
    return jsonify({"chunks": results})


@app.route("/api/related")
def api_related():
    """Given a selected law or case doc_id, return all other docs whose best
    chunk scores >= threshold against the query document's mean embedding.
    Mirrors the notebook's cell ⑤ logic, applied to the full corpus."""
    doc_id    = request.args.get("doc_id", "").strip()
    threshold = float(request.args.get("threshold", DEFAULT_THRESHOLD))
    if not doc_id:
        return jsonify({"error": "need doc_id"}), 400
    if _dense_normed is None:
        return jsonify({"error": "corpus not loaded"}), 503

    def _resolve(pid):
        rows = _docs[_docs["parent_id"] == pid]
        if not rows.empty: return rows
        rows = _docs[_docs["shared_num"] == pid]
        if not rows.empty: return rows
        return _docs[_docs["title"] == pid]

    q_rows = _resolve(doc_id)
    if q_rows.empty:
        return jsonify({"error": f"'{doc_id}' not found in corpus"}), 404

    # Mean embedding of query document, normalised
    q_idxs = q_rows.iloc[0]["chunk_indices"]
    q_vecs  = _dense_vecs[q_idxs].astype(np.float32)
    q_mean  = q_vecs.mean(axis=0)
    norm    = np.linalg.norm(q_mean)
    q_mean  = q_mean / (norm if norm else 1.0)

    # One matrix multiply: every corpus chunk vs query mean → shape (n_chunks,)
    all_scores = (_dense_normed @ q_mean).astype(np.float32)

    results = []
    for _, doc in _docs.iterrows():
        if doc["parent_id"] == q_rows.iloc[0]["parent_id"]:
            continue
        idxs        = doc["chunk_indices"]
        chunk_scores = all_scores[idxs]
        best_pos    = int(np.argmax(chunk_scores))
        best_score  = float(chunk_scores[best_pos])
        if best_score < threshold:
            continue
        best_chunk = str(_corpus.iloc[idxs[best_pos]]["text"])
        results.append({
            "parent_id":       doc["parent_id"],
            "source":          doc["source"],
            "shared_num":      doc["shared_num"],
            "title":           doc["title"],
            "score":           round(best_score, 4),
            "best_chunk_text": best_chunk,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return jsonify({"results": results[:TOP_N], "query_id": doc_id})


@app.route("/api/search", methods=["POST"])
def api_search():
    data      = request.get_json(force=True)
    query     = (data.get("query") or "").strip()
    threshold = float(data.get("threshold", app.config["THRESHOLD"]))

    if not query:
        return jsonify({"results": []})

    g.search_query = query
    g.search_threshold = threshold

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
        for chunk in pd.read_csv(LAW_CSV, sep=";", low_memory=False,
                                 usecols=law_cols, chunksize=500):
            rows = chunk[chunk["celex"].astype(str).str.strip() == parent_id]
            if not rows.empty:
                text = clean_legal_text(_assemble_law(rows.iloc[0]))
                break

    if not text and source in ("caselaw", ""):
        case_cols = ["CELEX [celex]", "ECLI [ecli]", "Case Number [publishedId]",
                     "Shared Base Case Number [by_DB]", "Markdown Content [by_DB]"]
        for chunk in pd.read_csv(CASE_CSV, sep=";", low_memory=False,
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


@app.route("/api/log_event", methods=["POST"])
def api_log_event():
    data = request.get_json(force=True) or {}
    ip   = request.headers.get("X-Forwarded-For", request.remote_addr)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as _f:
        csv.writer(_f).writerow([
            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            ip,
            "EVENT",
            data.get("event_type", ""),
            "",
            data.get("search_query", ""),
            data.get("search_threshold", ""),
            data.get("document_id", ""),
        ])
    return jsonify({"ok": True})


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


_law_names_cache  = None
_case_names_cache = None

def _load_names_csv(csv_path, key_hint: str, val_hint: str) -> dict:
    """Load a two-column CSV into a {key: value} dict using vectorised pandas ops."""
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str,
                         encoding="utf-8-sig", keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        key_col = next((c for c in df.columns if key_hint  in c.lower()), None)
        val_col = next((c for c in df.columns if val_hint  in c.lower()), None)
        if not key_col or not val_col:
            return {}
        keys = df[key_col].str.strip()
        vals = df[val_col].str.strip()
        result = {k: v for k, v in zip(keys, vals) if k and v}
        print(f"[names] {len(result)} entries loaded from {csv_path.name}")
        return result
    except Exception as e:
        print(f"[names] CSV load failed ({csv_path.name}): {e}")
        return {}


@app.route("/api/law_names")
def api_law_names():
    """Return {celex: human_readable_name} dict."""
    global _law_names_cache
    if _law_names_cache is None:
        _law_names_cache = _load_names_csv(
            DATASET_DIR / "260512-laws-celex-and-human-readable-names.csv",
            "celex", "human",
        )
    return jsonify(_law_names_cache)


@app.route("/api/case_names")
def api_case_names():
    """Return {shared_base_case_number: case_name} dict."""
    global _case_names_cache
    if _case_names_cache is None:
        _case_names_cache = _load_names_csv(
            DATASET_DIR / "260514091724-case-names-by-db.csv",
            "shared base", "case name",
        )
    return jsonify(_case_names_cache)


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
    app.run(port=args.port, debug=False, threaded=True)
