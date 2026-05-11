"""
build_similarity_network.py
============================
Builds a semantic similarity network of EU laws with community detection.

Source file: Data/laws.csv

Columns used:
  celex                          — unique law identifier (node ID)
  celex_decoded [by DB]          — short readable label (e.g. "Regulation 2011/25")
  title_en                       — full title shown in the sidebar on click
  title_short                    — preferred short label (often empty; falls back to celex_decoded)
  md_content_1/2/3               — full law text (3 overflow columns)

What is used for SIMILARITY:
  Article sections only — text from the first "Article" heading onwards,
  stripping preamble/recitals boilerplate that is identical across all EU laws
  and would otherwise inflate cosine similarity to 0.95+.

What is shown in the PANEL on click:
  Full law text (md_content_1+2+3), capped at 15,000 chars in the JSON.
  Rendered as markdown.

Cases CSV is loaded and available as cases_df but not yet wired into the graph.

Pipeline:
  1. Load legislation shortlist; concatenate md_content_1/2/3 into one text per law.
  2. Embed each law (one vector per law) with MiniLM.
  3. Compute full cosine similarity matrix (vectors are L2-normalised → dot product).
  4. Sparsify: keep top-K edges per node above a similarity floor.
  5. Detect communities (greedy modularity via networkx).
  6. Emit network_data.json consumed by network.html.

Usage (defaults work from any directory):
    python deliverables/build_similarity_network.py

    Override:
    python deliverables/build_similarity_network.py \
        --laws "Data/260421-legislation-shortlist.csv" \
        --cases "Data/260421-cases-shortlist.csv" \
        --output "deliverables/network_data.json" \
        --top-k-edges 6 \
        --min-similarity 0.45
"""
from __future__ import annotations
import networkx
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------- #
# Column definitions
# -------------------------------------------------------------------- #

# Laws — columns read from Data/laws.csv
LAWS_COLS = [
    "celex",
    "celex_decoded [by DB]",
    "title_en",
    "title_short",
    "md_content_1",
    "md_content_2_overflow_column",
    "md_content_3_overflow_column",
]

_ARTICLE_RE = re.compile(r'^#{1,6}\s+Article\b', re.MULTILINE | re.IGNORECASE)
DISPLAY_TEXT_CAP = 15_000   # chars stored per law in the JSON for panel display

# Cases — columns retained (available as cases_df; not yet used in graph)
CASES_IDENTIFIER_COLUMNS = [
    "Shared Base Case Number [by_DB]",
    "Case Number [publishedId]",
    "CELEX [celex]",
    "Case Name (EN) [usualNameML]",
]
CASES_CLUSTERING_SIGNAL_COLUMNS = [
    "Subject Matter [matCodeML]",
    "Descriptor [descriptorML]",
    "Markdown Content [by_DB]",
    "Outgoing Legislation Refs in Reasoning [citationsMotif]",
    "Outgoing Legislation Refs in Operative Part [citationsDispositif]",
    "Outgoing Cited Case Law in Reasoning (CELEX) [citationsMotif]",
    "Outgoing Cited Case Law in Operative Part (CELEX) [citationsDispositif]",
    "Incoming Cited By (Cases) [by_DB]",
]
CASES_FILTERING_COLUMNS = [
    "Court [jurisdiction]",
    "Document Type [docType]",
    "Formation [formationML]",
    "Year Lodged [introductionYear]",
    "Close Date (yyyy-mm-dd) [closeDate]",
]
CASES_COLS = CASES_IDENTIFIER_COLUMNS + CASES_CLUSTERING_SIGNAL_COLUMNS + CASES_FILTERING_COLUMNS


# -------------------------------------------------------------------- #
# Loaders
# -------------------------------------------------------------------- #

def load_laws(path: Path, sep: str = ";") -> pd.DataFrame:
    """One row per law.

    Sets two text fields:
      content      — article sections only (used for embedding)
      display_text — full law text capped at DISPLAY_TEXT_CAP (shown in panel)
    """
    import csv as _csv
    _csv.field_size_limit(10 ** 7)
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False,
                     encoding="utf-8-sig", engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in LAWS_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[load_laws] missing columns: {missing}")
    df = df[LAWS_COLS].copy()

    full = (
        df["md_content_1"].str.strip() + "\n"
        + df["md_content_2_overflow_column"].str.strip() + "\n"
        + df["md_content_3_overflow_column"].str.strip()
    ).str.strip()

    # Article text: everything from the first "## Article" heading onwards.
    # Strips the preamble/recitals boilerplate identical across all EU laws.
    def _articles(text: str) -> str:
        m = _ARTICLE_RE.search(text)
        return text[m.start():].strip() if m else text

    df["content"]      = full.apply(_articles)
    df["display_text"] = full.str[:DISPLAY_TEXT_CAP]
    df["text_truncated"] = full.str.len() > DISPLAY_TEXT_CAP

    # Label: title_short → celex_decoded → celex
    label = df["title_short"].str.strip()
    label = label.where(label != "", df["celex_decoded [by DB]"].str.strip())
    label = label.where(label != "", df["celex"])
    df["label"] = label

    df = df[df["content"].str.len() > 40].reset_index(drop=True)
    print(f"[load_laws] {len(df):,} laws  "
          f"({df['text_truncated'].sum()} display texts capped at {DISPLAY_TEXT_CAP//1000}KB)")
    return df


def load_cases(path: Path, sep: str = ";") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in CASES_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[load_cases] missing columns: {missing}")
    df = df[CASES_COLS].copy()
    print(f"[load_cases] {len(df):,} cases")
    return df


# -------------------------------------------------------------------- #
# Embedding
# -------------------------------------------------------------------- #

CHUNK_WORDS = 150  # MiniLM hard-caps at 256 word-pieces; ~150 words fits safely


def embed(texts: list[str], model_name: str, batch: int = 64) -> np.ndarray:
    """Chunk each text → embed all chunks → mean-pool back to one vector per law.

    Without chunking, MiniLM silently truncates long laws to their first ~150 words
    (EU preamble boilerplate), making every law look 0.95+ similar to every other.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[embed] loading {model_name}")
    model = SentenceTransformer(model_name)

    # Build flat list of chunks and remember which doc each belongs to
    all_chunks: list[str] = []
    chunk_doc: list[int] = []
    for doc_idx, text in enumerate(texts):
        words = text.split()
        for start in range(0, max(1, len(words)), CHUNK_WORDS):
            chunk = " ".join(words[start: start + CHUNK_WORDS])
            if chunk:
                all_chunks.append(chunk)
                chunk_doc.append(doc_idx)

    print(f"[embed] encoding {len(all_chunks):,} chunks for {len(texts):,} laws")
    chunk_vecs = model.encode(
        all_chunks,
        batch_size=batch,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # normalise after mean-pooling
    ).astype(np.float32)

    # Mean-pool chunks → one vector per document
    dim = chunk_vecs.shape[1]
    doc_vecs = np.zeros((len(texts), dim), dtype=np.float32)
    counts   = np.zeros(len(texts), dtype=np.int32)
    for chunk_idx, doc_idx in enumerate(chunk_doc):
        doc_vecs[doc_idx] += chunk_vecs[chunk_idx]
        counts[doc_idx]   += 1
    counts = np.maximum(counts, 1)
    doc_vecs /= counts[:, None]

    # L2-normalise so cosine == dot product
    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_vecs /= np.maximum(norms, 1e-8)
    return doc_vecs


# -------------------------------------------------------------------- #
# Similarity + sparsification
# -------------------------------------------------------------------- #

def build_edges(vecs: np.ndarray, top_k: int, min_sim: float) -> list[dict]:
    """Cosine similarity matrix → sparse edge list."""
    print(f"[sim] computing {len(vecs):,} x {len(vecs):,} cosine matrix")
    sim = vecs @ vecs.T  # (N, N), values in [-1, 1]

    kept: set[tuple[int, int]] = set()
    n = sim.shape[0]
    for i in range(n):
        row = sim[i].copy()
        row[i] = -1.0
        idx = np.argpartition(-row, min(top_k, n - 1))[:top_k]
        for j in idx:
            if row[j] >= min_sim:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                kept.add((a, b))

    edges = [
        {"source": a, "target": b, "weight": round(float(sim[a, b]), 4)}
        for a, b in sorted(kept)
    ]
    print(f"[edges] {len(edges):,} edges (top-{top_k} per node, floor={min_sim})")
    return edges


# -------------------------------------------------------------------- #
# Community detection
# -------------------------------------------------------------------- #

def detect_communities(edges: list[dict], n_nodes: int) -> list[int]:
    """Greedy modularity communities via networkx. Returns community id per node."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])

    communities = greedy_modularity_communities(G, weight="weight")
    node_community = [0] * n_nodes
    for cid, members in enumerate(communities):
        for node in members:
            node_community[node] = cid

    print(f"[community] {len(communities)} communities detected")
    return node_community


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #

def main() -> None:
    DATA = Path("C:/ADS MASTER/THESIS PROJECT/Data")
    DELIVERABLES = Path("C:/ADS MASTER/THESIS PROJECT/deliverables")

    ap = argparse.ArgumentParser()
    ap.add_argument("--laws", type=Path,
                    default=DATA / "laws.csv")
    ap.add_argument("--cases", type=Path,
                    default=DATA / "260421-cases-shortlist.csv")
    ap.add_argument("--output", type=Path,
                    default=DELIVERABLES / "network_data.json")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-k-edges", type=int, default=6,
                    help="Max edges to keep per law node.")
    ap.add_argument("--min-similarity", type=float, default=0.45,
                    help="Similarity floor; edges below this are dropped.")
    ap.add_argument("--cache", type=Path,
                    default=DELIVERABLES / ".embeddings_cache.npz")
    args = ap.parse_args()

    # ---- load ----
    cases_df = load_cases(args.cases)
    laws_df  = load_laws(args.laws)

    # ---- embed (one vector per law) ----
    if args.cache.exists():
        print(f"[cache] loading {args.cache}")
        z = np.load(args.cache, allow_pickle=True)
        if int(z["n_rows"]) == len(laws_df):
            vecs = z["vecs"]
        else:
            print("[cache] size mismatch — re-embedding")
            vecs = embed(laws_df["content"].tolist(), args.model)   # article text only
            np.savez_compressed(args.cache, vecs=vecs, n_rows=len(laws_df))
    else:
        vecs = embed(laws_df["content"].tolist(), args.model)
        np.savez_compressed(args.cache, vecs=vecs, n_rows=len(laws_df))
        print(f"[cache] saved to {args.cache}")

    # ---- edges + communities ----
    edges = build_edges(vecs, args.top_k_edges, args.min_similarity)
    communities = detect_communities(edges, len(laws_df))

    # ---- nodes ----
    nodes = []
    for i, row in laws_df.iterrows():
        nodes.append({
            "id":            i,
            "celex":         row["celex"],
            "label":         row["label"],           # short — shown on graph
            "name":          row["title_en"] or row["label"],  # full — shown in panel
            "community":     communities[i],
            "display_text":  row["display_text"],
            "text_truncated": bool(row["text_truncated"]),
        })

    # neighbour lookup for sidebar
    neighbours: dict[int, list[dict]] = {i: [] for i in range(len(laws_df))}
    for e in edges:
        neighbours[e["source"]].append({"id": e["target"], "weight": e["weight"]})
        neighbours[e["target"]].append({"id": e["source"], "weight": e["weight"]})
    for i in neighbours:
        neighbours[i].sort(key=lambda x: -x["weight"])

    payload = {
        "meta": {
            "model":          args.model,
            "n_laws":         len(laws_df),
            "n_edges":        len(edges),
            "n_communities":  max(communities) + 1,
            "top_k_edges":    args.top_k_edges,
            "min_similarity": args.min_similarity,
        },
        "nodes":      nodes,
        "edges":      edges,
        "neighbours": neighbours,
    }

    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[done] {args.output}  ({size_mb:.1f} MB)")
    print(f"       {len(nodes):,} laws  /  {len(edges):,} edges  /  {max(communities)+1} communities")


if __name__ == "__main__":
    main()
