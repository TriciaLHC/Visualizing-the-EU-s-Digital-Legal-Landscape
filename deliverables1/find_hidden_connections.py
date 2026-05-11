"""
find_hidden_connections.py
==========================
Surfaces the strongest *implicit* semantic links between laws and CJEU
cases — pairs where two documents discuss the same problem without
citing each other.

Why this matters for the thesis: it demonstrates that a semantic
model recovers connections invisible to citation graphs, validating
the retrieval system against a meaningful ground truth.

Pipeline:
  1. Load articles CSV (laws) and dispositif/cases CSV.
  2. Optionally split long articles into paragraphs so we score at
     the *paragraph* level (a single shared paragraph is more
     convincing than a diluted document-level cosine).
  3. Embed both sides with MiniLM.
  4. Compute paragraph<->paragraph cosine similarity in blocks.
  5. For each (law, case) pair keep the single strongest paragraph
     match.
  6. Remove pairs where the case explicitly cites the law (CELEX
     substring match, or via a 'cites' column if present).
  7. Rank by score; emit the top-N as JSON and as a styled HTML
     report.

Usage:
    python find_hidden_connections.py \
        --laws articles_clean.csv \
        --cases cases_clean.csv \
        --top 10 \
        --output hidden_connections
"""
import pip


%pip install networkx
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------- #
# Column resolution -- shared with build_similarity_network.py
# -------------------------------------------------------------------- #
LAW_ID  = ["celex", "law_celex", "law_id"]
LAW_NM  = ["law_name", "title", "law_title"]
CASE_ID = ["case_id", "celex", "ecli", "case_celex"]
CASE_NM = ["case_name", "title", "case_title", "name"]
CONTENT = ["content", "text", "article_text", "operative_text",
           "dispositif", "dispositif_text"]
CITES   = ["cites", "cited_celex", "cited_laws", "law_match_name_for_references",
           "references", "citations"]


def resolve(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"None of {candidates} in {list(df.columns)}")
    return None


# -------------------------------------------------------------------- #
# Paragraph splitting. The preprocessor's "content" is article-level;
# we split on blank lines and bolded paragraph markers (**1.**) so
# that very long articles contribute multiple comparable units.
# -------------------------------------------------------------------- #
_PARA_BREAK = re.compile(r"\n{2,}|(?:\*\*\d+\.\*\*)|(?<=\.)\s+(?=[A-Z][a-z])")
_WS = re.compile(r"\s+")

def paragraphs(text: str, min_len: int = 80) -> list[str]:
    """Split content into clean paragraph-sized units."""
    chunks = _PARA_BREAK.split(str(text))
    out = []
    for c in chunks:
        s = _WS.sub(" ", c).strip()
        if len(s) >= min_len:
            out.append(s)
    return out or ([_WS.sub(" ", str(text)).strip()] if str(text).strip() else [])


def explode_paragraphs(df: pd.DataFrame, content_col: str) -> pd.DataFrame:
    """One row per paragraph, preserving the parent document index."""
    rows = []
    for parent_idx, row in df.iterrows():
        for p_idx, para in enumerate(paragraphs(row[content_col])):
            rows.append({
                "parent_idx": parent_idx,
                "para_idx":   p_idx,
                "text":       para,
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- #
# Citation detection -- conservative: if any signal of citation is
# present, the pair is considered "explicit" and filtered out.
# -------------------------------------------------------------------- #
def case_cites_law(case_row: pd.Series, law_celex: str, law_name: str,
                   cites_col: str | None) -> bool:
    if cites_col and case_row.get(cites_col):
        blob = str(case_row[cites_col])
        if law_celex and law_celex in blob:
            return True
        if law_name and law_name in blob:
            return True
    # Fall back to scanning the case content directly
    body = str(case_row.get("__content__", ""))
    if law_celex and law_celex in body:
        return True
    if law_name and len(law_name) > 12 and law_name in body:
        return True
    return False


# -------------------------------------------------------------------- #
# Embedding
# -------------------------------------------------------------------- #
def embed(texts: list[str], model_name: str, batch: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"[embed] {len(texts):,} units")
    v = model.encode(texts, batch_size=batch, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)
    return v.astype(np.float32)


# -------------------------------------------------------------------- #
# Blocked top-pair extraction. For each law paragraph, find its best
# case-paragraph match; then aggregate up to (law, case) pairs.
# -------------------------------------------------------------------- #
def best_cross_pairs(
    law_vecs: np.ndarray, law_parent: np.ndarray,
    case_vecs: np.ndarray, case_parent: np.ndarray,
    block: int = 512,
) -> dict[tuple[int, int], tuple[float, int, int]]:
    """
    Returns {(law_parent, case_parent): (best_score, law_para_row, case_para_row)}
    keeping only the single strongest paragraph match per (law, case).
    """
    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    M = law_vecs.shape[0]
    for s in range(0, M, block):
        e = min(s + block, M)
        sims = law_vecs[s:e] @ case_vecs.T          # (block, |case_paras|)
        for li in range(sims.shape[0]):
            row = sims[li]
            ci = int(np.argmax(row))
            sc = float(row[ci])
            lp = int(law_parent[s + li])
            cp = int(case_parent[ci])
            key = (lp, cp)
            prev = best.get(key)
            if prev is None or sc > prev[0]:
                best[key] = (sc, s + li, ci)
        if (e % (block * 8) == 0) or e == M:
            print(f"  scored {e:,}/{M:,} law paragraphs")
    return best


# -------------------------------------------------------------------- #
# HTML report in the Digibeetle style
# -------------------------------------------------------------------- #
HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Hidden Semantic Connections – Digibeetle</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --ink:    #14181f;
  --sidebar:#1a1f2e;
  --paper:  #fdf8ec;
  --paper-2:#f5edd6;
  --rule:   #2a3142;
  --gold:   #f5c518;
  --gold-2: #e0a800;
  --muted:  #6b6f7a;
  --ok:     #2f7d4f;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--paper);color:var(--ink);
  font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:15px;line-height:1.55}
.page{display:grid;grid-template-columns:280px 1fr;min-height:100vh}
aside{background:var(--sidebar);color:#e8e9ef;padding:28px 22px;position:sticky;top:0;height:100vh;overflow-y:auto;border-right:3px solid var(--gold)}
aside h1{font-family:'Fraunces',Georgia,serif;font-weight:800;font-size:22px;letter-spacing:-.01em;margin:0 0 4px;color:#fff}
aside h1 .dot{color:var(--gold)}
aside .sub{color:#9aa1b3;font-size:12.5px;margin-bottom:28px;text-transform:uppercase;letter-spacing:.08em}
aside .group{font-size:11.5px;color:#7a8093;text-transform:uppercase;letter-spacing:.12em;margin:22px 0 10px;padding-bottom:6px;border-bottom:1px solid #2a3142}
aside ol{list-style:none;padding:0;margin:0;counter-reset:k}
aside ol li{counter-increment:k;margin:2px -22px;padding:9px 22px 9px 48px;position:relative;cursor:pointer;font-size:13.5px;color:#cfd3dd;border-left:3px solid transparent}
aside ol li::before{content:counter(k,decimal-leading-zero);position:absolute;left:22px;top:9px;color:#5a6075;font-family:'IBM Plex Mono',monospace;font-size:11.5px;letter-spacing:.05em}
aside ol li:hover{color:#fff;background:#222838}
aside ol li.active{color:var(--gold);background:#222838;border-left-color:var(--gold)}
main{padding:46px 64px 80px;max-width:1080px}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
h2.title{font-family:'Fraunces',Georgia,serif;font-weight:800;font-size:46px;line-height:1.05;letter-spacing:-.02em;margin:6px 0 8px}
h2.title em{font-style:normal;color:var(--gold-2)}
.kicker{color:var(--muted);max-width:62ch;margin-bottom:36px}
.callout{background:#fff;border:1px solid #e8dfc1;border-left:4px solid var(--gold);padding:18px 22px;border-radius:4px;margin:24px 0 40px;max-width:78ch}
.callout b{font-weight:600}
.pair{background:#fff;border:1px solid #e8dfc1;border-radius:6px;margin:0 0 26px;overflow:hidden;box-shadow:0 1px 0 rgba(20,24,31,.04)}
.pair header{display:flex;align-items:center;gap:14px;padding:14px 20px;background:linear-gradient(180deg,#fffaea,#fdf6dc);border-bottom:1px solid #ead9a0}
.rank{font-family:'Fraunces',Georgia,serif;font-weight:800;font-size:26px;color:var(--gold-2);min-width:34px}
.pair header h3{margin:0;font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:17px;flex:1}
.score{font-family:'IBM Plex Mono',monospace;font-size:13px;background:var(--ink);color:#fff;padding:5px 9px;border-radius:3px;letter-spacing:.04em}
.pair .body{display:grid;grid-template-columns:1fr 1fr;gap:0}
.side{padding:18px 22px}
.side+.side{border-left:1px dashed #d8caa0}
.side .tag{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
.side .law .tag{color:#7a5b00}
.side .case .tag{color:#1f4c2e}
.side .doc-name{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:15.5px;margin:0 0 10px;color:var(--ink)}
.side .doc-id{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--muted);margin-bottom:12px}
.side blockquote{margin:0;padding:12px 14px;background:#fbf5e3;border-left:3px solid var(--gold);font-size:13.8px;line-height:1.6;color:#2a2e38;border-radius:0 3px 3px 0}
footer.meta{margin-top:60px;padding-top:24px;border-top:1px solid #d8caa0;color:var(--muted);font-size:13px;font-family:'IBM Plex Mono',monospace}
.bar{height:3px;background:linear-gradient(90deg,var(--gold),transparent);margin:0 0 32px;border-radius:2px}
@media (max-width:920px){.page{grid-template-columns:1fr}aside{position:relative;height:auto}main{padding:32px 24px}.pair .body{grid-template-columns:1fr}.side+.side{border-left:0;border-top:1px dashed #d8caa0}}
</style></head><body>
<div class="page">
<aside>
  <h1>[•••]&nbsp;Digibeetle<span class="dot">.</span></h1>
  <div class="sub">Hidden Connections</div>
  <div class="group">Pairs</div>
  <ol id="nav">__NAV__</ol>
  <div class="group">About</div>
  <div style="color:#9aa1b3;font-size:12.5px;line-height:1.5">
    Each pair shows a law and a CJEU case discussing the same idea
    without citing one another. Scores are paragraph-level cosine
    similarities from MiniLM.
  </div>
</aside>
<main>
  <div class="eyebrow">Validation set · v1</div>
  <h2 class="title">Hidden Semantic <em>Connections</em></h2>
  <p class="kicker">Ten law–case pairs the citation graph misses. Each link is the strongest paragraph match between a law article and a case dispositif, after removing every pair where the case explicitly cites the law.</p>
  <div class="bar"></div>
  <div class="callout"><b>How to read this:</b> the left column is a law article (paragraph); the right column is a CJEU case dispositif (paragraph). The numeric score is cosine similarity in MiniLM embedding space — paragraphs above ~0.55 are typically about the same legal concept.</div>
__PAIRS__
  <footer class="meta">
    Model: __MODEL__ · __N__ pairs · scored from __NCASE__ case paragraphs × __NLAW__ law paragraphs
  </footer>
</main></div>
<script>
const items = document.querySelectorAll('#nav li');
items.forEach(li=>{li.addEventListener('click',()=>{
  const t = document.getElementById(li.dataset.target);
  if(t) t.scrollIntoView({behavior:'smooth', block:'start'});
  items.forEach(x=>x.classList.remove('active'));
  li.classList.add('active');
});});
</script>
</body></html>
"""


def render_html(pairs: list[dict], meta: dict) -> str:
    nav = "\n".join(
        f'<li data-target="p{i+1}">{p["law_name"][:46]}{"…" if len(p["law_name"])>46 else ""}'
        f' <span style="color:#7a8093">×</span> {p["case_name"][:30]}{"…" if len(p["case_name"])>30 else ""}</li>'
        for i, p in enumerate(pairs)
    )
    cards = []
    for i, p in enumerate(pairs):
        cards.append(f"""
<article class="pair" id="p{i+1}">
  <header>
    <div class="rank">{i+1:02d}</div>
    <h3>{p['law_name']} <span style="color:#9a9785">×</span> {p['case_name']}</h3>
    <div class="score">sim {p['score']:.3f}</div>
  </header>
  <div class="body">
    <div class="side law">
      <span class="tag">Law · Article</span>
      <p class="doc-name">{p['law_name']}</p>
      <p class="doc-id">{p['law_celex']}</p>
      <blockquote>{p['law_text']}</blockquote>
    </div>
    <div class="side case">
      <span class="tag">Case · Dispositif</span>
      <p class="doc-name">{p['case_name']}</p>
      <p class="doc-id">{p['case_id']}</p>
      <blockquote>{p['case_text']}</blockquote>
    </div>
  </div>
</article>""")
    html = HTML_TEMPLATE
    html = html.replace("__NAV__", nav)
    html = html.replace("__PAIRS__", "\n".join(cards))
    html = html.replace("__MODEL__", meta["model"])
    html = html.replace("__N__", str(len(pairs)))
    html = html.replace("__NCASE__", f"{meta['n_case_paras']:,}")
    html = html.replace("__NLAW__",  f"{meta['n_law_paras']:,}")
    return html


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laws",  type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--sep", default=";")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--output", type=Path, default=Path("hidden_connections"))
    args = ap.parse_args()

    laws  = pd.read_csv(args.laws,  sep=args.sep, dtype=str, keep_default_na=False)
    cases = pd.read_csv(args.cases, sep=args.sep, dtype=str, keep_default_na=False)
    for df in (laws, cases):
        df.columns = [c.strip() for c in df.columns]

    # --- standardise columns
    lid, lnm, lct = resolve(laws,  LAW_ID, required=False), resolve(laws,  LAW_NM), resolve(laws,  CONTENT)
    cid, cnm, cct = resolve(cases, CASE_ID, required=False), resolve(cases, CASE_NM), resolve(cases, CONTENT)
    cites_col = resolve(cases, CITES, required=False)

    laws = laws.rename(columns={lnm: "law_name", lct: "content",
                                **({lid: "celex"} if lid else {})})
    cases = cases.rename(columns={cnm: "case_name", cct: "content",
                                  **({cid: "case_id"} if cid else {})})
    if "celex"   not in laws.columns:  laws["celex"]   = laws["law_name"]
    if "case_id" not in cases.columns: cases["case_id"] = cases["case_name"]
    cases["__content__"] = cases["content"]

    # Aggregate to *document* level for laws (one row per law containing all article text)
    law_docs = (laws.groupby(["celex", "law_name"], sort=False)["content"]
                    .apply(lambda s: "\n\n".join(s)).reset_index())
    print(f"[load] {len(law_docs):,} laws  /  {len(cases):,} cases")

    # --- paragraph explosion
    law_para  = explode_paragraphs(law_docs, "content")
    case_para = explode_paragraphs(cases, "content")
    print(f"[split] {len(law_para):,} law paragraphs  /  {len(case_para):,} case paragraphs")

    # --- embed
    lv = embed(law_para["text"].tolist(),  args.model)
    cv = embed(case_para["text"].tolist(), args.model)

    # --- best paragraph pair per (law, case)
    best = best_cross_pairs(
        lv, law_para["parent_idx"].to_numpy(),
        cv, case_para["parent_idx"].to_numpy(),
    )
    print(f"[score] {len(best):,} (law, case) candidate pairs")

    # --- filter out explicit citations
    keep = []
    for (lp_idx, cp_idx), (sc, lpi, cpi) in best.items():
        if sc < args.min_score:
            continue
        law_row  = law_docs.iloc[lp_idx]
        case_row = cases.iloc[cp_idx]
        if case_cites_law(case_row, law_row["celex"], law_row["law_name"], cites_col):
            continue
        keep.append({
            "score":     round(sc, 4),
            "law_celex": law_row["celex"],
            "law_name":  law_row["law_name"],
            "law_text":  law_para.iloc[lpi]["text"],
            "case_id":   case_row["case_id"],
            "case_name": case_row["case_name"],
            "case_text": case_para.iloc[cpi]["text"],
        })

    keep.sort(key=lambda x: -x["score"])
    keep = keep[: args.top]
    print(f"[filter] kept top {len(keep)} after citation removal")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    html_path = args.output.with_suffix(".html")

    json_path.write_text(json.dumps({
        "meta": {"model": args.model, "n_law_paras": len(law_para),
                 "n_case_paras": len(case_para), "min_score": args.min_score},
        "pairs": keep,
    }, ensure_ascii=False, indent=2))

    html_path.write_text(render_html(keep, {
        "model": args.model,
        "n_law_paras": len(law_para),
        "n_case_paras": len(case_para),
    }))
    print(f"[done] {json_path}")
    print(f"[done] {html_path}")


if __name__ == "__main__":
    main()
