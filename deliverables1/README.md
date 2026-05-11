# Digibeetle — Law Similarity Network

Two deliverables for the meeting:

1. **`network.html`** — interactive law × law network. Force-directed
   graph where edges are semantic similarities between laws. Drag,
   zoom, search, filter, click a node to see its articles and its
   most-similar neighbours.

2. **`hidden_connections.html`** — the validation set. Ten law/case
   pairs that *don't* cite each other but discuss the same problem.
   Each pair shows the strongest matching paragraph from each side
   side-by-side. (e.g. Meta × DMA echoing each other.)

Everything else is the pipeline that produces these.

---

## Files

```
build_similarity_network.py   pipeline → network_data.json
find_hidden_connections.py    pipeline → hidden_connections.{json,html}
network.html                  reads network_data.json and renders the graph
demo/network_data.json        47-law demo so you can preview the UI immediately
demo/articles_demo.csv        the synthetic articles behind that demo
```

## Quick preview (no waiting)

To see the visualization style + interactivity right now, before
running anything on your real corpus:

```bash
cd demo
python -m http.server 8080
# open http://localhost:8080/../network.html
```

(Or just copy `demo/network_data.json` next to `network.html` and
open the HTML directly via a local server — `file://` won't work
because of the fetch.)

## Full pipeline on your real corpus

### 1. Build the network

```bash
python build_similarity_network.py \
    --laws articles_clean.csv \
    --output network_data.json \
    --sep ";" \
    --top-k-articles 3 \
    --top-k-edges 6 \
    --min-similarity 0.45
```

- **`--top-k-articles N`**: how the law-to-law similarity is pooled.
  For each pair of laws, the score is the mean of the top-N strongest
  article-pair similarities. `3` is a good default — captures specific
  overlap, ignores diluting noise.
- **`--top-k-edges N`**: cap edges shown per node. Lower = clearer
  graph, higher = denser.
- **`--min-similarity X`**: hard floor below which edges are dropped
  regardless of top-k. `0.45` is a sensible starting point for
  MiniLM; the HTML lets you change this live with a slider too.

The embedding step is slow (~10 min on CPU for 25k articles). It's
cached in `.embeddings_cache.npz`, so re-runs are seconds.

The output is a single self-contained JSON (~10 MB for full corpus)
that `network.html` consumes directly.

### 2. Find hidden law/case connections

```bash
python find_hidden_connections.py \
    --laws articles_clean.csv \
    --cases cases_clean.csv \
    --top 10 \
    --min-score 0.55 \
    --output hidden_connections
```

Produces `hidden_connections.html` (styled report) and
`hidden_connections.json` (the same data, for inspection).

Citation filter is conservative: a pair is excluded if the case's
`cites` column mentions the law's CELEX or name, **or** if the case's
text contains them as a substring. So results are reliably *implicit*.

---

## Why this design

**Why pool over top-k article pairs instead of mean-pooling articles
into one law vector?** A law is a bundle of often-unrelated articles.
Mean-pooling washes out the signal we actually want: "GDPR Art. 22 and
DSA Art. 27 talk about the same thing, even though the rest of those
statutes are unrelated." Top-k pooling preserves that local overlap.

**Why paragraph-level for hidden connections but article-level for the
network?** The network is a global view — too many edges hurt
readability, so we aggregate. The hidden-connections task is
explicitly about surfacing the single sharpest paragraph pair, so we
do not aggregate.

**Why the cream/yellow/dark visual aesthetic?** Modelled on the
Digibeetle dataset guide — academic, calm, readable, with the gold
accent doing the work of marking interaction targets and important
data.

## CSV column expectations

The pipeline accepts several reasonable column names. Minimum
required:

- **Laws CSV**: `law_name` (or `title`) and `content` (or `text` /
  `article_text`). Strongly recommended: `celex` (or `law_id`) and
  `article` (article number).
- **Cases CSV**: `case_name` (or `title`) and `content` (or
  `operative_text` / `dispositif`). Strongly recommended: `case_id`
  (or `ecli`) and a `cites` (or `law_match_name_for_references`)
  column listing the CELEX numbers of laws the case explicitly cites.

If `cites` is missing the script falls back to substring search of
the case body.

## Tuning knobs in the UI

The sidebar lets you change `min similarity` and `edges per node`
live without rebuilding. Useful for the meeting demo:

- Drag `min similarity` up to ~0.7 — only the strongest connections
  remain, the global topical clusters become obvious.
- Drag it down to ~0.4 — the long tail of weaker but interesting
  cross-cluster links appears (these are the "hidden connections"
  hint that practitioners care about).
