


```
http://localhost:5001/
```

> To stop: press `Ctrl + C` in the terminal.

--

### Prerequisites
- Python 3.10 or 3.11 (not 3.12 — FlagEmbedding may not support it)
- ~8 GB free RAM (BGE-M3 model for View 3)

  - `corpus/` folder (embeddings)
  - `Data/` folder (laws.csv, cases.csv)
  - `explicit_network_data.json`
  - `network.html`, `server.py`
  - The two name CSVs one level up (`260512-laws-celex-and-human-readable-names.csv`, `260514091724-case-names-by-db.csv`)

### Install dependencies (first time only)

```bash

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Run

```bash
.venv\Scripts\python.exe server.py --html network.html
```

Then open **http://localhost:5001/** in any modern browser (Chrome, Edge, Firefox).

> **Views 1 and 2 only** (no BGE-M3 needed): open `network.html` directly via a local server:
> ```bash
> .venv\Scripts\python.exe -m http.server 8080
> ```
> Then go to **http://localhost:8080/network.html**

---

## Controls

| Control | What it does |
|---|---|
| **+  −  ⊙** buttons (bottom-right) | Zoom in, zoom out, reset view |
| Scroll wheel / trackpad pinch | Zoom |
| Click + drag canvas | Pan |
| Click + drag a node | Move node |
| Click a node | Open detail panel (info, connections, full text) |
| **Min connections** slider | Hide nodes with fewer than N connections |
| **Similarity threshold** slider (View 3) | Low ≥ 0.35 · Medium ≥ 0.60 · High ≥ 0.80 |
| Search box | Filter by law name, CELEX, EuroVoc tag, or case number |
| Semantic search box (View 3, press Enter) | Find laws/cases by meaning, not keyword |

---

## Rebuilding the network data (optional)

If you need to regenerate `explicit_network_data.json` from the raw CSVs:

```bash
.venv\Scripts\python.exe build_explicit_network.py
```

This reads `Data/laws.csv` and `Data/cases.csv`, filters to **inclusion tier 1** (422 laws), and writes a fresh JSON file (~1.6 MB).  
The server can then be started normally.

---
