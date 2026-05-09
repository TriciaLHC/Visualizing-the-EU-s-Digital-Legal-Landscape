# Visualizing the EU's Digital Legal Landscape

An interactive semantic search tool for EU legislation and case law, combining BM25 keyword search with BGE-M3 dense retrieval.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/Visualizing-the-EU-s-Digital-Legal-Landscape.git
cd Visualizing-the-EU-s-Digital-Legal-Landscape
```

### 2. Download data files

Download the `dataset/` and `corpus/` folders from Google Drive and place them in the repo root:

```
Visualizing-the-EU-s-Digital-Legal-Landscape/
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
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Recommended: use a virtual environment first
> ```bash
> python -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

### 4. Run the server

```bash
python server.py
```

### 5. Open in browser
Go to [http://localhost:5001/](http://localhost:5001/)

## How it works

- **Bronze / Silver / Gold** results come from BM25 keyword matching in the browser
- **Platinum** results come from the BGE-M3 semantic model running on the local server — these are documents that are conceptually relevant even without keyword overlap
- Click any result to view its full text, with the most relevant passage highlighted
