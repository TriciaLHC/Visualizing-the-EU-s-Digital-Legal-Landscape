"""
build_layout.py
===============
Reads explicit_network_data.json, computes a fast pre-seeded layout,
and writes explicit_network_layout.json with x,y baked into every node.

Strategy (runs in ~10–30 seconds):
  1. Place each law near its EuroVoc group centroid (groups arranged in a circle).
  2. Run spring layout using ONLY citation edges (~16k) starting from those
     seeded positions — converges much faster than a cold start.
  3. Cases placed on an outer ring (they only appear in View 2 anyway).

Run after build_explicit_network.py:
    cd "C:/ADS MASTER/THESIS PROJECT/deliverables"
    python build_layout.py

Output: explicit_network_layout.json
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import networkx as nx
import numpy as np

random.seed(42)
np.random.seed(42)

# ---- paths ----------------------------------------------------------------
BASE       = Path(__file__).parent
INPUT_JSON = BASE / "explicit_network_data.json"
OUT_JSON   = BASE / "explicit_network_layout.json"

# ---- constants ------------------------------------------------------------
CANVAS_W    = 4000
CANVAS_H    = 3200
GROUP_RING  = 1300   # radius of group centroid circle
NODE_JITTER = 200    # initial spread within a group
SPRING_K    = 0.08   # networkx k (fraction of canvas; larger = looser)
SPRING_ITER = 50     # iterations — seeded so fewer needed
CASE_RING   = 1500   # cases on outer ring

CITATION_TYPES = {"amends", "repeals", "implicitly_repeals", "body_citation"}


def seed_positions(law_nodes: list[dict], n_groups: int) -> dict[str, list[float]]:
    cx, cy = CANVAS_W / 2, CANVAS_H / 2
    centroids = [
        (cx + GROUP_RING * math.cos(2 * math.pi * i / n_groups),
         cy + GROUP_RING * math.sin(2 * math.pi * i / n_groups))
        for i in range(n_groups)
    ]
    pos = {}
    for n in law_nodes:
        gid = min(n.get("group", n_groups - 1), n_groups - 1)
        gcx, gcy = centroids[gid]
        angle = random.uniform(0, 2 * math.pi)
        r     = random.uniform(0, NODE_JITTER)
        pos[n["id"]] = [gcx + r * math.cos(angle), gcy + r * math.sin(angle)]
    return pos


def run_spring(law_nodes: list[dict], edges: list[dict],
               init: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    node_ids = {n["id"] for n in law_nodes}
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for e in edges:
        if e["type"] in CITATION_TYPES and e["source"] in node_ids and e["target"] in node_ids:
            G.add_edge(e["source"], e["target"])

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} citation edges")
    print(f"  Running spring layout ({SPRING_ITER} iterations, seeded)…")

    # Normalise seed to [-1,1] for networkx
    scale = max(CANVAS_W, CANVAS_H) / 2
    nx_init = {nid: [x / scale - 1, y / scale - 1] for nid, (x, y) in init.items()}

    pos = nx.spring_layout(G, pos=nx_init, k=SPRING_K, iterations=SPRING_ITER, seed=42)
    return pos


def rescale(pos: dict, pad: int = 150) -> dict[str, tuple[float, float]]:
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = CANVAS_W - 2 * pad
    h = CANVAS_H - 2 * pad
    out = {}
    for nid, (x, y) in pos.items():
        rx = pad + (x - x0) / (x1 - x0) * w if x1 != x0 else CANVAS_W / 2
        ry = pad + (y - y0) / (y1 - y0) * h if y1 != y0 else CANVAS_H / 2
        out[nid] = (round(rx, 1), round(ry, 1))
    return out


def place_cases(case_nodes: list[dict]) -> dict[str, tuple[float, float]]:
    cx, cy = CANVAS_W / 2, CANVAS_H / 2
    n = len(case_nodes)
    result = {}
    for i, node in enumerate(case_nodes):
        angle = 2 * math.pi * i / max(n, 1)
        r = CASE_RING + random.uniform(-80, 80)
        result[node["id"]] = (
            round(cx + r * math.cos(angle), 1),
            round(cy + r * math.sin(angle), 1),
        )
    return result


def main() -> None:
    print(f"Loading {INPUT_JSON} …")
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    law_nodes  = data["law_nodes"]
    case_nodes = data["case_nodes"]
    n_groups   = len(data["meta"]["groups"])

    print(f"  {len(law_nodes):,} laws  |  {len(case_nodes):,} cases")
    print(f"  {len(data['law_law_edges']):,} law-law edges  |  {len(data['case_law_edges']):,} case-law edges")
    print(f"  {n_groups} EuroVoc groups\n")

    # Law layout
    init_pos  = seed_positions(law_nodes, n_groups)
    spring_pos = run_spring(law_nodes, data["law_law_edges"], init_pos)
    law_pos   = rescale(spring_pos)

    # Case layout (outer ring)
    case_pos = place_cases(case_nodes)

    # Inject into nodes
    for n in law_nodes:
        n["x"], n["y"] = law_pos.get(n["id"], (CANVAS_W / 2, CANVAS_H / 2))
    for n in case_nodes:
        n["x"], n["y"] = case_pos.get(n["id"], (CANVAS_W / 2, CANVAS_H / 2))

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    mb = os.path.getsize(OUT_JSON) / 1e6
    print(f"\n[done] {OUT_JSON}  ({mb:.1f} MB)")
    print("Serve with:  python -m http.server 8080")
    print("Open:        http://localhost:8080/network_static.html")


if __name__ == "__main__":
    main()
