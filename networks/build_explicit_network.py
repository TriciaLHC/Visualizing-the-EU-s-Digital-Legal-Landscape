##HERE:
#cd "C:\ADS MASTER\THESIS PROJECT\Final\networks"
#python build_explicit_network.py   # → explicit_network_data.json
#python build_layout.py             # → explicit_network_layout.json  (only needed for network_static.html)
#python -m http.server 8080


"""
build_explicit_network.py
=========================
Builds the explicit EU law and case-law network.

Source files:
  Data/laws.csv            — legislation
  Data/cases.csv           — court cases

Law × Law edge types (by column used):
  amends               → amends column
  repeals              → repeals + implicitly_repeals columns
  implements           → implements column
  completes            → completes column
  corrects             → corrects column
  body_citation        → outgoing/incoming_references_to_other_laws_based_on_..._body_text
  shared_subject_matter → shared labels in subject_matter_labels [by DB]
  shared_eurovoc       → shared labels in eurovoc_labels [by DB]

Case → Law edge types:
  reasoning  → Outgoing Legislation Refs in Reasoning [citationsMotif]
  operative  → Outgoing Legislation Refs in Operative Part [citationsDispositif]

Node colours: top-19 most frequent EuroVoc labels → group IDs, rest → "Other"

Output: deliverables/explicit_network_data.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

csv.field_size_limit(10 ** 7)

# -------------------------------------------------------------------- #
# Columns
# -------------------------------------------------------------------- #

LAWS_COLS = [
    "celex",
    "title_en",
    "title_short",
    "subject_matter_labels [by DB]",
    "eurovoc_labels [by DB]",
    "directory_labels [by DB]",
    "amends",
    "repeals",
    "implicitly_repeals",
    "implements",
    "completes",
    "corrects",
    "outgoing_references_to_other_laws_based_on_original_or_consolidated_body_text",
    "incoming_references_by_other_laws_based_on_body_text",
    "incoming_references_by_case_law_in_the_cases_reasoning",
    "incoming_references_by_case_law_in_the_cases_operative_part",
    "inclusion_tier",
]

CASES_COLS = [
    "Case Number [publishedId]",
    "CELEX [celex]",
    "Subject Matter [matCodeML]",
    "Descriptor [descriptorML]",
    "Outgoing Legislation Refs in Reasoning [citationsMotif]",
    "Outgoing Legislation Refs in Operative Part [citationsDispositif]",
]

GENEALOGY = {
    "amends":             "amends",
    "repeals":            "repeals",
    "implicitly_repeals": "implicitly_repeals",
    "implements":         "implements",
    "completes":          "completes",
    "corrects":           "corrects",
}

N_GROUPS             = 20    # top EuroVoc groups + "Other"
MAX_TOPIC_PER_LAW    = 15   # cap topic edges per law node
MAX_LABEL_SIZE       = 500  # skip labels shared by >500 laws (too generic)


# -------------------------------------------------------------------- #
# Parsing helpers
# -------------------------------------------------------------------- #

def split_pipe(text: str) -> list[str]:
    return [p.strip() for p in str(text).split("|") if p.strip()]


def parse_genealogy(text: str, valid: set[str]) -> list[str]:
    """Parse pipe-separated CELEX list (genealogy columns)."""
    result = []
    for part in split_pipe(text):
        token = re.split(r'[\s\(\[]', part)[0]
        token = re.sub(r'\(\d+\)$', '', token)  # strip (01) suffixes
        if token in valid:
            result.append(token)
    return result


def parse_body_refs_outgoing(text: str, valid: set[str]) -> list[str]:
    """Parse consolidated body-text refs: '0YYYY...-date (name) [from: ...]'."""
    result = []
    for part in split_pipe(text):
        token = re.split(r'[\s\(\[]', part)[0]
        for candidate in [token, token.split('-')[0]]:
            if candidate in valid:
                result.append(candidate)
                break
            if candidate.startswith('0'):
                norm = '3' + candidate[1:]
                if norm in valid:
                    result.append(norm)
                    break
    return result


def parse_body_refs_incoming(text: str, valid: set[str]) -> list[str]:
    """Parse incoming body-text refs: '3YYYY... [from: ...]'."""
    result = []
    for part in split_pipe(text):
        token = re.split(r'[\s\(\[]', part)[0]
        if token in valid:
            result.append(token)
    return result


def parse_case_refs(text: str, valid: set[str]) -> list[str]:
    """Parse case citation refs: 'CELEX: N XX | CELEX2: F'."""
    result = []
    for part in split_pipe(text):
        celex = part.split(':')[0].strip() if ':' in part else part
        if celex in valid:
            result.append(celex)
    return result


# -------------------------------------------------------------------- #
# Loading
# -------------------------------------------------------------------- #

def load_laws(path: Path, tier: int | None = 1) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False,
                     encoding="utf-8-sig", engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in LAWS_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[load_laws] missing columns: {missing}")
    df = df[LAWS_COLS].copy().reset_index(drop=True)
    if tier is not None:
        before = len(df)
        df = df[df["inclusion_tier"].str.strip() == str(tier)].reset_index(drop=True)
        print(f"[load_laws] {len(df):,} tier-{tier} laws  (filtered from {before:,})")
    else:
        print(f"[load_laws] {len(df):,} laws")
    return df


def load_cases(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False,
                     encoding="utf-8-sig", engine="python")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in CASES_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[load_cases] missing columns: {missing}")
    df = df[CASES_COLS].copy().reset_index(drop=True)
    df = df[df["CELEX [celex]"].str.strip() != ""].reset_index(drop=True)
    print(f"[load_cases] {len(df):,} cases")
    return df


# -------------------------------------------------------------------- #
# Group assignment (EuroVoc dominant label → node colour)
# -------------------------------------------------------------------- #

def assign_groups(laws_df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    label_count: dict[str, int] = defaultdict(int)
    celex_labels: dict[str, list[str]] = {}

    for _, row in laws_df.iterrows():
        labels = [l for l in split_pipe(row["eurovoc_labels [by DB]"]) if l]
        if not labels:
            labels = [l for l in split_pipe(row["subject_matter_labels [by DB]"]) if l]
        celex_labels[row["celex"]] = labels
        for l in labels:
            label_count[l] += 1

    top = sorted(label_count, key=label_count.__getitem__, reverse=True)[: N_GROUPS - 1]
    top_set = set(top)
    group_names = top + ["Other"]
    label_to_gid = {l: i for i, l in enumerate(top)}

    celex_to_gid: dict[str, int] = {}
    for celex, labels in celex_labels.items():
        gid = len(group_names) - 1
        for l in labels:
            if l in top_set:
                gid = label_to_gid[l]
                break
        celex_to_gid[celex] = gid

    return group_names, celex_to_gid


# -------------------------------------------------------------------- #
# Law × Law edges
# -------------------------------------------------------------------- #

def build_law_law_edges(laws_df: pd.DataFrame) -> list[dict]:
    valid = set(laws_df["celex"])
    edges: list[dict] = []
    seen: set[tuple] = set()

    def add(src: str, tgt: str, etype: str, shared: int | None = None) -> None:
        if src == tgt or src not in valid or tgt not in valid:
            return
        key = (min(src, tgt), max(src, tgt), etype)
        if key in seen:
            return
        seen.add(key)
        e: dict = {"source": src, "target": tgt, "type": etype}
        if shared is not None:
            e["shared_count"] = shared
        edges.append(e)

    # Genealogy
    for _, row in laws_df.iterrows():
        src = row["celex"]
        for col, etype in GENEALOGY.items():
            for tgt in parse_genealogy(row[col], valid):
                add(src, tgt, etype)
    print(f"[edges] genealogy: {len(edges)}")

    # Body text citations
    n0 = len(edges)
    for _, row in laws_df.iterrows():
        src = row["celex"]
        for tgt in parse_body_refs_outgoing(
            row["outgoing_references_to_other_laws_based_on_original_or_consolidated_body_text"],
            valid,
        ):
            add(src, tgt, "body_citation")
        for citing in parse_body_refs_incoming(
            row["incoming_references_by_other_laws_based_on_body_text"],
            valid,
        ):
            add(citing, src, "body_citation")
    print(f"[edges] body citations: +{len(edges)-n0}  total={len(edges)}")

    # Topic: shared subject matter
    n0 = len(edges)
    sm_idx: dict[str, list[str]] = defaultdict(list)
    for _, row in laws_df.iterrows():
        for label in split_pipe(row["subject_matter_labels [by DB]"]):
            if label:
                sm_idx[label].append(row["celex"])

    shared_sm: dict[tuple, int] = defaultdict(int)
    for label, celexes in sm_idx.items():
        if len(celexes) > MAX_LABEL_SIZE:
            continue
        for i in range(len(celexes)):
            for j in range(i + 1, len(celexes)):
                shared_sm[(min(celexes[i], celexes[j]), max(celexes[i], celexes[j]))] += 1

    # Per-law: keep top MAX_TOPIC_PER_LAW by shared count
    per_law: dict[str, list] = defaultdict(list)
    for (a, b), cnt in shared_sm.items():
        per_law[a].append((cnt, b))
        per_law[b].append((cnt, a))
    for celex, cands in per_law.items():
        for cnt, other in sorted(cands, reverse=True)[:MAX_TOPIC_PER_LAW]:
            add(celex, other, "shared_subject_matter", shared=cnt)
    print(f"[edges] shared subject matter: +{len(edges)-n0}  total={len(edges)}")

    # Topic: shared EuroVoc
    n0 = len(edges)
    ev_idx: dict[str, list[str]] = defaultdict(list)
    for _, row in laws_df.iterrows():
        for label in split_pipe(row["eurovoc_labels [by DB]"]):
            if label:
                ev_idx[label].append(row["celex"])

    shared_ev: dict[tuple, int] = defaultdict(int)
    for label, celexes in ev_idx.items():
        if len(celexes) > MAX_LABEL_SIZE:
            continue
        for i in range(len(celexes)):
            for j in range(i + 1, len(celexes)):
                shared_ev[(min(celexes[i], celexes[j]), max(celexes[i], celexes[j]))] += 1

    per_law_ev: dict[str, list] = defaultdict(list)
    for (a, b), cnt in shared_ev.items():
        per_law_ev[a].append((cnt, b))
        per_law_ev[b].append((cnt, a))
    for celex, cands in per_law_ev.items():
        for cnt, other in sorted(cands, reverse=True)[:MAX_TOPIC_PER_LAW]:
            add(celex, other, "shared_eurovoc", shared=cnt)
    print(f"[edges] shared EuroVoc: +{len(edges)-n0}  total={len(edges)}")

    return edges


# -------------------------------------------------------------------- #
# Case × Law edges
# -------------------------------------------------------------------- #

def build_case_law_edges(cases_df: pd.DataFrame, valid_laws: set[str]) -> list[dict]:
    edges: list[dict] = []
    seen: set[tuple] = set()

    for _, row in cases_df.iterrows():
        cid = row["CELEX [celex]"].strip()
        if not cid:
            continue
        for law in parse_case_refs(
            row["Outgoing Legislation Refs in Reasoning [citationsMotif]"], valid_laws
        ):
            key = (cid, law, "reasoning")
            if key not in seen:
                seen.add(key)
                edges.append({"source": cid, "target": law, "type": "reasoning"})
        for law in parse_case_refs(
            row["Outgoing Legislation Refs in Operative Part [citationsDispositif]"], valid_laws
        ):
            key = (cid, law, "operative")
            if key not in seen:
                seen.add(key)
                edges.append({"source": cid, "target": law, "type": "operative"})

    print(f"[edges] case-law: {len(edges)}")
    return edges


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #

def main() -> None:
    ROOT     = Path(__file__).resolve().parent.parent
    DATA     = ROOT / "Data"
    NETWORKS = Path(__file__).resolve().parent   # same folder as this script

    ap = argparse.ArgumentParser()
    ap.add_argument("--laws",   type=Path, default=DATA / "laws.csv")
    ap.add_argument("--cases",  type=Path, default=DATA / "cases.csv")
    ap.add_argument("--output", type=Path, default=NETWORKS / "explicit_network_data.json")
    ap.add_argument("--tier",   type=int,  default=0,
                    help="Only include laws with this inclusion_tier. Pass 0 (default) for all.")
    args = ap.parse_args()

    laws_df  = load_laws(args.laws, tier=args.tier if args.tier > 0 else None)
    cases_df = load_cases(args.cases)

    group_names, celex_to_gid = assign_groups(laws_df)
    valid_laws = set(laws_df["celex"])

    # Law nodes
    law_nodes = []
    for _, row in laws_df.iterrows():
        label = row["title_short"].strip() or row["celex"]
        name  = row["title_en"].strip() or label
        law_nodes.append({
            "id":             row["celex"],
            "label":          label,
            "name":           name,
            "group":          celex_to_gid.get(row["celex"], N_GROUPS - 1),
            "subject_matter": [l for l in split_pipe(row["subject_matter_labels [by DB]"]) if l][:6],
            "eurovoc":        [l for l in split_pipe(row["eurovoc_labels [by DB]"]) if l][:10],
            "directory":      [l for l in split_pipe(row["directory_labels [by DB]"]) if l][:5],
        })

    # Case nodes (deduplicated)
    seen_cases: set[str] = set()
    case_nodes = []
    for _, row in cases_df.iterrows():
        cid = row["CELEX [celex]"].strip()
        if not cid or cid in seen_cases:
            continue
        seen_cases.add(cid)
        case_nodes.append({
            "id":             cid,
            "label":          row["Case Number [publishedId]"].strip() or cid,
            "name":           row["Case Number [publishedId]"].strip() or cid,
            "subject_matter": row["Subject Matter [matCodeML]"].strip(),
            "descriptor":     row["Descriptor [descriptorML]"].strip(),
        })

    law_law_edges  = build_law_law_edges(laws_df)
    case_law_edges = build_case_law_edges(cases_df, valid_laws)

    # Only keep case nodes that appear in at least one case-law edge
    active_cases = {e["source"] for e in case_law_edges}
    case_nodes   = [n for n in case_nodes if n["id"] in active_cases]

    payload = {
        "meta": {
            "n_laws":           len(law_nodes),
            "n_cases":          len(case_nodes),
            "n_edges_law_law":  len(law_law_edges),
            "n_edges_case_law": len(case_law_edges),
            "groups":           group_names,
        },
        "law_nodes":      law_nodes,
        "case_nodes":     case_nodes,
        "law_law_edges":  law_law_edges,
        "case_law_edges": case_law_edges,
    }

    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    mb = os.path.getsize(args.output) / 1e6
    print(f"\n[done] {args.output}  ({mb:.1f} MB)")
    print(f"  {len(law_nodes):,} laws  |  {len(case_nodes):,} cases")
    print(f"  {len(law_law_edges):,} law-law edges  |  {len(case_law_edges):,} case-law edges")
    print(f"  {len(group_names)} groups: {group_names[:5]} ...")


if __name__ == "__main__":
    main()
