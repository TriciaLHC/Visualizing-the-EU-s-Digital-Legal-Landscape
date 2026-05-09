"""
Preprocess the laws and cases tables for operative-content-only embedding.

What this does:
  - For each law: extracts every article (level-6 markdown headings:
    `###### Article N — Title`) into a single concatenated `content`
    column. Drops the preamble and recitals.
  - For each case: extracts only the operative part — the section
    starting with "On those grounds, the Court hereby rules:"
    (or close variants) — into the `content` column. Drops the facts
    and the reasoning.
  - Optionally also extracts the reasoning section separately for
    laws/cases where you may later want to embed it (off by default).

Output files (written next to the input):
  - laws_articles.csv     same columns as laws.csv + new `content` column
  - cases_operative.csv   same columns as cases.csv + new `content` column
  - extraction_report.csv per-document audit: how many articles found,
                          whether operative part was found, content length

Run:
  python preprocess.py --laws ../Data/laws.csv --cases ../Data/cases.csv
                       --out-dir ../Data/processed
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Loaders (same tolerant logic as pipeline.py)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    """Tolerant CSV reader for messy markdown-cell exports."""
    import csv as _csv
    _csv.field_size_limit(min(2**31 - 1, sys.maxsize))

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    return pd.read_csv(
        path, sep=sep, engine="python",
        quotechar='"', on_bad_lines="skip",
        dtype=str,
    )


# ---------------------------------------------------------------------------
# Article extraction (laws)
# ---------------------------------------------------------------------------

# Articles are level-6 markdown headings: `###### Article 6` or
# `###### Article 6 — Lawfulness of processing`. The split is on the
# heading marker itself.
ARTICLE_HEADING_RE = re.compile(r"^######\s+(.+?)$", re.MULTILINE)


def extract_articles(md: str) -> tuple[str, int]:
    """Extract concatenated article content from a law's markdown.

    Returns (content, n_articles). Content is the article bodies
    concatenated with double-newlines between them, with each article's
    heading preserved as a marker.
    """
    if not md or not isinstance(md, str):
        return "", 0

    # Find article heading positions
    matches = list(ARTICLE_HEADING_RE.finditer(md))
    if not matches:
        return "", 0

    parts: list[str] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        if body:
            # Keep the heading inline so the embedder sees "Article 6 — ..."
            # at the top of each chunk
            parts.append(f"###### {heading}\n{body}")

    return "\n\n".join(parts), len(parts)


def get_law_body(row: pd.Series) -> str:
    """Pull the body markdown for one law. Use original (not consolidated)."""
    parts = []
    for c in ("md_content_1",
              "md_content_2_overflow_column",
              "md_content_3_overflow_column"):
        v = row.get(c)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Operative-part extraction (cases)
# ---------------------------------------------------------------------------

# Variants of the operative-part marker observed in CJEU judgments.
# Ordered most-specific first.
OPERATIVE_MARKERS = [
    r"On\s+those\s+grounds,?\s+the\s+(?:Court|General\s+Court)[^.\n]*?(?:hereby|rules?|orders?|declares?)",
    r"On\s+those\s+grounds,?\s+THE\s+(?:COURT|GENERAL\s+COURT)",
    # Some orders use slightly different phrasing
    r"For\s+those\s+reasons,?\s+the\s+(?:Court|General\s+Court)",
    # Older cases / French-translated cases
    r"hereby\s+(?:rules|orders|declares)\s*:",
]
OPERATIVE_RE = re.compile(
    "(?:" + "|".join(OPERATIVE_MARKERS) + ")",
    re.IGNORECASE,
)


def extract_operative(md: str) -> tuple[str, bool]:
    """Extract the operative part from a judgment's markdown.

    Returns (content, found). If no operative marker is found, returns
    ("", False) — caller can decide whether to fall back to full text or
    skip.
    """
    if not md or not isinstance(md, str):
        return "", False

    m = OPERATIVE_RE.search(md)
    if not m:
        return "", False

    # Take from the start of the matched marker to the end of the document.
    # We include the "On those grounds, the Court hereby rules:" line itself
    # because it's part of the operative section.
    operative = md[m.start():].strip()

    # Some judgments have post-script signatures we may want to trim
    # (Registrar / President names, dates). Cut at common signature markers.
    sig_markers = [
        r"\n\s*\[Signatures?\]",
        r"\n\s*Signed\.?\s*\n",
        r"\n\s*Registrar\s*\n",
    ]
    for sm in sig_markers:
        sm_match = re.search(sm, operative, re.IGNORECASE)
        if sm_match:
            operative = operative[:sm_match.start()].rstrip()
            break

    return operative, True


def get_case_body(row: pd.Series) -> str:
    """Pull the markdown for one case."""
    for col in ("Markdown Content [by_DB]", "Markdown Content"):
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            return str(v)
    return ""


# ---------------------------------------------------------------------------
# Optional: reasoning extraction (cases)
# ---------------------------------------------------------------------------

# The reasoning ("Consideration of the questions referred") sits BETWEEN
# the facts and the operative part. We extract it as: from the start of
# a heading like "Consideration of the questions referred" / "The Court's
# assessment" / "Substance" up to the operative marker.
REASONING_START_RE = re.compile(
    r"^(?:#{1,6}\s+)?(?:Consideration\s+of\s+the\s+questions?\s+referred"
    r"|The\s+Court'?s?\s+assessment"
    r"|Findings\s+of\s+the\s+Court"
    r"|Substance"
    r"|Assessment\s+of\s+the\s+Court)",
    re.MULTILINE | re.IGNORECASE,
)


def extract_reasoning(md: str) -> tuple[str, bool]:
    """Extract the reasoning section. Best-effort — heading conventions
    vary. Returns (content, found)."""
    if not md or not isinstance(md, str):
        return "", False

    start_m = REASONING_START_RE.search(md)
    if not start_m:
        return "", False

    end_m = OPERATIVE_RE.search(md, pos=start_m.end())
    end = end_m.start() if end_m else len(md)
    reasoning = md[start_m.start():end].strip()
    return reasoning, True


# ---------------------------------------------------------------------------
# Top-level processing
# ---------------------------------------------------------------------------

def process_laws(laws_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a `content` column with concatenated articles. Also returns a
    per-row audit dataframe."""
    contents: list[str] = []
    n_articles_list: list[int] = []
    body_lengths: list[int] = []

    for _, row in laws_df.iterrows():
        body = get_law_body(row)
        content, n = extract_articles(body)
        contents.append(content)
        n_articles_list.append(n)
        body_lengths.append(len(body))

    out = laws_df.copy()
    out["content"] = contents

    audit = pd.DataFrame({
        "celex": laws_df.get("celex"),
        "title_short": laws_df.get("title_short"),
        "body_chars": body_lengths,
        "n_articles_extracted": n_articles_list,
        "content_chars": [len(c) for c in contents],
    })
    return out, audit


def process_cases(cases_df: pd.DataFrame, *,
                  fallback_to_full: bool = False
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a `content` column with the operative part. If fallback_to_full
    is True, cases without a detected operative marker get their full text;
    otherwise they get an empty content (and won't be embedded)."""
    contents: list[str] = []
    found_operative_list: list[bool] = []
    found_reasoning_list: list[bool] = []
    body_lengths: list[int] = []

    case_no_col = next(
        (c for c in cases_df.columns if "Case Number" in c), None)

    for _, row in cases_df.iterrows():
        body = get_case_body(row)
        operative, found_op = extract_operative(body)
        _, found_reason = extract_reasoning(body)

        if found_op:
            content = operative
        elif fallback_to_full and body:
            content = body
        else:
            content = ""

        contents.append(content)
        found_operative_list.append(found_op)
        found_reasoning_list.append(found_reason)
        body_lengths.append(len(body))

    out = cases_df.copy()
    out["content"] = contents

    audit = pd.DataFrame({
        "case_number": cases_df.get(case_no_col) if case_no_col else None,
        "body_chars": body_lengths,
        "operative_found": found_operative_list,
        "reasoning_found": found_reasoning_list,
        "content_chars": [len(c) for c in contents],
    })
    return out, audit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--laws", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--out-dir", required=True,
                   help="Where to write laws_articles.csv, cases_operative.csv")
    p.add_argument("--fallback-full-text", action="store_true",
                   help="For cases without an operative marker, include the "
                        "full text instead of leaving content empty")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] laws  ← {args.laws}")
    laws_df = _read_csv(Path(args.laws))
    print(f"[load] cases ← {args.cases}")
    cases_df = _read_csv(Path(args.cases))
    print(f"[load] {len(laws_df):,} laws, {len(cases_df):,} cases")

    print("[process] extracting articles from laws...")
    laws_out, laws_audit = process_laws(laws_df)
    print("[process] extracting operative parts from cases...")
    cases_out, cases_audit = process_cases(
        cases_df, fallback_to_full=args.fallback_full_text)

    # Reports
    print()
    print("=== LAWS extraction summary ===")
    print(f"Total laws:                    {len(laws_audit):>6,}")
    print(f"With body markdown:            {(laws_audit['body_chars'] > 0).sum():>6,}")
    print(f"Articles extracted (any):      {(laws_audit['n_articles_extracted'] > 0).sum():>6,}")
    print(f"Total articles across corpus:  {laws_audit['n_articles_extracted'].sum():>6,}")
    print(f"Mean articles per law:         {laws_audit['n_articles_extracted'].mean():.1f}")
    print(f"Mean content chars per law:    {laws_audit['content_chars'].mean():,.0f}")

    print()
    print("=== CASES extraction summary ===")
    print(f"Total cases:                   {len(cases_audit):>6,}")
    print(f"With body markdown:            {(cases_audit['body_chars'] > 0).sum():>6,}")
    print(f"Operative part found:          {cases_audit['operative_found'].sum():>6,}")
    print(f"Reasoning section found:       {cases_audit['reasoning_found'].sum():>6,}")
    print(f"Mean content chars per case:   {cases_audit['content_chars'].mean():,.0f}")

    # Write
    laws_path = out_dir / "laws_articles.csv"
    cases_path = out_dir / "cases_operative.csv"
    audit_path = out_dir / "extraction_report.csv"

    laws_out.to_csv(laws_path, sep=";", index=False)
    cases_out.to_csv(cases_path, sep=";", index=False)
    pd.concat([
        laws_audit.assign(_kind="law"),
        cases_audit.assign(_kind="case"),
    ], ignore_index=True).to_csv(audit_path, sep=";", index=False)

    print()
    print(f"[write] {laws_path}")
    print(f"[write] {cases_path}")
    print(f"[write] {audit_path}")
    print()
    print("Spot-check a few rows in each output before embedding. "
          "If extraction looks right, point run.py at these new files "
          "with a small change to the chunker (see README).")


if __name__ == "__main__":
    main()
