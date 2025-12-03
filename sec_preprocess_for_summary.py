"""Utility to clean SEC filing sections for summarization.

The script reads *_sections.json files from an input directory, cleans and
reduces Item 1/7/2 content, and writes *_clean.json files for downstream
summarization steps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TAG_PATTERN = re.compile(r"<[^>]+>")
NUMERIC_SYMBOLS = set("0123456789%$,().-+")
HEADER_PATTERNS = [
    "table of contents",
    "form 10-k",
    "form 10-q",
    "united states securities and exchange commission",
    "securities and exchange commission",
]

SECTION_BUDGETS: Dict[str, int] = {
    "item1": 12_000,
    "item7": 16_000,
    "item2": 10_000,
}


def clean_raw_text(text: str) -> str:
    """Remove HTML-ish tags, normalize whitespace, and drop junk lines."""

    no_tags = TAG_PATTERN.sub(" ", text)
    normalized_newlines = no_tags.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: List[str] = []
    for raw_line in normalized_newlines.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if _is_junk_line(line):
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def _is_junk_line(line: str) -> bool:
    short_and_not_caps = len(line) < 20 and not line.isupper()
    lower_line = line.lower()
    has_header = any(pattern in lower_line for pattern in HEADER_PATTERNS)
    return short_and_not_caps or has_header


def split_into_paragraphs(text: str, min_length: int = 80, numeric_ratio: float = 0.4) -> List[str]:
    """Split text into paragraphs and filter out trivial or numeric-heavy ones."""

    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    filtered: List[str] = []
    for para in raw_paragraphs:
        if len(para) < min_length:
            continue
        if _is_numeric_heavy(para, numeric_ratio):
            continue
        filtered.append(para)
    return filtered


def _is_numeric_heavy(paragraph: str, threshold: float) -> bool:
    if not paragraph:
        return True
    numeric_count = sum(1 for ch in paragraph if ch in NUMERIC_SYMBOLS)
    ratio = numeric_count / max(len(paragraph), 1)
    return ratio >= threshold


KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "item1": (
        "business",
        "segment",
        "product",
        "service",
        "customers",
        "competition",
        "market",
        "revenue",
    ),
    "item7": (
        "results of operations",
        "increase",
        "decrease",
        "margin",
        "costs",
        "expenses",
        "drivers",
        "management",
    ),
    "item2": (
        "quarter",
        "three months",
        "six months",
        "trend",
        "risk",
        "uncertainties",
    ),
}


def score_paragraph(paragraph: str, section_type: str) -> float:
    """Assign a heuristic score based on length and keyword presence."""

    max_length = 1_200
    length_score = min(len(paragraph), max_length) / max_length
    keyword_score = 0.0
    lowered = paragraph.lower()
    for keyword in KEYWORDS.get(section_type, ()):  # pragma: no branch
        if keyword in lowered:
            keyword_score += 0.5
    return length_score + keyword_score


def reduce_section(text: Optional[str], section_type: str) -> Optional[str]:
    if text is None:
        return None

    cleaned = clean_raw_text(text)
    if not cleaned:
        return None

    paragraphs = split_into_paragraphs(cleaned)
    if not paragraphs:
        return None

    scored_paragraphs = sorted(
        ((para, score_paragraph(para, section_type)) for para in paragraphs),
        key=lambda item: item[1],
        reverse=True,
    )

    budget = SECTION_BUDGETS.get(section_type, 12_000)
    selected: List[str] = []
    current_length = 0
    for para, _score in scored_paragraphs:
        if current_length >= budget and selected:
            break
        selected.append(para)
        current_length += len(para) + 2  # account for blank line
    return "\n\n".join(selected)[:budget]


def process_file(path: Path, output_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    cleaned_payload = {
        "source_file": data.get("source_file"),
        "form_type": data.get("form_type"),
        "item1_clean": reduce_section(data.get("item1"), "item1"),
        "item7_clean": reduce_section(data.get("item7"), "item7"),
        "item2_clean": reduce_section(data.get("item2"), "item2"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem.replace('_sections', '')}_clean.json"
    output_path.write_text(json.dumps(cleaned_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sections_found = [key for key in ("item1", "item7", "item2") if data.get(key)]
    print(f"Processed {path.name}: sections={','.join(sections_found) if sections_found else 'none'}")


def process_all(input_dir: str = "json", output_dir: str = "json_clean") -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for file_path in sorted(input_path.glob("*_sections.json")):
        process_file(file_path, output_path)


if __name__ == "__main__":
    process_all()
