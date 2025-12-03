import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from bs4 import BeautifulSoup


ITEM_REGEXES = {
    "item1": re.compile(r"item\s*1\.?\s*business", re.IGNORECASE),
    "item7": re.compile(
        r"item\s*7\.?\s*management'?s\s+discussion\s+and\s+analysis", re.IGNORECASE
    ),
    "item2": re.compile(r"item\s*2\.?\s*management'?s\s+discussion", re.IGNORECASE),
}


@dataclass
class FilingSections:
    source_file: str
    form_type: Optional[str]
    item1: Optional[str]
    item7: Optional[str]
    item2: Optional[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)


def _read_html(path: Path) -> str:
    html_text = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text("\n")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_filing_text(path: Path) -> str:
    if path.suffix.lower() in {".html", ".htm"}:
        return _read_html(path)
    return _read_text(path)


def _split_into_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        normalized = line.strip()
        if normalized:
            yield normalized


def _find_item_positions(lines: Iterable[str]) -> Dict[str, int]:
    positions: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        for key, pattern in ITEM_REGEXES.items():
            if key in positions:
                continue
            if pattern.search(line):
                positions[key] = idx
    return positions


def _slice_section(lines: Iterable[str], start_idx: int, end_idx: Optional[int]) -> str:
    selected = list(lines)[start_idx:end_idx]
    return "\n".join(selected).strip() or None


def extract_sections(text: str) -> Dict[str, Optional[str]]:
    lines = list(_split_into_lines(text))
    positions = _find_item_positions(lines)

    sorted_items = sorted(positions.items(), key=lambda kv: kv[1])
    boundaries: Dict[str, Optional[int]] = {}
    for i, (item_key, start_idx) in enumerate(sorted_items):
        next_start = sorted_items[i + 1][1] if i + 1 < len(sorted_items) else None
        boundaries[item_key] = next_start

    sections: Dict[str, Optional[str]] = {"item1": None, "item7": None, "item2": None}
    for item_key, start_idx in positions.items():
        sections[item_key] = _slice_section(lines, start_idx, boundaries[item_key])
    return sections


def guess_form_type(path: Path) -> Optional[str]:
    lower_name = path.name.lower()
    if "10k" in lower_name or "10-k" in lower_name:
        return "10-K"
    if "10q" in lower_name or "10-q" in lower_name:
        return "10-Q"
    return None


def build_filing_sections_from_text(
    text: str, source_name: str, form_type: Optional[str] = None
) -> FilingSections:
    sections = extract_sections(text)
    return FilingSections(
        source_file=source_name,
        form_type=form_type,
        item1=sections.get("item1"),
        item7=sections.get("item7"),
        item2=sections.get("item2"),
    )


def build_filing_sections(path: Path) -> FilingSections:
    text = load_filing_text(path)
    return build_filing_sections_from_text(
        text, source_name=path.name, form_type=guess_form_type(path)
    )


def dump_filing_sections(sections: FilingSections, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(sections.to_json(), encoding="utf-8")


def load_sections_from_json(path: Path) -> FilingSections:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FilingSections(
        source_file=data.get("source_file", path.name),
        form_type=data.get("form_type"),
        item1=data.get("item1"),
        item7=data.get("item7"),
        item2=data.get("item2"),
    )
