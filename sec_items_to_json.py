import argparse
import json
from pathlib import Path
from typing import Iterable

from sec_pipeline.sections import build_filing_sections, dump_filing_sections


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    for suffix in ("*.txt", "*.html", "*.htm"):
        yield from input_dir.glob(suffix)


def convert_filings(input_dir: Path, output_dir: Path) -> None:
    for filing_path in iter_input_files(input_dir):
        sections = build_filing_sections(filing_path)
        output_path = output_dir / f"{filing_path.stem}_sections.json"
        dump_filing_sections(sections, output_path)
        print(json.dumps(sections.__dict__, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Item 1, Item 7, Item 2 sections from SEC filings into JSON.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("sec_filings"),
        help="Directory containing 10-K/10-Q files (.txt, .html).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("json"),
        help="Directory where section JSON files will be saved.",
    )
    args = parser.parse_args()
    convert_filings(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
