"""Fetch S&P 500 filings, extract Item 1/7/2, summarize, and embed.

This script pulls the current S&P 500 constituents from Wikipedia, resolves
their CIKs via the SEC API, downloads the latest 10-K/10-Q, extracts sections,
summarizes the full paragraphs with a long-context summarizer, and saves
embeddings to ``chromaDB/sec``.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from sec_pipeline.sections import (
    FilingSections,
    build_filing_sections_from_text,
    dump_filing_sections,
)
from sec_pipeline.workflow import (
    DEFAULT_CHUNK_CHAR_BUDGET,
    build_summaries,
    embed_summaries,
    save_summary_json,
)


SEC_BASE = "https://data.sec.gov"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
RATE_LIMIT_SECONDS = 0.2
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHROMA_OUTPUT_DIR = Path("chromaDB") / "sec"


@dataclass
class LatestFilings:
    ten_k: Optional[FilingSections]
    ten_q: Optional[FilingSections]


def _headers(user_agent: str) -> Dict[str, str]:
    return {"User-Agent": user_agent}


def _sleep_between_requests() -> None:
    time.sleep(RATE_LIMIT_SECONDS)


def fetch_ticker_map(user_agent: str) -> Dict[str, str]:
    resp = requests.get(TICKER_MAP_URL, headers=_headers(user_agent))
    resp.raise_for_status()
    data = resp.json()
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


def fetch_sp500_tickers(user_agent: str) -> List[str]:
    resp = requests.get(SP500_WIKI_URL, headers=_headers(user_agent))
    resp.raise_for_status()
    html = resp.text

    pattern = re.compile(r"<td><a href=\"/wiki/[^\"]+\"[^>]*>([A-Z\.]+)</a></td>")
    seen = set()
    tickers: List[str] = []
    for match in pattern.finditer(html):
        symbol = match.group(1).upper()
        if symbol not in seen:
            seen.add(symbol)
            tickers.append(symbol)
    return tickers


def resolve_cik(identifier: str, user_agent: str, ticker_map: Optional[Dict[str, str]] = None) -> str:
    identifier = identifier.strip()
    if identifier.isdigit():
        return identifier.zfill(10)

    ticker_lookup = ticker_map or fetch_ticker_map(user_agent)
    cik = ticker_lookup.get(identifier.upper())
    if not cik:
        raise ValueError(f"CIK lookup failed for identifier: {identifier}")
    return cik


def fetch_company_submissions(cik: str, user_agent: str) -> Dict:
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=_headers(user_agent))
    resp.raise_for_status()
    return resp.json()


def _build_document_url(cik: str, accession: str, primary_doc: str) -> str:
    accession_nodash = accession.replace("-", "")
    return f"{ARCHIVES_BASE}/{int(cik)}/{accession_nodash}/{primary_doc}"


def fetch_filing_text(url: str, user_agent: str) -> str:
    resp = requests.get(url, headers=_headers(user_agent))
    resp.raise_for_status()
    return resp.text


def _extract_latest_filing(
    submissions: Dict, cik: str, form_type: str, user_agent: str
) -> Optional[FilingSections]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])

    for idx, form in enumerate(forms):
        if form.lower() != form_type.lower():
            continue
        accession = accessions[idx]
        primary_doc = documents[idx]
        url = _build_document_url(cik, accession, primary_doc)
        _sleep_between_requests()
        filing_text = fetch_filing_text(url, user_agent)
        return build_filing_sections_from_text(
            filing_text, source_name=primary_doc, form_type=form_type
        )
    return None


def fetch_latest_sections(
    identifier: str, user_agent: str, ticker_map: Optional[Dict[str, str]] = None
) -> LatestFilings:
    cik = resolve_cik(identifier, user_agent, ticker_map=ticker_map)
    submissions = fetch_company_submissions(cik, user_agent)

    ten_k = _extract_latest_filing(submissions, cik, "10-K", user_agent)
    ten_q = _extract_latest_filing(submissions, cik, "10-Q", user_agent)
    return LatestFilings(ten_k=ten_k, ten_q=ten_q)


def save_sections(
    filings: LatestFilings, identifier: str, output_dir: Path
) -> Dict[str, Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Optional[Path]] = {"10-K": None, "10-Q": None}

    if filings.ten_k:
        path = output_dir / f"{identifier}_latest_10K_sections.json"
        dump_filing_sections(filings.ten_k, path)
        saved_paths["10-K"] = path

    if filings.ten_q:
        path = output_dir / f"{identifier}_latest_10Q_sections.json"
        dump_filing_sections(filings.ten_q, path)
        saved_paths["10-Q"] = path

    return saved_paths


def _combine_latest_sections(filings: LatestFilings) -> Optional[FilingSections]:
    if not (filings.ten_k or filings.ten_q):
        return None
    return FilingSections(
        source_file=(filings.ten_k or filings.ten_q).source_file,
        form_type="combined",
        item1=filings.ten_k.item1 if filings.ten_k else None,
        item7=filings.ten_k.item7 if filings.ten_k else None,
        item2=filings.ten_q.item2 if filings.ten_q else None,
    )


def _summarize_and_embed(
    filings: LatestFilings,
    identifier: str,
    summary_dir: Path,
    chunk_budget: int,
    max_workers: int,
) -> Optional[Path]:
    combined = _combine_latest_sections(filings)
    if not combined:
        return None

    summaries = build_summaries(
        combined, char_budget=chunk_budget, max_workers=max_workers
    )
    summary_path = summary_dir / f"{identifier}_latest_summaries.json"
    save_summary_json(summaries, summary_path)

    CHROMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embed_path = CHROMA_OUTPUT_DIR / f"{identifier}_latest_embeddings.json"
    embed_summaries(summaries, embed_path)
    return embed_path


def _log(msg: str) -> None:
    print(msg)


def _batched(iterable: Iterable[str], limit: Optional[int]) -> List[str]:
    items = list(iterable)
    return items[:limit] if limit else items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch S&P 500 latest 10-K/10-Q, summarize, and embed sections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("json"),
        help="Directory to store section JSON outputs.",
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help="Custom User-Agent header required by the SEC (e.g., 'Name Contact@domain').",
    )
    parser.add_argument(
        "--max-companies",
        type=int,
        default=None,
        help="Optional cap for S&P 500 tickers to process (for quick runs).",
    )
    parser.add_argument(
        "--chunk-char-budget",
        type=int,
        default=DEFAULT_CHUNK_CHAR_BUDGET,
        help="Character budget per chunk (~4 chars/token).",
    )
    parser.add_argument(
        "--summary-workers",
        type=int,
        default=4,
        help="Parallel workers for long-context summarizer.",
    )

    args = parser.parse_args()

    ticker_map = fetch_ticker_map(args.user_agent)
    tickers = _batched(fetch_sp500_tickers(args.user_agent), args.max_companies)
    _log(f"Processing {len(tickers)} S&P 500 tickers (chunk budget={args.chunk_char_budget}).")

    summary_dir = args.output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ticker in tickers:
        try:
            filings = fetch_latest_sections(ticker, args.user_agent, ticker_map=ticker_map)
            saved = save_sections(filings, ticker, args.output_dir)
            embed_path = _summarize_and_embed(
                filings,
                ticker,
                summary_dir,
                chunk_budget=args.chunk_char_budget,
                max_workers=args.summary_workers,
            )
            results.append(
                {
                    "ticker": ticker,
                    "ten_k": saved.get("10-K").name if saved.get("10-K") else None,
                    "ten_q": saved.get("10-Q").name if saved.get("10-Q") else None,
                    "embedding": embed_path.name if embed_path else None,
                }
            )
            _log(
                f"{ticker}: 10-K={'yes' if saved.get('10-K') else 'no'} | "
                f"10-Q={'yes' if saved.get('10-Q') else 'no'} | "
                f"embeddings={'yes' if embed_path else 'no'}"
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"{ticker}: skipped due to error -> {exc}")

    print(json.dumps({"processed": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
