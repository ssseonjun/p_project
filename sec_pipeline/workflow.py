import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from sec_pipeline.embedding_pipeline import EmbeddingConfig, SummaryEmbeddingPipeline
from sec_pipeline.sections import FilingSections
from sec_pipeline.summary_templates import SummaryResult


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
DEFAULT_CHUNK_CHAR_BUDGET = 4096 * 4  # ~4 chars/token heuristic for 4096-token chunks


def naive_summarizer(text: str, max_sentences: int = 4) -> str:
    sentences = SENTENCE_SPLIT.split(text.strip())
    shortened = sentences[:max_sentences]
    return " ".join(shortened).strip()


def _chunk_paragraphs(paragraphs: Iterable[str], char_budget: int) -> Iterable[str]:
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if not para:
            continue
        if current_len + len(para) + 1 > char_budget and current:
            yield "\n\n".join(current)
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2
    if current:
        yield "\n\n".join(current)


def _summarize_with_long_context(
    text: str,
    summarizer: Callable[[str], str],
    char_budget: int = DEFAULT_CHUNK_CHAR_BUDGET,
) -> str:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks = list(_chunk_paragraphs(paragraphs, char_budget=char_budget)) or [""]
    chunk_summaries = [summarizer(chunk) for chunk in chunks if chunk]
    return "\n\n".join(chunk_summaries).strip()


def build_summaries(
    sections: FilingSections,
    summarizer: Callable[[str], str] = naive_summarizer,
    char_budget: int = DEFAULT_CHUNK_CHAR_BUDGET,
    max_workers: int = 4,
) -> SummaryResult:
    section_map: Dict[str, Optional[str]] = {
        "item1": sections.item1,
        "item7": sections.item7,
        "item2": sections.item2,
    }

    def summarize_payload(payload: str) -> str:
        return _summarize_with_long_context(payload, summarizer=summarizer, char_budget=char_budget)

    return parallel_summarize_sections(section_map, summarize_payload, max_workers=max_workers)


def parallel_summarize_sections(
    sections: Dict[str, Optional[str]],
    summarizer: Callable[[str], str],
    max_workers: int = 4,
) -> SummaryResult:
    def _submit(job_sections: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {"item1": None, "item7": None, "item2": None}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                name: executor.submit(summarizer, content)
                for name, content in job_sections.items()
                if content
            }
            for name, future in future_map.items():
                results[name] = future.result()
        return results

    summarized = _submit(sections)
    return SummaryResult(
        item1=summarized.get("item1"),
        item7=summarized.get("item7"),
        item2=summarized.get("item2"),
    )


def embed_summaries(
    summaries: SummaryResult, output_path: Path, config: EmbeddingConfig = EmbeddingConfig()
) -> Dict[str, Dict[str, list]]:
    pipeline = SummaryEmbeddingPipeline(config)
    embeddings = pipeline.embed_summary_payload(
        {"item1": summaries.item1, "item7": summaries.item7, "item2": summaries.item2}
    )
    pipeline.save_embeddings(embeddings, output_path)
    return embeddings


def save_summary_json(summaries: SummaryResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "item1": summaries.item1,
        "item7": summaries.item7,
        "item2": summaries.item2,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
