from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class SummaryResult:
    item1: Optional[str]
    item7: Optional[str]
    item2: Optional[str]


ItemSummarizer = Callable[[str], str]


ITEM1_TEMPLATE = (
    "회사 정체성 요약: 무슨 제품/서비스를 누구에게 제공하고, 어떤 수익 모델과 경쟁우위를 가지는지 3-5줄로 서술하세요."
)
ITEM7_TEMPLATE = (
    "연간 MD&A 요약: 매출/마진의 방향성, 주요 원인, 경영진 코멘트와 리스크를 3-5개의 bullet로 요약하세요."
)
ITEM2_TEMPLATE = (
    "분기 MD&A 요약: 직전 분기 혹은 전년동기 대비 변화와 신규 리스크를 3-5줄로 서술하세요."
)


def build_prompt(section_name: str) -> str:
    if section_name == "item1":
        return ITEM1_TEMPLATE
    if section_name == "item7":
        return ITEM7_TEMPLATE
    if section_name == "item2":
        return ITEM2_TEMPLATE
    raise ValueError(f"Unsupported section: {section_name}")


def summarize_sections(
    sections: Dict[str, Optional[str]],
    summarizer: ItemSummarizer,
) -> SummaryResult:
    summaries: Dict[str, Optional[str]] = {"item1": None, "item7": None, "item2": None}
    for section_name, content in sections.items():
        if not content:
            continue
        prompt = build_prompt(section_name)
        summaries[section_name] = summarizer(f"{prompt}\n\n{content}")
    return SummaryResult(**summaries)
