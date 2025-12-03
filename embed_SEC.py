import re
import json
from pathlib import Path
from typing import Optional, List, Dict

# ------------------------------
# 1. 섹션 추출용 패턴 정의
# ------------------------------

SECTION_PATTERNS = {
    "10-K": {
        "item1": {
            # 시작 패턴: Item 1. Business / ITEM 1 – BUSINESS / ITEM 1.
            "start": [
                r"^\s*item\s+1\.\s*business",        # Item 1. Business
                r"^\s*item\s+1\s*[-–]\s*business",   # Item 1 - Business
                r"^\s*item\s+1\.\s*$"                # Item 1.
            ],
            # 끝 패턴: 보통 Item 1A, 1B, 2 근처에서 끝
            "end": [
                r"^\s*item\s+1a\.",                  # Item 1A. Risk Factors
                r"^\s*item\s+1b\.",
                r"^\s*item\s+2\."                    # Item 2.
            ]
        },
        "item7": {
            # 시작 패턴: Item 7. Management's Discussion ...
            "start": [
                r"^\s*item\s+7\.\s*management[’'`s]{0,3}\s+discussion",
                r"^\s*item\s+7\.\s*$"                # 희귀케이스: 제목만 Item 7.
            ],
            # 끝 패턴: 보통 Item 7A 또는 Item 8에서 끝
            "end": [
                r"^\s*item\s+7a\.",                  # Item 7A. Quantitative ...
                r"^\s*item\s+8\."                    # Item 8. Financial Statements ...
            ]
        }
    },
    "10-Q": {
        "item2": {
            # 시작 패턴: Item 2. Management's Discussion ...
            "start": [
                r"^\s*item\s+2\.\s*management[’'`s]{0,3}\s+discussion",
                r"^\s*item\s+2\.\s*$"
            ],
            # 끝 패턴: 보통 Item 3 또는 4에서 끝
            "end": [
                r"^\s*item\s+3\.",                   # Item 3. Quantitative ...
                r"^\s*item\s+4\."                    # Item 4. Controls and Procedures
            ]
        }
    }
}


# ------------------------------
# 2. 유틸 함수들
# ------------------------------

def detect_form_type(text: str) -> Optional[str]:
    """
    SEC 원문에서 Form 10-K / 10-Q 타입 자동 감지.
    """
    head = text[:5000].lower()  # 앞부분만 보면 대부분 충분
    if "form 10-k" in head:
        return "10-K"
    if "form 10-q" in head:
        return "10-Q"
    return None


def maybe_strip_html(text: str) -> str:
    """
    만약 HTML 태그가 많이 보이면, 아주 거칠게 태그 제거.
    (이미 텍스트로 정제돼 있으면 그대로 사용)
    """
    if "<html" in text.lower() or "<document>" in text.lower():
        # 너무 세게 지우면 안되니까, 그냥 태그만 날리는 수준
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def find_section(
    text: str,
    start_patterns: List[str],
    end_patterns: List[str]
) -> Optional[str]:
    """
    start_patterns 중 하나를 찾은 위치를 시작점으로,
    end_patterns 중 가장 먼저 등장하는 것을 끝점으로 해서 섹션 텍스트를 반환.
    못 찾으면 None.
    """
    start_idx = None

    # start 찾기
    for pat in start_patterns:
        regex = re.compile(pat, re.IGNORECASE | re.MULTILINE)
        m = regex.search(text)
        if m:
            start_idx = m.start()
            break

    if start_idx is None:
        return None

    end_idx = len(text)

    # end 찾기 (start 이후에서)
    for pat in end_patterns:
        regex = re.compile(pat, re.IGNORECASE | re.MULTILINE)
        m = regex.search(text, pos=start_idx + 10)  # 제목 바로 다음에서부터 검색
        if m:
            if m.start() < end_idx:
                end_idx = m.start()

    section = text[start_idx:end_idx].strip()
    # 너무 짧으면(헤딩만 잡은 듯) None 처리
    if len(section) < 50:
        return None
    return section


def extract_sections_from_text(text: str) -> Dict[str, Optional[str]]:
    """
    원문 텍스트에서 form 타입을 감지하고,
    해당 타입에 맞는 Item 1 / 7 / 2를 추출해서 dict로 반환.
    """
    text = maybe_strip_html(text)

    form_type = detect_form_type(text)
    if form_type is None:
        return {"form_type": None, "item1": None, "item7": None, "item2": None}

    patterns = SECTION_PATTERNS.get(form_type, {})

    item1 = None
    item7 = None
    item2 = None

    if form_type == "10-K":
        # Item 1
        p = patterns.get("item1")
        if p:
            item1 = find_section(text, p["start"], p["end"])

        # Item 7
        p = patterns.get("item7")
        if p:
            item7 = find_section(text, p["start"], p["end"])

    elif form_type == "10-Q":
        # Item 2
        p = patterns.get("item2")
        if p:
            item2 = find_section(text, p["start"], p["end"])

    return {
        "form_type": form_type,
        "item1": item1,
        "item7": item7,
        "item2": item2,
    }


# ------------------------------
# 3. 디렉토리 순회 → json 저장
# ------------------------------

def process_filings(
    input_dir: str = "sec_filings",
    output_dir: str = "json"
):
    """
    input_dir 안의 .txt / .html / .htm 파일을 모두 읽어서
    각 파일에 대해 item1 / item7 / item2를 추출하고
    output_dir/json 에 {stem}_sections.json 으로 저장.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in input_path.glob("*"):
        if not file_path.suffix.lower() in [".txt", ".html", ".htm"]:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="latin-1", errors="ignore")

        sections = extract_sections_from_text(text)

        out_data = {
            "source_file": file_path.name,
            "form_type": sections["form_type"],
            "item1": sections["item1"],
            "item7": sections["item7"],
            "item2": sections["item2"],
        }

        out_file = output_path / f"{file_path.stem}_sections.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] {file_path.name} -> {out_file.name} (form: {sections['form_type']})")


if __name__ == "__main__":
    # 필요하면 여기서 경로 수정해서 사용하면 됨
    process_filings(input_dir="sec_filings", output_dir="json")