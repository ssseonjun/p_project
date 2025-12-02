# sec_item1_to_json.py
import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "json"
SEC_DIR = DATA_DIR / "business_json"
SEC_DIR.mkdir(parents=True, exist_ok=True)

ITEM1_RAW_DIR = SEC_DIR / "raw"
ITEM1_RAW_DIR.mkdir(parents=True, exist_ok=True)

# SEC 권장: User-Agent에 본인 이름/메일 꼭 넣기
SEC_HEADERS = {
    "User-Agent": "SEONJUN qkrtjswns@gachon.ac.kr",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

SP500_LIST_CSV = BASE_DIR / "csv/sp500_list.csv"


def load_sp500_list() -> pd.DataFrame:
    if not SP500_LIST_CSV.exists():
        raise FileNotFoundError(
            f"{SP500_LIST_CSV} 파일이 없습니다. "
            f"index,symbol,name,cik 형식으로 저장해 주세요."
        )

    df = pd.read_csv(SP500_LIST_CSV)
    df = df.rename(columns={"symbol": "ticker", "name": "name", "cik": "cik"})

    required = {"ticker", "name", "cik"}
    if not required.issubset(df.columns):
        raise ValueError("CSV에 'symbol','name','cik' 컬럼이 있는지 확인해 주세요.")

    return df


def get_latest_10k_metadata(cik: str) -> Optional[Dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS)
    if r.status_code != 200:
        print(f"[WARN] submissions fetch failed for CIK={cik}: {r.status_code}")
        return None

    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession = recent.get("accessionNumber", [])
    primary_doc = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    for form, acc, doc, fdate in zip(forms, accession, primary_doc, filing_dates):
        if form == "10-K":
            return {
                "accession": acc,
                "primary_doc": doc,
                "filing_date": fdate,
                "company_name": data.get("name"),
            }
    return None


def fetch_10k_html(cik: str, accession: str, primary_doc: str) -> str:
    cik_num = str(int(cik))  # "0001579241" -> "1579241"
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_nodash}/{primary_doc}"

    r = requests.get(url, headers={"User-Agent": SEC_HEADERS["User-Agent"]}, timeout=30)
    r.raise_for_status()
    return r.text


def extract_itmem1(text: str, min_chars: int = 200) -> str:
    """
    10-K 전체 텍스트에서 'Item 1 계열(1,1A,1B,...) ~ 다음 Item 번호(2,3,...) 직전' 구간을 추출.
    - Item 1/1A/1B가 목차/본문에 여러 번 나와도, 그 중 길이가 min_chars 이상인
      섹션들만 후보로 두고, 가장 긴 섹션 하나를 반환.
    """
    if not text or not text.strip():
        return ""

    lower = text.lower()

    # 모든 item 헤더 위치 잡기: Item 1, 1A, 2, 3, 7A, 10, 11B ...
    header_re = re.compile(
        r"item\s*([0-9]{1,2}[a-z]?)\b",
        flags=re.IGNORECASE,
    )

    matches = list(header_re.finditer(lower))
    if not matches:
        return ""

    def is_item1_family(item_no: str) -> bool:
        """
        '1', '1a', '1b', '1c' 등을 Item 1 계열로 보고,
        '10', '11' 같은 건 제외하기 위한 헬퍼.
        """
        item_no = item_no.lower()
        if item_no == "1":
            return True
        if len(item_no) == 2 and item_no[0] == "1" and item_no[1].isalpha():
            return True
        return False

    candidates = []

    # 모든 Item 1 계열 헤더를 시작점으로 삼아, 다음 "비-1계열" item까지를 블록으로 본다.
    for i, m in enumerate(matches):
        item_no = m.group(1).lower()
        if not is_item1_family(item_no):
            continue

        start_idx = m.start()

        # 다음 "Item 2 이상"이 나오는 지점 찾기
        end_idx = len(text)
        for j in range(i + 1, len(matches)):
            next_item = matches[j].group(1).lower()
            if not is_item1_family(next_item):
                end_idx = matches[j].start()
                break

        section = text[start_idx:end_idx].strip()
        if len(section) >= min_chars:
            candidates.append(section)

    if not candidates:
        return ""

    # 가장 긴 섹션 = 진짜 본문일 확률이 가장 높음
    best_section = max(candidates, key=len)
    return best_section


def split_item1(block_text: str) -> Dict[str, str]:
    """
    Item 1 계열 블록 텍스트를 받아서
      - item1
      - item1a
      - item1b
      ...
    로 나눠서 dict로 반환.

    규칙:
      - 'item1'은 무조건 저장.
      - 'item1a' 이후는:
          * NOTE0-9(대소문자 무시) 패턴이 존재하고
          * 전체 길이가 100 미만이면 → 버림.
    """
    sections: Dict[str, str] = {}

    if not block_text or not block_text.strip():
        return sections

    lower = block_text.lower()

    # Item 1, 1A, 1B ... 헤더 찾기
    #  - group(1)이 ''이면 item1
    #  - group(1)이 'a','b'면 item1a, item1b
    header_re = re.compile(
        r"item\s*1([a-z])?\b",
        flags=re.IGNORECASE,
    )

    markers = []
    for m in header_re.finditer(lower):
        letter = m.group(1).lower() if m.group(1) else ""
        label = "item1" if letter == "" else f"item1{letter}"
        markers.append((m.start(), label))

    if not markers:
        return sections

    # 출현 순서대로 정렬
    markers.sort(key=lambda x: x[0])

    # 각 헤더 기준으로 다음 헤더 직전까지 잘라서 섹션 구성
    for idx, (start_pos, label) in enumerate(markers):
        if idx + 1 < len(markers):
            end_pos = markers[idx + 1][0]
        else:
            end_pos = len(block_text)

        seg_text = block_text[start_pos:end_pos].strip()
        if not seg_text:
            continue

        if label == "item1":
            # item1은 무조건 저장
            sections[label] = seg_text
        else:
            # item1a, item1b ... 에 대해 NOTE + 길이 필터
            has_note = re.search(r"note\s*[0-9]", seg_text, flags=re.IGNORECASE) is not None
            seg_len = len(seg_text)

            if has_note and seg_len < 100:
                # NOTE0-9 포함 + 길이 < 100 → 버림
                continue
            else:
                sections[label] = seg_text

    return sections



def main():
    sp500_df = load_sp500_list()
    print(f"[INFO] Total S&P500 companies: {len(sp500_df)}")

    miss_list = []

    start_idx = 0
    for _, row in tqdm(
        sp500_df.iloc[start_idx:].iterrows(),
        total=len(sp500_df) - start_idx,
        desc="SEC Item1 -> JSON (raw only)",
    ):
        ticker = row["ticker"]
        name = row["name"]
        cik = str(row["cik"]).zfill(10)

        raw_path = SEC_DIR / f"{ticker}.json"
        if raw_path.exists():
            continue

        meta = get_latest_10k_metadata(cik)
        if meta is None:
            print(f"[MISS 10-K] {ticker} {name} (CIK={cik})")
            miss_list.append(
                {"ticker": ticker, "name": name, "cik": cik, "reason": "no_10k"}
            )
            continue

        try:
            html = fetch_10k_html(cik, meta["accession"], meta["primary_doc"])
        except Exception as e:
            print(f"[ERR GET HTML] {ticker} {name}: {e}")
            miss_list.append(
                {"ticker": ticker, "name": name, "cik": cik, "reason": "html_error"}
            )
            continue

        soup = BeautifulSoup(html, "html.parser")
        full_text = soup.get_text("\n")

        item1_block = extract_itmem1(full_text, min_chars=400)
        if not item1_block.strip():
            print(f"[MISS ITEM1_BLOCK] {ticker} {name}")
            miss_list.append(
                {"ticker": ticker, "name": name, "cik": cik, "reason": "no_item1_block"}
            )
            continue

        sections = split_item1(item1_block)

        item1_text = sections.get("item1", "")
        if not item1_text.strip():
            print(f"[MISS ITEM1] {ticker} {name}")
            miss_list.append(
                {"ticker": ticker, "name": name, "cik": cik, "reason": "no_item1"}
            )
            continue

        raw_data = {
            "ticker": ticker,
            "cik": cik,
            "company_name": meta["company_name"],
            "filing_date": meta["filing_date"],
            "accession": meta["accession"],
            "primary_doc": meta["primary_doc"],
            "item_sections": sections,
        }
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

    miss_path = SEC_DIR / "_item1_miss_list.json"
    with open(miss_path, "w", encoding="utf-8") as f:
        json.dump(miss_list, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved miss list to {miss_path}")


if __name__ == "__main__":
    main()