# sec_item1_to_json.py
import json
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "json"
SEC_DIR = DATA_DIR / "business_json"
SEC_DIR.mkdir(parents=True, exist_ok=True)

ITEM1_RAW_DIR = SEC_DIR / "raw"
ITEM1_SUM_DIR = SEC_DIR / "summary"
ITEM1_RAW_DIR.mkdir(parents=True, exist_ok=True)
ITEM1_SUM_DIR.mkdir(parents=True, exist_ok=True)

# SEC 권장: User-Agent에 본인 이름/메일 꼭 넣기
SEC_HEADERS = {
    "User-Agent": "SEONJUN qkrtjswns@gachon.ac.kr",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# 네가 준 CSV: index,symbol,name,cik
SP500_LIST_CSV = BASE_DIR / "csv/sp500_list.csv"


# -----------------------------
# 1. S&P500 리스트 로드 (로컬 CSV 사용)
# -----------------------------
def load_sp500_list() -> pd.DataFrame:
    if not SP500_LIST_CSV.exists():
        raise FileNotFoundError(f"{SP500_LIST_CSV} 파일이 없습니다. "
                                f"index,symbol,name,cik 형식으로 저장해 주세요.")

    df = pd.read_csv(SP500_LIST_CSV)

    # 표준 컬럼명으로 정리
    df = df.rename(columns={
        "symbol": "ticker",
        "name": "name",
        "cik": "cik",
    })

    required = {"ticker", "name", "cik"}
    if not required.issubset(df.columns):
        raise ValueError("CSV에 'symbol','name','cik' 컬럼이 있는지 확인해 주세요.")

    return df


# -----------------------------
# 2. 최신 10-K 메타데이터 + HTML
# -----------------------------
def get_latest_10k_metadata(cik: str) -> Optional[Dict]:
    """
    submissions API에서 최신 10-K 하나 가져오기.
    cik: 10자리 zero-padding 문자열 ("0001579241")
    """
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
    """
    예:
      cik        = "0001579241"
      accession  = "0001579241-25-000008"
      primary_doc= "alle-20241231.htm"

    실제 URL:
      https://www.sec.gov/Archives/edgar/data/1579241/000157924125000008/alle-20241231.htm
    """
    cik_num = str(int(cik))  # "0001579241" -> "1579241"
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_nodash}/{primary_doc}"

    r = requests.get(url, headers={"User-Agent": SEC_HEADERS["User-Agent"]}, timeout=30)
    r.raise_for_status()
    return r.text


# -----------------------------
# 3. Item 1. Business 추출
# -----------------------------
import re
from typing import List

import re
from typing import List

def extract_item1_business(text: str, min_chars: int = 400) -> str:
    """
    10-K 전체 텍스트에서 Item 1. Business 섹션 추출.

    로직:
      1) "Item 1 ... Business" 패턴(start_re)을 전부 찾는다. (TOC + 본문 모두)
      2) 각 시작점 기준으로,
         - "Item 1A" 또는
         - "Item 2"
         가 나오는 지점을 종료 후보로 본다.
      3) [start, end) 구간을 잘라서 길이가 min_chars 이상이면 '진짜 섹션 후보'로 채택한다.
      4) 후보가 여러 개면 가장 긴 섹션을 반환.
         (목차는 몇 줄 안 되기 때문에 자동으로 걸러짐)
    """

    if not text or not text.strip():
        return ""

    lower = text.lower()

    # 1) 시작 패턴: Item 1 ... Business
    #   - "ITEM 1. BUSINESS"
    #   - "Item 1 Business"
    #   - "item 1\nbusiness" 등 대응
    start_re = re.compile(
        r"item\s*1[\s\.\-]*[\s\S]{0,100}?business",
        flags=re.IGNORECASE
    )

    # 2) 종료 후보 패턴들
    #   - Item 1A (Risk Factors)
    #   - Item 2 (Properties 등)
    end_res = [
        re.compile(r"item\s*1a[\s\.\-]*[\s\S]{0,80}?risk", flags=re.IGNORECASE),
        re.compile(r"item\s*1a[\s\.\-]*", flags=re.IGNORECASE),
        re.compile(r"item\s*2[\s\.\-]*[\s\S]{0,80}?properties", flags=re.IGNORECASE),
        re.compile(r"item\s*2[\s\.\-]*", flags=re.IGNORECASE),
    ]

    candidates: List[str] = []

    for m_start in start_re.finditer(lower):
        start_idx = m_start.start()

        end_idx_candidates = []
        for end_re in end_res:
            m_end = end_re.search(lower, m_start.end())
            if m_end:
                end_idx_candidates.append(m_end.start())

        if end_idx_candidates:
            end_idx = min(end_idx_candidates)
        else:
            # 종료 패턴을 못 찾으면 끝까지를 Item 1 구간으로 가정
            end_idx = len(text)

        section = text[start_idx:end_idx].strip()

        # 너무 짧으면 (목차 같은) 가짜로 보고 버린다
        if len(section) >= min_chars:
            candidates.append(section)

    if not candidates:
        return ""

    # 여러 후보 중 가장 긴 섹션(= 진짜 본문일 확률이 가장 높음)을 반환
    best_section = max(candidates, key=len)
    return best_section

# -----------------------------
# 4. 요약용 헬퍼
# -----------------------------
def chunk_text(text: str, max_chars: int = 1800):
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]


def summarize_item1(text: str, summarizer) -> str:
    if not text.strip():
        return ""

    chunks = list(chunk_text(text, max_chars=1800))
    summaries = []
    for ch in chunks:
        out = summarizer(
            ch,
            max_length=256,
            min_length=64,
            do_sample=False,
        )
        summaries.append(out[0]["summary_text"])
    return "\n".join(summaries)


# -----------------------------
# 5. main
# -----------------------------
def main():
    # 1) S&P500 리스트 로드
    sp500_df = load_sp500_list()
    print(f"[INFO] Total S&P500 companies: {len(sp500_df)}")

    # 2) 요약 모델 로드
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        tokenizer="sshleifer/distilbart-cnn-12-6",
    )

    miss_list = []

    for _, row in tqdm(sp500_df.iterrows()[158], total=len(sp500_df), desc="SEC Item1 -> JSON"):
        ticker = row["ticker"]
        name = row["name"]
        cik = str(row["cik"]).zfill(10)

        raw_path = ITEM1_RAW_DIR / f"{ticker}.json"
        sum_path = ITEM1_SUM_DIR / f"{ticker}.json"
        if raw_path.exists() and sum_path.exists():
            continue

        # 3) 최신 10-K 메타데이터
        meta = get_latest_10k_metadata(cik)
        if meta is None:
            print(f"[MISS 10-K] {ticker} {name} (CIK={cik})")
            miss_list.append({"ticker": ticker, "name": name, "cik": cik, "reason": "no_10k"})
            continue

        # 4) 10-K HTML 다운로드 + soup 생성
        try:
            html = fetch_10k_html(cik, meta["accession"], meta["primary_doc"])
        except Exception as e:
            print(f"[ERR GET HTML] {ticker} {name}: {e}")
            miss_list.append({"ticker": ticker, "name": name, "cik": cik, "reason": "html_error"})
            continue

        soup = BeautifulSoup(html, "html.parser")
        full_text = soup.get_text("\n")


        # 5) Item 1. Business 추출
        item1_text = extract_item1_business(full_text, min_chars=400)

        if not item1_text.strip():
            print(f"[MISS ITEM1] {ticker} {name}")
            miss_list.append({"ticker": ticker, "name": name, "cik": cik, "reason": "no_item1"})
            continue

        # 6) raw JSON 저장
        raw_data = {
            "ticker": ticker,
            "cik": cik,
            "company_name": meta["company_name"],
            "filing_date": meta["filing_date"],
            "accession": meta["accession"],
            "primary_doc": meta["primary_doc"],
            "item1_text": item1_text,
        }
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        # 7) 요약 생성 + 저장
        summary_text = summarize_item1(item1_text, summarizer)
        sum_data = {
            "ticker": ticker,
            "cik": cik,
            "company_name": meta["company_name"],
            "filing_date": meta["filing_date"],
            "item1_summary": summary_text,
        }
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(sum_data, f, ensure_ascii=False, indent=2)

    # 8) miss 리스트 저장
    miss_path = SEC_DIR / "_item1_miss_list.json"
    with open(miss_path, "w", encoding="utf-8") as f:
        json.dump(miss_list, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved miss list to {miss_path}")


if __name__ == "__main__":
    main()