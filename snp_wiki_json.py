# snp_wiki_json.py
"""
S&P500 위키 테이블에서
 - ticker, name, CIK를 읽어서 csv/sp500_list.csv 생성
 - 각 기업의 위키 페이지 텍스트를 가져와 전처리 후 json/wiki_json/{TICKER}.json 저장

1) 위키 'List of S&P 500 companies' 페이지의 테이블에는 이미 CIK 컬럼이 있음.
2) Symbol, Security, CIK만 뽑아서 sp500_list.csv(index, ticker, name, cik)로 저장.
3) 각 기업별로:
   - NON_RESULTS_NAME_MAP을 통해 회사명 보정
   - wikipedia-api로 페이지 텍스트 가져오기
   - 텍스트 전처리 (각주 제거, tail section 제거 등)
   - json/wiki_json/{TICKER}.json 에 저장.

필요 패키지:
    pip install pandas requests wikipedia-api beautifulsoup4
"""

import json
import math
import re
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import wikipediaapi

# -----------------------------
# 경로 설정
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "json"
WIKI_JSON_DIR = DATA_DIR / "wiki_json"
WIKI_JSON_DIR.mkdir(parents=True, exist_ok=True)

SP500_LIST_CSV = CSV_DIR / "sp500_list.csv"

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# 컷오프 섹션 (여기 제목 나오면 이후는 버림)
CUTOFF_HEADINGS = {
    "see also",
    "references",
    "official website",
}

# 기존 non-results 대응용 이름 매핑
NON_RESULTS_NAME_MAP: Dict[str, str] = {
    # 클래스 구분 계열
    "Alphabet Inc. (Class A)": "Alphabet Inc.",
    "Alphabet Inc. (Class C)": "Alphabet Inc.",
    "Fox Corporation (Class A)": "Fox Corporation",
    "Fox Corporation (Class B)": "Fox Corporation",
    "News Corp (Class A)": "News Corp",
    "News Corp (Class B)": "News Corp",

    # (The) 꼬리표 붙은 것들
    "Campbell's Company (The)": "Campbell Soup Company",
    "Coca-Cola Company (The)": "The Coca-Cola Company",
    "Cooper Companies (The)": "The Cooper Companies",
    "Estée Lauder Companies (The)": "The Estée Lauder Companies",
    "Hartford (The)": "The Hartford",
    "Hershey Company (The)": "The Hershey Company",
    "Home Depot (The)": "The Home Depot",
    "Interpublic Group of Companies (The)": "The Interpublic Group of Companies",
    "Mosaic Company (The)": "The Mosaic Company",
    "J.M. Smucker Company (The)": "The J. M. Smucker Company",
    "Trade Desk (The)": "The Trade Desk",
    "Travelers Companies (The)": "The Travelers Companies",
    "Walt Disney Company (The)": "The Walt Disney Company",

    # 약식/다른 공식명
    "Lilly (Eli)": "Eli Lilly and Company",
    "O’Reilly Automotive": "O'Reilly Auto Parts",

    # 예외 케이스들
    "Qnity Electronics": "Qnity Electronics, Inc.",
    "Insulet Corporation": "Insulet Corporation",
}

# wikipedia-api 설정
wiki = wikipediaapi.Wikipedia(
    user_agent="graduation_project (qkrtjswns@gachon.ac.kr)",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)


# -----------------------------
# 1. S&P500 리스트 (Symbol, Security, CIK) 읽기
# -----------------------------
def get_sp500_from_wiki() -> pd.DataFrame:
    """
    위키 S&P500 컴포넌트 테이블에서
    Symbol, Security, (있다면) CIK 컬럼을 읽어 DataFrame 반환.
    """
    resp = requests.get(
        SP500_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; sp500-wiki-crawler/0.1)"},
        timeout=15,
    )
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise ValueError("위키에서 테이블을 하나도 찾지 못했습니다.")

    target = None

    # 여러 개 테이블 중에서 'Symbol'과 'Security'가 있는 걸 찾아서 사용
    for t in tables:
        # MultiIndex일 수도 있어서 첫 레벨만 사용
        cols = [c[0] if isinstance(c, tuple) else c for c in t.columns]
        # 문자열로 캐스팅
        cols = [str(c) for c in cols]

        if "Symbol" in cols and "Security" in cols:
            t.columns = cols
            target = t
            break

    if target is None:
        raise ValueError(
            f"S&P500 테이블( Symbol, Security 포함 )을 찾지 못했습니다. "
            f"현재 테이블 수: {len(tables)}"
        )

    # 기본은 Symbol / Security만
    df = target[["Symbol", "Security"]].copy()

    # 위키 테이블에 CIK 컬럼이 있으면 같이 사용, 없으면 None으로 채우기
    if "CIK" in target.columns:
        df["CIK"] = target["CIK"]
    else:
        df["CIK"] = None

    return df


def save_sp500_csv(df: pd.DataFrame):
    """
    위키에서 읽은 DataFrame(Symbol, Security, CIK)을
    index, ticker, name, cik 포맷으로 csv/sp500_list.csv에 저장.
    """
    out_rows = []
    for idx, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        name = str(row["Security"]).strip()
        cik_val = row["CIK"]

        if pd.isna(cik_val):
            cik_str = ""
        else:
            # 위키 CIK는 int일 수 있으니 10자리 zero-padding
            try:
                cik_str = f"{int(cik_val):010d}"
            except Exception:
                cik_str = str(cik_val).strip()

        out_rows.append(
            {
                "index": idx,
                "ticker": symbol,
                "name": name,
                "cik": cik_str,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(SP500_LIST_CSV, index=False)
    print(f"[INFO] Saved S&P500 list with CIK to {SP500_LIST_CSV}")


# -----------------------------
# 2. 위키 텍스트 가져오기 + 전처리
# -----------------------------
def get_corrected_company_name(original_name: str) -> str:
    """
    S&P 500 Security 이름을 기준으로,
    NON_RESULTS_NAME_MAP에 있으면 value로 교체하고,
    없으면 원래 이름 그대로 반환.
    """
    return NON_RESULTS_NAME_MAP.get(original_name, original_name)


def fetch_wiki_text(title_base: str) -> Tuple[str, str, bool]:
    """
    title_base를 기준으로 위키 문서를 찾는다.
      - 1차: title_base
      - 2차: title_base + " (company)"
    성공: (텍스트, 사용한 제목, True)
    실패: ("", "", False)
    """
    candidates = [title_base, f"{title_base} (company)"]
    tried = set()

    for title in candidates:
        if title in tried:
            continue
        tried.add(title)

        page = wiki.page(title)
        if page.exists():
            return page.text, title, True

    return "", "", False


def strip_tail_sections(text: str) -> str:
    """See also / References / External links 이후는 모두 제거"""
    kept_lines: List[str] = []

    if isinstance(text, float):
        if math.isnan(text):
            return ""
        text = str(text)

    lines = str(text).splitlines()
    for line in lines:
        title = line.strip().lower()
        # 완전 일치 기준 (간단하게)
        if title in CUTOFF_HEADINGS:
            break
        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def basic_cleanup(text: str) -> str:
    """각주/공백 정리 등 기본 클리닝"""
    text = re.sub(r"\[\d+\]", " ", text)          # [1], [23] 같은 각주
    text = text.replace("[citation needed]", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_company_wiki(text: str) -> str:
    """위키 텍스트 전체 전처리"""
    text = strip_tail_sections(text)
    text = basic_cleanup(text)
    return text


# -----------------------------
# 3. main: CSV + 기업별 JSON 생성
# -----------------------------
def main():
    print("[INFO] Fetching S&P500 list (with CIK) from Wikipedia...")
    df_sp500 = get_sp500_from_wiki()
    print(f"[INFO] Loaded S&P500 table from wiki: {len(df_sp500)} rows")

    # 1) CSV 저장 (index, ticker, name, cik)
    save_sp500_csv(df_sp500)

    results = []
    not_found = []

    for idx, row in df_sp500.iterrows():
        symbol = str(row["Symbol"]).strip()
        company_name = str(row["Security"]).strip()
        cik_val = row["CIK"]

        if pd.isna(cik_val):
            cik_str = ""
        else:
            try:
                cik_str = f"{int(cik_val):010d}"
            except Exception:
                cik_str = str(cik_val).strip()

        corrected_name = get_corrected_company_name(company_name)

        print(
            f"[{idx+1}/{len(df_sp500)}] {symbol} - "
            f"원래: '{company_name}' / 검색용: '{corrected_name}' / CIK={cik_str}"
        )

        text, used_title, exists = fetch_wiki_text(corrected_name)

        if not exists:
            print("  -> 위키 문서 없음")
            not_found.append(
                {
                    "ticker": symbol,
                    "name": company_name,
                    "cik": cik_str,
                    "corrected_name": corrected_name,
                }
            )
            continue

        print(f"  -> 문서 찾음: '{used_title}' (텍스트 길이: {len(text)}자)")

        clean_text = preprocess_company_wiki(text)

        # 기업별 JSON 저장
        out_path = WIKI_JSON_DIR / f"{symbol}.json"
        data = {
            "ticker": symbol,
            "cik": cik_str,
            "company_name": company_name,
            "corrected_name": corrected_name,
            "wiki_title_used": used_title,
            "exists": True,
            "raw_text": text,
            "clean_text": clean_text,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        results.append(data)

        # 서버 부담 줄이기
        time.sleep(0.3)

    # 못 찾은 기업 리스트 저장
    nf_path = WIKI_JSON_DIR / "_wiki_not_found.json"
    with open(nf_path, "w", encoding="utf-8") as f:
        json.dump(not_found, f, ensure_ascii=False, indent=2)

    print()
    print(f"[INFO] 위키 문서를 찾지 못한 기업 수: {len(not_found)}")
    if not_found:
        missed_names = [x["name"] for x in not_found]
        print("기업 리스트:", missed_names)
    print(f"[INFO] Saved per-ticker wiki JSONs to {WIKI_JSON_DIR}")


if __name__ == "__main__":
    main()
