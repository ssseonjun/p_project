"""
S&P 500 기업 리스트를 위키백과에서 가져오고,
SEC companyfacts API를 호출해서 2019~2023 재무제표를 정리한 후

    financial_json/{TICKER}.json

으로 기업별 JSON을 생성하는 스크립트.

각 JSON 구조 예시:

{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "cik": "0000320193",
  "years": [2019, 2020, 2021, 2022, 2023],
  "yearly_fs": {
    "2019": { "revenue": ..., "net_income": ..., ... },
    "2020": { ... },
    ...
  }
}

이 JSON은 h_fs 계산 코드의 compute_year_features / build_company_year_feature_matrix와
호환되도록 키 이름을 맞춰두었다.

필요 패키지:
    pip install pandas requests tqdm
"""

import json
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# 공통 설정
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

FIN_JSON_DIR = DATA_DIR / "json/financial_json"
FIN_JSON_DIR.mkdir(parents=True, exist_ok=True)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# h_fs 코드랑 맞추려면 여기만 바꿔주면 됨 (ex. [2020,2021,2022,2023,2024])
YEARS = [2020,2021,2022,2023,2024]

# SEC 권장: User-Agent에 본인 이름/메일 명시
SEC_HEADERS = {
    "User-Agent": "Seonjun qkrtjswns@gachon.ac.kr",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

COMPANY_TICKERS_JSON = DATA_DIR / "json/tickers.json"  # SEC 매핑 캐시


# -----------------------------
# 1. S&P500 리스트 (403 방지 버전)
# -----------------------------

def get_sp500_companies() -> pd.DataFrame:
    """
    위키 S&P500 페이지를 requests로 가져와서 read_html에 직접 넘긴다.
    (User-Agent를 지정해 403 방지)
    return: DataFrame[ticker, name]
    """
    resp = requests.get(
        SP500_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; sp500-wiki-crawler/0.1)"},
        timeout=10,
    )
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))

    target = None
    for t in tables:
        cols = [c[0] if isinstance(c, tuple) else c for c in t.columns]
        if "Symbol" in cols and "Security" in cols:
            t.columns = cols
            target = t
            break

    if target is None:
        raise ValueError(f"S&P 500 테이블을 찾지 못했습니다. 테이블 수: {len(tables)}")

    df = target[["Symbol", "Security"]].rename(
        columns={"Symbol": "ticker", "Security": "name"}
    )
    return df


# -----------------------------
# 2. SEC company_tickers.json
# -----------------------------

def download_company_tickers_json():
    """
    SEC가 제공하는 company_tickers.json을 한 번 다운받아 로컬에 저장.
    """
    if COMPANY_TICKERS_JSON.exists():
        return
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=SEC_HEADERS)
    r.raise_for_status()
    COMPANY_TICKERS_JSON.write_bytes(r.content)
    print(f"[INFO] Downloaded company_tickers.json -> {COMPANY_TICKERS_JSON}")


def load_ticker_maps():
    """
    company_tickers.json을 로드해서
      by_ticker: TICKER -> info
      by_title : COMPANY TITLE -> info
    두 개 딕셔너리로 반환.
    """
    download_company_tickers_json()
    with open(COMPANY_TICKERS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_ticker: Dict[str, Dict] = {}
    by_title: Dict[str, Dict] = {}
    for _, v in data.items():
        t = v["ticker"].upper()
        title = v["title"]
        by_ticker[t] = v
        by_title[title.upper()] = v
    return by_ticker, by_title


def get_cik_for_company(
    ticker: str,
    name: str,
    by_ticker: Dict[str, Dict],
    by_title: Dict[str, Dict],
) -> Optional[str]:
    """
    ticker 기반으로 먼저 찾고, 안 되면 회사명으로 fallback.
    Meta처럼 티커/이름이 헷갈리는 케이스를 최대한 커버.
    return: 10자리 CIK 문자열 또는 None
    """
    t_norm = ticker.upper().replace(".", "-")
    if t_norm in by_ticker:
        return f'{int(by_ticker[t_norm]["cik_str"]):010d}'

    # 회사명으로 찾기 (완전 일치 우선)
    name_norm = name.upper()
    if name_norm in by_title:
        return f'{int(by_title[name_norm]["cik_str"]):010d}'

    # Inc., Corp. 꼬리 제거해서 다시 시도
    import re
    simple_name = re.sub(r",? (INCORPORATED|INC\.?|CORPORATION|CORP\.?|COMPANY|CO\.?)$", "", name_norm)
    for title, info in by_title.items():
        if simple_name == title:
            return f'{int(info["cik_str"]):010d}'

    return None


# -----------------------------
# 3. SEC companyfacts 호출
# -----------------------------

def fetch_companyfacts(cik: str) -> Optional[Dict]:
    """
    https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS)
    if r.status_code == 403:
        print(f"[WARN] 403 Forbidden on companyfacts for CIK={cik} (rate limit? User-Agent?)")
        return None
    if r.status_code != 200:
        print(f"[WARN] companyfacts fetch failed for CIK={cik}: {r.status_code}")
        return None
    return r.json()


def pick_yearly_fact(
    facts: Dict,
    concept_candidates: List[str],
    years: List[int],
    currency: str = "USD",
) -> Dict[int, Optional[float]]:
    """
    companyfacts JSON에서 us-gaap 특정 개념 후보들 중 하나를 골라
    연도별 (FY) 값을 dict로 반환.

    return: {2019: val or None, 2020: ..., ...}
    """
    result: Dict[int, Optional[float]] = {y: None for y in years}
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    concept_data = None
    for cname in concept_candidates:
        if cname in us_gaap:
            concept_data = us_gaap[cname]
            break
    if concept_data is None:
        return result

    units = concept_data.get("units", {}).get(currency, [])
    # units: [{"end": "2020-12-31", "val": ..., "fp": "FY", "form": "10-K", ...}, ...]

    for entry in units:
        try:
            if entry.get("fp") != "FY":
                continue
            form = entry.get("form", "")
            if form not in ("10-K", "20-F"):
                continue
            end = entry.get("end", "")
            year = int(end[:4])
        except Exception:
            continue

        if year not in years:
            continue

        result[year] = entry.get("val")

    return result


# -----------------------------
# 4. main: 기업별로 financial_json/{ticker}.json 저장
# -----------------------------

def main():
    print("[INFO] Loading S&P 500 list from Wikipedia...")
    sp500_df = get_sp500_companies()
    print(f"[INFO] Total S&P500 companies: {len(sp500_df)}")

    print("[INFO] Loading SEC ticker -> CIK mapping...")
    by_ticker, by_title = load_ticker_maps()

    # compute_year_features에서 쓰는 키와 맞추기
    concept_map = {
        "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
        "gross_profit": ["GrossProfit"],
        "operating_income": ["OperatingIncomeLoss"],
        "net_income": ["NetIncomeLoss"],
        "rnd_expense": ["ResearchAndDevelopmentExpense"],
        "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
        "interest_expense": ["InterestExpense"],
        "income_tax_expense": ["IncomeTaxExpenseBenefit"],

        "total_assets": ["Assets"],
        "total_liabilities": ["Liabilities"],
        "total_equity": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "current_assets": ["AssetsCurrent"],
        "current_liabilities": ["LiabilitiesCurrent"],
        "cash_and_equiv": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashAndCashEquivalentsPeriodIncreaseDecrease",
        ],
        "short_term_debt": ["DebtCurrent"],
        "long_term_debt": ["LongTermDebtNoncurrent"],

        "accounts_receivable": ["AccountsReceivableNetCurrent", "AccountsReceivableNet"],
        "inventory": ["InventoryNet"],
        "ppe": ["PropertyPlantAndEquipmentNet"],
        "goodwill": ["Goodwill"],

        "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
        "cfi": ["NetCashProvidedByUsedInInvestingActivities"],
        "cff": ["NetCashProvidedByUsedInFinancingActivities"],
        "depr_amort": ["DepreciationDepletionAndAmortization"],
        "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
        "dividends": ["PaymentsOfDividends"],
        "share_repurchases": ["PaymentsForRepurchaseOfCommonStock"],
    }

    miss_list: List[Dict] = []

    for _, row in tqdm(sp500_df.iterrows(), total=len(sp500_df), desc="SEC financials"):
        ticker = row["ticker"]
        name = row["name"]

        out_path = FIN_JSON_DIR / f"{ticker}.json"
        # 이미 만들어져 있으면 스킵 (재실행 시)
        if out_path.exists():
            continue

        cik = get_cik_for_company(ticker, name, by_ticker, by_title)
        if cik is None:
            print(f"[MISS CIK] {ticker} {name}")
            miss_list.append({"ticker": ticker, "name": name, "reason": "no_cik"})
            continue

        facts = fetch_companyfacts(cik)
        if facts is None:
            miss_list.append({"ticker": ticker, "name": name, "reason": "no_companyfacts"})
            continue

        # 연도별 fs dict 초기화
        yearly_fs: Dict[str, Dict[str, float]] = {str(y): {} for y in YEARS}

        # 각 key마다 연도별 값 채우기
        for key, candidates in concept_map.items():
            values_by_year = pick_yearly_fact(facts, candidates, YEARS)
            for y, val in values_by_year.items():
                if val is None:
                    continue
                yearly_fs[str(y)][key] = float(val)

        data = {
            "symbol": ticker,
            "name": name,
            "cik": cik,
            "years": YEARS,
            "yearly_fs": yearly_fs,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 못 가져온 기업 로그
    if miss_list:
        miss_path = DATA_DIR / "financial_miss_list.json"
        with open(miss_path, "w", encoding="utf-8") as f:
            json.dump(miss_list, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Some companies were missed. Details -> {miss_path}")

    print("[DONE] 기업별 재무 JSON 생성 완료. 저장 경로:", FIN_JSON_DIR)


if __name__ == "__main__":
    main()