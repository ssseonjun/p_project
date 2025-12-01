"""
S&P 500 전체에 대해:
1) 위키피디아에서 Symbol / Security / CIK 리스트 가져오기
2) SEC companyfacts로 2019~2023 재무/현금흐름 숫자 추출 → {SYMBOL}_numerical.json
3) SEC 10-K 텍스트에서
   - 2019~2023 각 연도의 Item 1. Business 파트 추출
   → {SYMBOL}_business.json (business_by_year에 연도별로 저장)

필요 패키지:
    pip install requests pandas beautifulsoup4
"""

import requests
import pandas as pd
import time
import json
from io import StringIO
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import re

# -----------------------------------------------------
# 전역 설정
# -----------------------------------------------------

YEARS = [2019, 2020, 2021, 2022, 2023]
MIN_SECTION_LENGTH = 500

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "file" / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------
# 0. S&P500 CIK 리스트 가져오기 (위키)
# -----------------------------------------------------

def load_sp500_ciks_from_wikipedia(user_agent: str) -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": user_agent}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))

    target = None
    for t in tables:
        if "CIK" in t.columns:
            target = t
            break

    if target is None:
        all_cols = [tbl.columns.tolist() for tbl in tables]
        raise ValueError(f"CIK 컬럼을 찾지 못했습니다. tables columns: {all_cols}")

    df = target.copy()

    symbol_col_candidates = ["Symbol", "Ticker"]
    name_col_candidates = ["Security", "Company", "Name"]

    symbol_col = next((c for c in symbol_col_candidates if c in df.columns), None)
    name_col = next((c for c in name_col_candidates if c in df.columns), None)

    if symbol_col is None or name_col is None:
        raise ValueError(
            f"Symbol 또는 이름 컬럼을 찾지 못했습니다. 현재 columns: {df.columns.tolist()}"
        )

    out = df[[symbol_col, name_col, "CIK"]].copy()
    out = out.rename(columns={symbol_col: "Symbol", name_col: "Security"})

    out["CIK"] = out["CIK"].astype(str).str.strip()
    out = out[out["CIK"].str.match(r"^\d+$")]
    out["CIK"] = out["CIK"].str.zfill(10)

    return out

# -----------------------------------------------------
# 1. XBRL 태그 후보 정의 (재무제표 + 현금흐름)
# -----------------------------------------------------

TAG_CANDIDATES: Dict[str, List[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "NetIncomeLoss",
    ],
    "rnd_expense": [
        "ResearchAndDevelopmentExpense",
    ],
    "sga_expense": [
        "SellingGeneralAndAdministrativeExpense",
    ],
    "interest_expense": [
        "InterestExpense",
    ],
    "income_tax_expense": [
        "IncomeTaxExpenseBenefit",
        "IncomeTaxExpenseBenefitContinuingOperations",
    ],

    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "total_equity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "cash_and_equiv": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "short_term_debt": [
        "DebtCurrent",
        "LongTermDebtCurrent",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
    ],
    "inventory": [
        "InventoryNet",
    ],
    "ppe": [
        "PropertyPlantAndEquipmentNet",
    ],
    "goodwill": [
        "Goodwill",
    ],

    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "cfi": [
        "NetCashProvidedByUsedInInvestingActivities",
    ],
    "cff": [
        "NetCashProvidedByUsedInFinancingActivities",
    ],
    "depr_amort": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "dividends": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ],
    "share_repurchases": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
}

ALLOWED_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}

# -----------------------------------------------------
# 2. SEC companyfacts → numerical json
# -----------------------------------------------------

def fetch_companyfacts(cik_10digit: str, user_agent: str) -> Dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_10digit}.json"
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _pick_best_fact_for_year(fact_list: List[Dict], year: int) -> Optional[float]:
    candidates = []
    for item in fact_list:
        end = item.get("end")
        form = item.get("form")
        fy = item.get("fy")
        val = item.get("val")
        filed = item.get("filed")

        year_ok = False
        if fy is not None and str(fy).isdigit() and int(fy) == year:
            year_ok = True
        elif end:
            try:
                end_year = int(end[:4])
                if end_year == year:
                    year_ok = True
            except Exception:
                pass

        if not year_ok:
            continue
        if form and form not in ALLOWED_FORMS:
            continue

        try:
            val_float = float(val)
        except Exception:
            continue

        filed_dt = None
        if filed:
            try:
                filed_dt = datetime.strptime(filed, "%Y-%m-%d")
            except Exception:
                filed_dt = None

        candidates.append((filed_dt, val_float))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0] is None, x[0]))
    _, val_float = candidates[-1]
    return val_float


def get_value_for_key_and_year(facts_us_gaap: Dict, key: str, year: int) -> Optional[float]:
    tag_candidates = TAG_CANDIDATES.get(key, [])
    for tag in tag_candidates:
        tag_fact = facts_us_gaap.get(tag)
        if not tag_fact:
            continue
        units = tag_fact.get("units", {})
        usd_list = units.get("USD")
        if not usd_list:
            continue

        val = _pick_best_fact_for_year(usd_list, year)
        if val is not None:
            return val
    return None


def extract_yearly_facts(companyfacts_json: Dict,
                         years: List[int] = YEARS) -> Dict[int, Dict[str, float]]:
    facts = companyfacts_json.get("facts", {})
    facts_us_gaap = facts.get("us-gaap", {})

    yearly_data: Dict[int, Dict[str, float]] = {}
    for y in years:
        fs_dict: Dict[str, float] = {}
        for key in TAG_CANDIDATES.keys():
            v = get_value_for_key_and_year(facts_us_gaap, key, y)
            if v is not None:
                fs_dict[key] = v
        yearly_data[y] = fs_dict
    return yearly_data


def save_numerical_json(symbol: str, name: str, cik: str,
                        yearly_fs: Dict[int, Dict[str, float]]):
    data = {
        "symbol": symbol,
        "name": name,
        "cik": cik,
        "yearly_fs": yearly_fs
    }
    out_path = JSON_DIR / f"{symbol}_numerical.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [NUM] Saved numerical JSON -> {out_path}")

# -----------------------------------------------------
# 3. 10-K 텍스트 다운로드 + 텍스트 변환
# -----------------------------------------------------

def get_annual_filings_by_year(cik_10digit: str,
                               user_agent: str,
                               years: List[int]) -> Dict[int, Dict]:
    """
    submissions/CIKxxxx.json에서 최근 filings 중 연간보고서(10-K/20-F/40-F 등)를 보고
    reportDate(없으면 filingDate)의 연도가 years에 포함되면 그 해의 filing으로 매핑.

    return: { year: {form, filing_date, report_date, filing_url, company_name, ...}, ... }
    """
    url = f"https://data.sec.gov/submissions/CIK{cik_10digit}.json"
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    filings_recent = data.get("filings", {}).get("recent", {})
    forms = filings_recent.get("form", [])
    accession_numbers = filings_recent.get("accessionNumber", [])
    primary_docs = filings_recent.get("primaryDocument", [])
    filing_dates = filings_recent.get("filingDate", [])
    report_dates = filings_recent.get("reportDate", [None] * len(forms))

    result: Dict[int, Dict] = {}
    company_name = data.get("name", "")

    for form, acc_no, primary_doc, fdate, rdate in zip(
        forms, accession_numbers, primary_docs, filing_dates, report_dates
    ):
        if form not in ALLOWED_FORMS:
            continue

        year = None
        if rdate and len(rdate) >= 4 and rdate[:4].isdigit():
            year = int(rdate[:4])
        elif fdate and len(fdate) >= 4 and fdate[:4].isdigit():
            year = int(fdate[:4])

        if year is None or year not in years:
            continue
        if year in result:
            # 이미 해당 연도의 연간 보고서가 매핑되어 있으면 스킵 (가장 최근 filings만 사용)
            continue

        acc_no_nodash = acc_no.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik_10digit)}/{acc_no_nodash}/{primary_doc}"
        )
        result[year] = {
            "form": form,
            "accession_number": acc_no,
            "filing_date": fdate,
            "report_date": rdate,
            "primary_document": primary_doc,
            "filing_url": filing_url,
            "company_name": company_name,
        }

    return result


def download_filing_html(url: str, user_agent: str) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    time.sleep(0.2)
    return resp.text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

# -----------------------------------------------------
# 4. Item 1. Business 섹션 추출
# -----------------------------------------------------

def extract_item1_business(text: str) -> str:
    """
    Item 1. Business 섹션 추출 (단순 버전)

    1) "Item 1 ... Business" 패턴을 찾고
    2) 그 지점부터 "Item 1A" 또는 "Item 2" 전까지 잘라서 반환.
    """
    lower = text.lower()

    start_re = re.compile(
        r"item\s*1[\s\S]{0,100}?business",
        flags=re.IGNORECASE
    )

    end_res = [
        re.compile(r"item\s*1a[\s\S]{0,50}?risk", flags=re.IGNORECASE),
        re.compile(r"item\s*2[\s\S]{0,50}?properties", flags=re.IGNORECASE),
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
            end_idx = len(text)

        section = text[start_idx:end_idx].strip()
        if len(section) >= MIN_SECTION_LENGTH:
            candidates.append(section)

    if candidates:
        return candidates[0]
    return ""

# -----------------------------------------------------
# 5. 사업 텍스트 JSON 저장 (연도별)
# -----------------------------------------------------

def save_business_json(symbol: str,
                       cik: str,
                       company_name: str,
                       business_by_year: Dict[str, Dict]):
    """
    business_by_year: {
      "2019": {
        "filing_type": "...",
        "filing_date": "...",
        "report_date": "...",
        "filing_url": "...",
        "business_and_revenue": "..."
      },
      ...
    }
    """
    data = {
        "symbol": symbol,
        "cik": cik,
        "company_name": company_name,
        "business_by_year": business_by_year,
    }
    out_path = JSON_DIR / f"{symbol}_business.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [TXT] Saved business JSON -> {out_path}")

# -----------------------------------------------------
# 6. 메인 루프
# -----------------------------------------------------

def process_sp500_companies(user_agent: str,
                            sleep_sec: float = 0.5,
                            max_companies: Optional[int] = None):
    df = load_sp500_ciks_from_wikipedia(user_agent)
    print(f"[INFO] Loaded S&P 500 from Wikipedia: {len(df)} rows")

    rows = df.to_dict("records")
    df.loc[df['Symbol'] == 'BLK', 'CIK'] = "0001364742"
    rows = df[df['Symbol']=='BLK'].to_dict("records")
    if max_companies is not None:
        rows = rows[:max_companies]

    error_log = {}

    for idx, row in enumerate(rows, start=1):
        symbol = row["Symbol"].upper()
        name = row["Security"]
        cik = row["CIK"]

        print(f"\n[{idx}/{len(rows)}] Processing {symbol} ({name}), CIK={cik}")

        # 1) numerical (companyfacts)
        try:
            companyfacts = fetch_companyfacts(cik, user_agent=user_agent)
            yearly_fs = extract_yearly_facts(companyfacts, YEARS)
            save_numerical_json(symbol, name, cik, yearly_fs)
        except Exception as e:
            print(f"  [ERROR] numerical for {symbol}: {e}")
            error_log.setdefault(symbol, {})["numerical"] = str(e)

        # 2) text (2019~2023 각 연도 10-K/20-F → Item 1. Business)
        try:
            filings_by_year = get_annual_filings_by_year(cik, user_agent, YEARS)

            business_by_year: Dict[str, Dict] = {}
            company_name_for_text: Optional[str] = None

            for year in YEARS:
                meta = filings_by_year.get(year)
                if not meta:
                    continue  # 해당 연도 보고서 없으면 패스

                html_10k = download_filing_html(meta["filing_url"], user_agent)
                text_10k = html_to_text(html_10k)
                business_section = extract_item1_business(text_10k)

                business_by_year[str(year)] = {
                    "filing_type": meta.get("form"),
                    "filing_date": meta.get("filing_date"),
                    "report_date": meta.get("report_date"),
                    "filing_url": meta.get("filing_url"),
                    "business_and_revenue": business_section,
                }

                if not company_name_for_text:
                    company_name_for_text = meta.get("company_name") or name

                time.sleep(0.2)  # SEC 매너 타임

            if business_by_year:
                save_business_json(symbol, cik,
                                   company_name_for_text or name,
                                   business_by_year)
            else:
                raise RuntimeError("No business sections found for 2019–2023")

        except Exception as e:
            print(f"  [ERROR] business text for {symbol}: {e}")
            error_log.setdefault(symbol, {})["business_text"] = str(e)

        time.sleep(sleep_sec)

    log_path = JSON_DIR / "sp500_extraction_error_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(error_log, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Done. Error log saved to {log_path}")


if __name__ == "__main__":
    USER_AGENT = "SEONJUN qkrtjswns@gachon.ac.kr"

    # 테스트: 앞 3개만
    # process_sp500_companies(USER_AGENT, sleep_sec=0.5, max_companies=3)

    # 전체 실행하려면 아래 주석 해제
    process_sp500_companies(USER_AGENT, sleep_sec=0.5, max_companies=None)