# embed_financial.py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import chromadb

BASE_DIR = Path(__file__).resolve().parent
NUM_DIR = BASE_DIR / "json" / "financial_json"   # <- 재무 JSON 디렉토리
CHROMA_PATH = BASE_DIR / "chromaDB" / "chroma_fs"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

# ---------- 유틸 ----------
def safe_div(a, b):
    try:
        if a is None or b is None: return None
        if b == 0: return None
        return float(a) / float(b)
    except Exception:
        return None

def year_keys(data: Dict) -> List[int]:
    ys = []
    for k in data.keys():
        ks = str(k)
        if ks.isdigit() and len(ks) == 4:
            ys.append(int(ks))
    return sorted(ys)

def take_last_n_years(ys: List[int], n: int = 5) -> List[int]:
    return sorted(ys)[-n:]

def list_mean_std_slope(vals: List[Optional[float]]) -> Tuple[float, float, float]:
    """5개년 값에서 평균/표준편차/기울기(연도 인덱스에 대한 선형회귀)를 반환. 결측은 제외."""
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    # slope: 값 vs. [0..k-1]
    if arr.size >= 2:
        x = np.arange(arr.size, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])  # 1차 회귀 기울기
    else:
        slope = 0.0
    return mean, std, slope

def cagr(first: Optional[float], last: Optional[float], years: int) -> float:
    if first is None or last is None or years <= 0: return 0.0
    if first <= 0 or last <= 0: return 0.0
    try:
        return float((last / first) ** (1.0 / years) - 1.0)
    except Exception:
        return 0.0

# ---------- 1개 회사: 5개년 특징 생성 ----------
def build_features_5y(fs_by_year: Dict[int, Dict[str, float]], years_5: List[int]) -> Tuple[List[float], List[str], Dict[str, float]]:
    """
    fs_by_year: {year: {field: value}}
    years_5: 최근 5개년 (부족하면 있는 만큼만)
    """
    # 연도 순서대로 값 꺼내기
    def seq(field: str) -> List[Optional[float]]:
        return [fs_by_year.get(y, {}).get(field) for y in years_5]

    # 기본 시퀀스
    revenue   = seq("revenue")
    cogs      = seq("cogs")
    op_inc    = seq("operating_income")
    net_inc   = seq("net_income")
    sga       = seq("sga_expense")
    tax_exp   = seq("income_tax_expense")
    tot_assets= seq("total_assets")
    tot_liab  = seq("total_liabilities")
    tot_equity= seq("total_equity")
    cur_assets= seq("current_assets")
    cur_liab  = seq("current_liabilities")
    cash      = seq("cash_and_equiv")
    ppe       = seq("ppe")
    goodwill  = seq("goodwill")
    cfo       = seq("cfo")
    cfi       = seq("cfi")
    capex     = seq("capex")
    repurch   = seq("share_repurchases")

    # 파생 시퀀스
    # 수익성
    net_margin = [safe_div(ni, rv) for ni, rv in zip(net_inc, revenue)]       # 순이익률
    op_margin  = [safe_div(oi, rv) for oi, rv in zip(op_inc, revenue)]        # 영업이익률
    roa        = [safe_div(ni, ta) for ni, ta in zip(net_inc, tot_assets)]    # ROA
    roe        = [safe_div(ni, te) for ni, te in zip(net_inc, tot_equity)]    # ROE

    # 유동성/레버리지
    current_ratio = [safe_div(ca, cl) for ca, cl in zip(cur_assets, cur_liab)]
    cash_ratio    = [safe_div(ca2, cl) for ca2, cl in zip(cash, cur_liab)]
    de_ratio      = [safe_div(tl, te) for tl, te in zip(tot_liab, tot_equity)]  # 부채/자본

    # 자산 구성
    goodwill_ratio = [safe_div(gw, ta) for gw, ta in zip(goodwill, tot_assets)]
    ppe_ratio      = [safe_div(pp, ta) for pp, ta in zip(ppe, tot_assets)]

    # 현금흐름/투자성향
    cfo_margin   = [safe_div(cf, rv) for cf, rv in zip(cfo, revenue)]  # CFO/매출
    fcf          = [(cf - (cx or 0.0)) if (cf is not None) else None for cf, cx in zip(cfo, capex)]
    fcf_assets   = [safe_div(f, ta) for f, ta in zip(fcf, tot_assets)] # FCF/자산
    cfo_to_ni    = [safe_div(cf, ni) for cf, ni in zip(cfo, net_inc)]  # 이익의 현금화
    capex_to_cfo = [safe_div(cx, cf) for cx, cf in zip(capex, cfo)]    # Capex/CFO

    # 5개년 summary: 평균/표준편차/추세
    metrics = {
        "net_margin": net_margin,
        "op_margin": op_margin,
        "roa": roa,
        "roe": roe,
        "current_ratio": current_ratio,
        "cash_ratio": cash_ratio,
        "de_ratio": de_ratio,
        "goodwill_assets": goodwill_ratio,
        "ppe_assets": ppe_ratio,
        "cfo_margin": cfo_margin,
        "fcf_assets": fcf_assets,
        "cfo_to_netincome": cfo_to_ni,
        "capex_to_cfo": capex_to_cfo,
    }

    vec: List[float] = []
    names: List[str] = []

    for name, series in metrics.items():
        mean, std, slope = list_mean_std_slope(series)
        vec += [mean, std, slope]
        names += [f"{name}_mean", f"{name}_std", f"{name}_slope"]

    # 성장(CAGR) 3종: 매출/순이익/CFO (연수: 실제 사용 연도 수 - 1)
    years_span = max(1, len(years_5) - 1)
    rv_cagr  = cagr(revenue[0], revenue[-1], years_span) if len(revenue) >= 2 else 0.0
    ni_cagr  = cagr(net_inc[0], net_inc[-1], years_span) if len(net_inc) >= 2 else 0.0
    cfo_cagr = cagr(cfo[0], cfo[-1], years_span)       if len(cfo) >= 2 else 0.0

    vec += [rv_cagr, ni_cagr, cfo_cagr]
    names += ["revenue_cagr", "netincome_cagr", "cfo_cagr"]

    # 매출 의존 지표 결측 플래그(있으면 0, 없으면 1)
    need_rev_metrics = [net_margin, op_margin, cfo_margin]
    missing_flags = [1.0 if all(v is None for v in seq_) else 0.0 for seq_ in need_rev_metrics]
    vec += missing_flags
    names += ["_miss_net_margin", "_miss_op_margin", "_miss_cfo_margin"]

    # 디버그용 요약
    debug = {
        "years_used": years_5,
        "has_revenue": 0 if all(rv is None for rv in revenue) else 1,
    }
    return vec, names, debug

# ---------- 파일 단위 처리 ----------
def load_fs_yearly(path: Path) -> Tuple[str, Dict[int, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ticker는 json에 있으면 그거, 없으면 파일명 사용
    ticker = data.get("ticker") or path.stem

    # 구조: "years": [2020, 2021, ...]
    years_list = data.get("years", [])
    years_all = []
    for y in years_list:
        # 2020 / "2020" 둘 다 방어
        ys = str(y)
        if ys.isdigit() and len(ys) == 4:
            years_all.append(int(ys))

    # 구조: "yearly_fs": { "2020": {...}, "2021": {...}, ... }
    fs_root = data.get("yearly_fs", {})
    fs_by_year: Dict[int, Dict[str, float]] = {
        y: fs_root.get(str(y), {}) for y in years_all
    }

    return ticker, fs_by_year

def main():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    coll = client.get_or_create_collection("financial_features")

    files = sorted(p for p in NUM_DIR.glob("*.json") if not p.name.startswith("_"))
    print(f"[INFO] Found {len(files)} numerical json files in {NUM_DIR}")

    # feature 이름은 첫 회사 기준으로 저장(동일 스키마)
    feature_names: Optional[List[str]] = None

    for path in files:
        ticker, fs_by_year = load_fs_yearly(path)
        years_all = sorted(fs_by_year.keys())
        if not years_all:
            print(f"[WARN] No yearly data: {path.name}")
            continue
        years_5 = take_last_n_years(years_all, 5)

        vec, names, dbg = build_features_5y(fs_by_year, years_5)

        if feature_names is None:
            feature_names = names
            print(f"[INFO] feature dim = {len(feature_names)}")
        else:
            # 보장: 모든 회사 동일 순서/길이
            assert feature_names == names, "Feature schema mismatch!"

        coll.upsert(
            ids=[ticker],
            embeddings=[vec],
            metadatas=[{
                "ticker": ticker,
                "years_used": ",".join(map(str, dbg["years_used"])),
                "has_revenue": dbg["has_revenue"],
                "feature_schema": ",".join(feature_names),
            }],
        )
        print(f"[OK] {ticker} -> dim={len(vec)} years={years_5}")

    print("[DONE] Stored financial feature vectors in 'financial_features'")

if __name__ == "__main__":
    main()
