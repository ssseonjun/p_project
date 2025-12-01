"""
sp500_fs_2019_2023.json (SEC_parsing.py로 만든 전체 S&P500 재무 데이터)을 읽어서
모든 기업에 대해:

  yearly_fs(2019~2023) -> feature matrix -> MLP -> attention

으로 최종 재무 구조 벡터 h_fs를 계산하고,

1) sp500_h_fs.npy (N x 64)
2) sp500_h_fs_mapping.csv (index, symbol, name, cik)
3) ChromaDB 컬렉션에 embeddings + metadata upsert

까지 수행.

pip install torch numpy pandas chromadb
"""

import json
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import chromadb

YEARS = [2019, 2020, 2021, 2022, 2023]


# -----------------------------
# 1. feature engineering 유틸
# -----------------------------

def safe_div(x: float, y: float, eps: float = 1e-9) -> float:
    if y is None or abs(y) < eps:
        return 0.0
    return float(x) / float(y)


def log1p_safe(x: float) -> float:
    if x is None:
        x = 0.0
    return float(np.log1p(max(x, 0.0)))


def compute_year_features(fs: Dict[str, float]) -> np.ndarray:
    """
    한 연도 재무제표 dict -> feature vector (약 30개)
    SEC_parsing.py에서 채운 키들 사용.
    """
    # ----- 원시 계정 -----
    rev = fs.get("revenue", 0.0)
    cogs = fs.get("cogs", 0.0)
    gross_profit = fs.get("gross_profit", rev - cogs)
    op_inc = fs.get("operating_income", 0.0)
    net_inc = fs.get("net_income", 0.0)
    rnd = fs.get("rnd_expense", 0.0)
    sga = fs.get("sga_expense", 0.0)
    interest = fs.get("interest_expense", 0.0)
    tax = fs.get("income_tax_expense", 0.0)

    total_assets = fs.get("total_assets", 0.0)
    total_liab = fs.get("total_liabilities", 0.0)
    total_equity = fs.get("total_equity", total_assets - total_liab)
    curr_assets = fs.get("current_assets", 0.0)
    curr_liab = fs.get("current_liabilities", 0.0)
    cash = fs.get("cash_and_equiv", 0.0)
    st_debt = fs.get("short_term_debt", 0.0)
    lt_debt = fs.get("long_term_debt", 0.0)
    total_debt = st_debt + lt_debt
    ar = fs.get("accounts_receivable", 0.0)
    inventory = fs.get("inventory", 0.0)
    ppe = fs.get("ppe", 0.0)
    goodwill = fs.get("goodwill", 0.0)

    cfo = fs.get("cfo", 0.0)
    cfi = fs.get("cfi", 0.0)
    cff = fs.get("cff", 0.0)
    d_and_a = fs.get("depr_amort", 0.0)
    capex = fs.get("capex", 0.0)
    dividends = fs.get("dividends", 0.0)
    buybacks = fs.get("share_repurchases", 0.0)

    fcf = cfo - capex

    feats: List[float] = []

    # 1) 규모 (log 스케일)
    feats.extend([
        log1p_safe(rev),
        log1p_safe(total_assets),
        log1p_safe(total_equity),
        log1p_safe(total_debt),
        log1p_safe(cfo),
    ])

    # 2) 마진/수익성
    feats.extend([
        safe_div(gross_profit, rev),   # gross margin
        safe_div(op_inc, rev),         # op margin
        safe_div(net_inc, rev),        # net margin
        safe_div(rnd, rev),            # R&D / Sales
        safe_div(sga, rev),            # SG&A / Sales
        safe_div(interest, rev),       # Interest / Sales
        safe_div(tax, rev),            # Tax / Sales
    ])

    # 3) 레버리지/유동성
    feats.extend([
        safe_div(total_debt, total_equity),   # Debt / Equity
        safe_div(total_debt, total_assets),   # Debt / Assets
        safe_div(curr_assets, curr_liab),     # Current ratio
        safe_div(cash, total_assets),         # Cash / Assets
        safe_div(cash, rev),                  # Cash / Sales
    ])

    # 4) 효율/구조
    feats.extend([
        safe_div(rev, total_assets),          # Asset turnover
        safe_div(ar, rev),                    # AR / Sales
        safe_div(inventory, rev),             # Inventory / Sales
        safe_div(ppe, total_assets),          # PPE / Assets
        safe_div(goodwill, total_assets),     # Goodwill / Assets
    ])

    # 5) 현금흐름 구조
    feats.extend([
        safe_div(cfo, rev),                   # OCF margin
        safe_div(fcf, rev),                   # FCF / Sales
        safe_div(fcf, net_inc),               # FCF / Net income
        safe_div(capex, rev),                 # CapEx / Sales
        safe_div(dividends, net_inc),         # Dividend payout
        safe_div(buybacks, net_inc),          # Buyback / Net income
        safe_div(cfi, rev),                   # CFI / Sales
        safe_div(cff, rev),                   # CFF / Sales
    ])

    return np.array(feats, dtype=np.float32)


def build_company_year_feature_matrix(
    yearly_fs: Dict[str, Dict[str, float]],
    years: List[int] = YEARS,
) -> np.ndarray:
    """
    yearly_fs 구조:
      {
        "2019": { ... },
        "2020": { ... },
        ...
      }
    -> (num_years, feature_dim)
    """
    year_features = []
    for y in years:
        fs_y = yearly_fs.get(str(y), {}) or {}
        feats_y = compute_year_features(fs_y)
        year_features.append(feats_y)

    return np.stack(year_features, axis=0)


# -----------------------------
# 2. PyTorch 모듈: 연도별 MLP + attention
# -----------------------------

class FinancialYearEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        h_flat = self.net(x_flat)
        h = h_flat.view(B, T, -1)
        return h


class YearAttentionAggregator(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.attn = nn.Linear(emb_dim, 1)

    def forward(self, year_embs: torch.Tensor, mask: torch.Tensor = None):
        B, T, D = year_embs.shape
        scores = self.attn(year_embs).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)   # (B, T)
        h_fs = torch.sum(year_embs * attn_weights.unsqueeze(-1), dim=1)  # (B, D)

        return h_fs, attn_weights


class FinancialStructureEncoder(nn.Module):
    def __init__(self, feature_dim: int, year_emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.year_encoder = FinancialYearEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            emb_dim=year_emb_dim,
        )
        self.aggregator = YearAttentionAggregator(emb_dim=year_emb_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor = None):
        year_embs = self.year_encoder(features)
        h_fs, attn_weights = self.aggregator(year_embs, mask)
        return h_fs, attn_weights


# -----------------------------
# 3. main: 전체 기업 h_fs 계산 + NPY + CSV + ChromaDB 저장
# -----------------------------

def main():
    # 재현성
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- 0) 경로 세팅 ----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "sp500_fs_2019_2023.json")
    out_npy = os.path.join(base_dir, "sp500_h_fs.npy")
    out_csv = os.path.join(base_dir, "sp500_h_fs_mapping.csv")
    chroma_path = os.path.join(base_dir, "chromaDB/chroma_fs")  # 여기 폴더에 ChromaDB 파일들 저장

    print("[INFO] base_dir:", base_dir)
    print("[INFO] JSON path:", json_path)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} 파일이 없습니다. "
                                f"SEC_parsing.py로 JSON 먼저 생성됐는지 확인하세요.")

    # 1) JSON 로드
    with open(json_path, "r") as f:
        dataset = json.load(f)

    symbols = list(dataset.keys())
    print(f"[INFO] Loaded {len(symbols)} companies from {json_path}")

    feature_mats = []
    mapping_rows = []

    # 2) 회사별 feature matrix 만들기
    for idx, sym in enumerate(symbols):
        entry = dataset[sym]
        yearly_fs = entry["yearly_fs"]
        mat = build_company_year_feature_matrix(yearly_fs, YEARS)  # (5, feature_dim)
        feature_mats.append(mat)

        mapping_rows.append({
            "index": idx,
            "symbol": sym,
            "name": entry.get("name", ""),
            "cik": entry.get("cik", ""),
        })

    feature_mats = np.stack(feature_mats, axis=0)  # (N, 5, feature_dim)
    feature_tensor = torch.from_numpy(feature_mats)

    B, T, D = feature_tensor.shape
    print(f"[INFO] feature_tensor shape: {feature_tensor.shape}  (B={B}, T={T}, D={D})")

    # 3) 모델 초기화 & forward
    model = FinancialStructureEncoder(feature_dim=D, year_emb_dim=64, hidden_dim=128)
    model.eval()

    with torch.no_grad():
        h_fs, attn = model(feature_tensor)  # h_fs: (N, 64)

    print("[INFO] h_fs shape:", h_fs.shape)

    # 4) NPY / CSV 저장 (원하면 유지, 아니면 주석 처리해도 됨)
    h_fs_np = h_fs.detach().numpy()
    print("[INFO] Saving NPY to:", out_npy)
    np.save(out_npy, h_fs_np)

    mapping_df = pd.DataFrame(mapping_rows)
    print("[INFO] Saving CSV to:", out_csv)
    mapping_df.to_csv(out_csv, index=False)

    # 5) ChromaDB에 저장
    print("[INFO] Initializing ChromaDB at:", chroma_path)
    client = chromadb.PersistentClient(path=chroma_path)

    # embedding_function=None: 이미 직접 만든 임베딩을 넣을 것이기 때문에
    collection = client.get_or_create_collection(
        name="sp500_financial_structure",
        embedding_function=None,
    )

    ids = [row["symbol"] for row in mapping_rows]  # 심볼을 ID로 사용 (각 기업 유니크)
    metadatas = mapping_rows                      # index, symbol, name, cik 전부 metadata에 넣기
    documents = [row["name"] for row in mapping_rows]  # 문서 텍스트는 일단 회사 이름 정도

    embeddings_list = h_fs_np.tolist()  # (N, 64) -> List[List[float]]

    print("[INFO] Upserting embeddings into Chroma collection...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents,
    )

    print("[INFO] Upsert completed into collection 'sp500_financial_structure'.")
    print("[INFO] Saved financial embeddings (NPY/CSV) and ChromaDB index.")

if __name__ == "__main__":
    main()