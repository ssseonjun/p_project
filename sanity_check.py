import chromadb
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ============================================
# 0. 설정: SBERT / FinBERT 선택
# ============================================
EMB_TYPE = "finbert"   # "sbert" 또는 "finbert"

BIZ_PATH = "chromaDB/chroma_business"
WIKI_PATH     = "chromaDB/chroma_wiki"
if EMB_TYPE == "sbert":
    BIZ_COLL_NAME  = "business_sbert"
    WIKI_COLL_NAME = "wiki_sbert"
elif EMB_TYPE == "finbert":
    BIZ_COLL_NAME  = "business_finbert"
    WIKI_COLL_NAME = "wiki_finbert"
else:
    raise ValueError("EMB_TYPE must be 'sbert' or 'finbert'")

print(f"=== Using embeddings: {EMB_TYPE} ===")
print(f"Business DB path: {BIZ_PATH}")
print(f"Wiki DB path    : {WIKI_PATH}")
print(f"Business collection: {BIZ_COLL_NAME}")
print(f"Wiki collection    : {WIKI_COLL_NAME}")

# ============================================
# 1. Chroma 에서 사업보고서 / 위키 임베딩 불러오기
# ============================================
client_biz = chromadb.PersistentClient(path=BIZ_PATH)
client_wiki = chromadb.PersistentClient(path=WIKI_PATH)

coll_biz  = client_biz.get_collection(BIZ_COLL_NAME)
coll_wiki = client_wiki.get_collection(WIKI_COLL_NAME)

# ids도 함께 가져오기 (doc_id = ticker 역할)
res_biz  = coll_biz.get(include=["embeddings", "metadatas"])
res_wiki = coll_wiki.get(include=["embeddings", "metadatas"])

embs_biz  = np.array(res_biz["embeddings"])
embs_wiki = np.array(res_wiki["embeddings"])

metas_biz  = res_biz["metadatas"]
metas_wiki = res_wiki["metadatas"]

ids_biz    = res_biz["ids"]     # ← ids는 이렇게 그냥 키로 꺼내면 됨
ids_wiki   = res_wiki["ids"]


print(f"#business docs: {len(embs_biz)}, dim = {embs_biz.shape[1]}")
print(f"#wiki docs    : {len(embs_wiki)}, dim = {embs_wiki.shape[1]}")

# ============================================
# 메타데이터 유틸
# ============================================
def extract_first(meta, candidates):
    """여러 후보 키 중에서 처음 존재하는 값을 반환."""
    for k in candidates:
        if k in meta and meta[k] is not None:
            return meta[k]
    return None

def ticker_from_meta_or_id(meta, doc_id):
    """
    가능한 정보에서 ticker를 유연하게 복구:
    1) meta["ticker"] / meta["symbol"]
    2) meta["file_name"] (예: AAPL.json → AAPL)
    3) doc_id (embed 시점에 ticker를 id로 썼으므로)
    """
    t = extract_first(meta, ["ticker", "symbol"])
    if t:
        return t

    fname = meta.get("file_name")
    if isinstance(fname, str) and fname.endswith(".json"):
        return Path(fname).stem

    # 최후의 fallback
    return doc_id

def company_from_meta(meta):
    return extract_first(meta, ["company_name", "corrected_name"])

def sector_from_meta(meta):
    return extract_first(meta, ["sector", "gics_sector"]) or "unknown"

def metas_to_df(ids, metas, source_name):
    rows = []
    for doc_id, m in zip(ids, metas):
        ticker = ticker_from_meta_or_id(m, doc_id)
        company = company_from_meta(m)
        sector  = sector_from_meta(m)
        rows.append({
            "id": doc_id,
            "ticker": ticker,
            "company": company,
            "sector": sector,
            "source": source_name,
        })
    df = pd.DataFrame(rows)
    df["idx"] = np.arange(len(df))
    return df

df_biz  = metas_to_df(ids_biz,  metas_biz,  source_name="business")
df_wiki = metas_to_df(ids_wiki, metas_wiki, source_name="wiki")

# ============================================
# 2. 임베딩 collapse 여부 체크 (차원별 분산)
# ============================================
def print_dim_stats(embs, name):
    dim_std = embs.std(axis=0)
    print(f"\n=== Per-dimension std of {name} embeddings ===")
    print(f"mean std : {dim_std.mean():.6f}")
    print(f"min std  : {dim_std.min():.6f}")
    print(f"max std  : {dim_std.max():.6f}")
    return dim_std

std_biz  = print_dim_stats(embs_biz,  "BUSINESS")
std_wiki = print_dim_stats(embs_wiki, "WIKI")

zero_dims_biz  = (std_biz == 0).sum()
zero_dims_wiki = (std_wiki == 0).sum()
print("zero-variance dims (biz):", zero_dims_biz)
print("zero-variance dims (wiki):", zero_dims_wiki)

# ============================================
# 3. t-SNE 시각화 (business + wiki 함께)
# ============================================
max_points = 1000
N_biz  = len(embs_biz)
N_wiki = len(embs_wiki)

def sample_indices(N, max_points_half, seed):
    if N <= max_points_half:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=max_points_half, replace=False)

max_each = max_points // 2
idx_biz_sample  = sample_indices(N_biz,  max_each, seed=42)
idx_wiki_sample = sample_indices(N_wiki, max_each, seed=43)

embs_all = np.vstack([
    embs_biz[idx_biz_sample],
    embs_wiki[idx_wiki_sample],
])

df_all = pd.concat([
    df_biz.iloc[idx_biz_sample].reset_index(drop=True),
    df_wiki.iloc[idx_wiki_sample].reset_index(drop=True),
], ignore_index=True)

print("\nRunning t-SNE on combined (business + wiki)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Z = tsne.fit_transform(embs_all)
df_all["x"] = Z[:, 0]
df_all["y"] = Z[:, 1]

# (1) 출처별 색 (business vs wiki)
plt.figure(figsize=(8, 6))
for src in df_all["source"].unique():
    mask = df_all["source"] == src
    plt.scatter(
        df_all.loc[mask, "x"],
        df_all.loc[mask, "y"],
        label=src,
        alpha=0.6,
        s=10,
    )
plt.legend()
plt.title(f"t-SNE of {EMB_TYPE} (color = source)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.show()

# (2) 섹터별 색 (있으면)
unique_sectors = [s for s in df_all["sector"].unique() if s != "unknown"]
if len(unique_sectors) > 0:
    plt.figure(figsize=(8, 6))
    for sec in unique_sectors:
        mask = df_all["sector"] == sec
        plt.scatter(
            df_all.loc[mask, "x"],
            df_all.loc[mask, "y"],
            label=sec,
            alpha=0.6,
            s=10,
        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"t-SNE of {EMB_TYPE} (color = sector)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.show()

# ============================================
# 4. 기업별로 사업보고서 / 위키 임베딩 평균 후
#    같은 기업끼리 cosine similarity 계산
# ============================================
def aggregate_by_ticker(df, embs):
    """
    (ticker, source) 기준으로 임베딩 평균
    """
    group_dict = defaultdict(list)
    for i, row in df.iterrows():
        tic = row["ticker"]
        if tic is None:
            continue
        group_dict[(tic, row["source"])] .append(row["idx"])

    rows = []
    agg_embs = []
    for (tic, src), idxs in group_dict.items():
        rows.append({
            "ticker": tic,
            "source": src,
            "company": df.loc[df["idx"] == idxs[0], "company"].values[0],
            "sector": df.loc[df["idx"] == idxs[0], "sector"].values[0],
            "n_docs": len(idxs),
        })
        agg_embs.append(embs[idxs].mean(axis=0))

    if not agg_embs:
        return pd.DataFrame(columns=["ticker", "source", "company", "sector", "n_docs"]), np.empty((0, embs.shape[1]))

    agg_df = pd.DataFrame(rows)
    agg_embs = np.vstack(agg_embs)
    return agg_df, agg_embs

agg_biz_df,  agg_biz_embs  = aggregate_by_ticker(df_biz,  embs_biz)
agg_wiki_df, agg_wiki_embs = aggregate_by_ticker(df_wiki, embs_wiki)

print("\n#(ticker, business) groups:", len(agg_biz_df))
print("#(ticker, wiki) groups    :", len(agg_wiki_df))

biz_dict  = {row["ticker"]: agg_biz_embs[i]  for i, row in agg_biz_df.iterrows()}
wiki_dict = {row["ticker"]: agg_wiki_embs[i] for i, row in agg_wiki_df.iterrows()}
name_dict = {row["ticker"]: row["company"]   for _, row in agg_biz_df.iterrows()}

common_tickers = sorted(set(biz_dict.keys()) & set(wiki_dict.keys()))
print("#companies with both business & wiki:", len(common_tickers))

if len(common_tickers) > 0:
    biz_mat  = np.vstack([biz_dict[t]  for t in common_tickers])
    wiki_mat = np.vstack([wiki_dict[t] for t in common_tickers])

    num = np.sum(biz_mat * wiki_mat, axis=1)
    denom = np.linalg.norm(biz_mat, axis=1) * np.linalg.norm(wiki_mat, axis=1)
    cos_sim = num / denom

    sim_df = pd.DataFrame({
        "ticker": common_tickers,
        "company": [name_dict.get(t, "") for t in common_tickers],
        "cos_sim_biz_wiki": cos_sim,
    }).sort_values("cos_sim_biz_wiki")

    print("\n=== Lowest 10 business vs wiki similarities ===")
    print(sim_df.head(10))

    print("\n=== Highest 10 business vs wiki similarities ===")
    print(sim_df.tail(10))

    plt.figure(figsize=(6, 4))
    plt.hist(sim_df["cos_sim_biz_wiki"], bins=30)
    plt.title(f"Cosine similarity (business vs wiki, same ticker) - {EMB_TYPE}")
    plt.xlabel("cosine similarity")
    plt.ylabel("#companies")
    plt.tight_layout()
    plt.show()
else:
    print("No companies have both business and wiki embeddings (ticker 기준). 메타데이터 / ids 확인 필요.")

# ============================================
# 5. 전체 기업벡터 pairwise similarity 분포
# ============================================
def random_pairwise_stats(agg_embs, name, max_sample=400):
    N = len(agg_embs)
    if N == 0:
        print(f"[{name}] No embeddings.")
        return

    if N > max_sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=max_sample, replace=False)
        sub_embs = agg_embs[idx]
    else:
        sub_embs = agg_embs

    sim_mat = cosine_similarity(sub_embs)
    upper_tri = sim_mat[np.triu_indices_from(sim_mat, k=1)]

    print(f"\n=== Pairwise cosine similarities ({name}) ===")
    print(f"mean  : {upper_tri.mean():.4f}")
    print(f"std   : {upper_tri.std():.4f}")
    print(f"min   : {upper_tri.min():.4f}")
    print(f"max   : {upper_tri.max():.4f}")

    plt.figure(figsize=(6, 4))
    plt.hist(upper_tri, bins=40)
    plt.title(f"Pairwise cosine similarity ({name}) - {EMB_TYPE}")
    plt.xlabel("cosine similarity")
    plt.ylabel("#pairs")
    plt.tight_layout()
    plt.show()

random_pairwise_stats(agg_biz_embs,  name="BUSINESS (per-ticker)")
random_pairwise_stats(agg_wiki_embs, name="WIKI (per-ticker)")