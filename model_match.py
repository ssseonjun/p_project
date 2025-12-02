# company_matcher.py
import argparse
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb


# -----------------------------
# Chroma에서 기업 벡터 로딩
# -----------------------------
def load_chroma_collection(
    persist_dir: str,
    collection_name: str,
) -> Dict[str, np.ndarray]:
    """
    ChromaDB에서 지정 컬렉션의 (id, embedding)을 dict[ticker] = vector 로 로드.
    전제:
      - ids는 ticker 또는 유니크 기업 식별자
      - embeddings는 list[list[float]] 형태
    """
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection(name=collection_name, embedding_function=None)

    data = col.get(include=["embeddings"], limit=10000)
    ids = data["ids"]              # list[str]
    embs = np.array(data["embeddings"], dtype=np.float32)  # (N, D)

    return {tid: emb for tid, emb in zip(ids, embs)}


# -----------------------------
# fs(재무벡터) projection 로드/생성
# -----------------------------
def load_or_init_fs_projection(
    path: str,
    in_dim: int = 45,
    out_dim: int = 768,
    seed: int = 42,
) -> np.ndarray:
    """
    financial_features(45차원)을 768차원으로 올리는 고정 projection matrix.

    - path에 .npy 파일이 있으면 로드
    - 없으면 랜덤으로 생성 후 저장

    반환: W_fs (in_dim, out_dim) = (45, 768)
    """

    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    if os.path.exists(path):
        W = np.load(path).astype(np.float32)
        if W.shape != (in_dim, out_dim):
            raise ValueError(
                f"기존 projection {path} shape {W.shape} != ({in_dim}, {out_dim})"
            )
        print(f"[INFO] 기존 fs projection 로드: {path}, shape={W.shape}")
        return W

    # 새로 생성
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=1.0 / np.sqrt(in_dim), size=(in_dim, out_dim)).astype(
        np.float32
    )
    np.save(path, W)
    print(f"[INFO] 새 fs projection 생성 및 저장: {path}, shape={W.shape}")
    return W


# -----------------------------
# ticker 리스트 cell 파싱
# -----------------------------
def parse_ticker_list_cell(cell: Any) -> List[str]:
    """
    'AAPL;MSFT;NVDA' -> ['AAPL','MSFT','NVDA']
    """
    if cell is None:
        return []
    if isinstance(cell, float) and np.isnan(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


# -----------------------------
# 서로 다른 소스(텍스트+재무)를 합쳐 기업 임베딩 생성
# -----------------------------
def merge_sources_for_companies(
    biz_sbert: Dict[str, np.ndarray],
    biz_finbert: Dict[str, np.ndarray],
    wiki_sbert: Dict[str, np.ndarray],
    wiki_finbert: Dict[str, np.ndarray],
    fs_vecs: Dict[str, np.ndarray],
    W_fs: np.ndarray,               # (45, 768) 고정 projection
    w_business: float = 0.4,
    w_wiki: float = 0.4,
    w_fs: float = 0.2,
    comp_w_sbert: float = 0.5,
    comp_w_finbert: float = 0.5,
) -> Tuple[List[str], np.ndarray]:
    """
    기업별 최종 텍스트+재무 임베딩 구성.

    v_business = comp_w_sbert * business_sbert + comp_w_finbert * business_finbert  (768)
    v_wiki     = comp_w_sbert * wiki_sbert     + comp_w_finbert * wiki_finbert      (768)
    v_fs_proj  = fs (45) @ W_fs (45x768) = 768
    v_final    = w_business * v_business + w_wiki * v_wiki + w_fs * v_fs_proj
    """
    # SBERT/FinBERT weight 정규화
    if abs(comp_w_sbert + comp_w_finbert - 1.0) > 1e-6:
        total = comp_w_sbert + comp_w_finbert
        comp_w_sbert /= total
        comp_w_finbert /= total

    # 공통 ticker (재무벡터까지 포함)
    common = set(biz_sbert.keys()) & set(biz_finbert.keys()) \
        & set(wiki_sbert.keys()) & set(wiki_finbert.keys()) \
        & set(fs_vecs.keys())

    tickers = sorted(common)
    vectors = []

    in_dim, out_dim = W_fs.shape
    if out_dim != 768:
        print(f"[WARN] W_fs out_dim={out_dim}, 텍스트 임베딩 dim과 불일치할 수 있음. "
              f"SBERT/FinBERT dim이 {out_dim}인지 확인 필요.")

    for t in tickers:
        vb = comp_w_sbert * biz_sbert[t] + comp_w_finbert * biz_finbert[t]   # (768,)
        vw = comp_w_sbert * wiki_sbert[t] + comp_w_finbert * wiki_finbert[t] # (768,)
        vf = fs_vecs[t]                                                       # (45,)

        if vf.shape[0] != in_dim:
            raise ValueError(
                f"financial_features dim {vf.shape} != projection in_dim {in_dim} for ticker {t}"
            )

        # 45 -> 768 projection
        vf_proj = vf @ W_fs    # (768,)

        # 최종 벡터
        v_final = w_business * vb + w_wiki * vw + w_fs * vf_proj
        vectors.append(v_final)

    mat = np.stack(vectors).astype(np.float32)  # (N, 768)
    # L2 normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms
    return tickers, mat


# -----------------------------
# 매칭 모델 클래스
# -----------------------------
class CompanyMatcher:
    def __init__(
        self,
        # SBERT / FinBERT 모델 이름
        sbert_name: str = "sentence-transformers/all-mpnet-base-v2",
        finbert_name: str = "yiyanghkust/finbert-tone",
        # Chroma 디렉토리 + 컬렉션 이름들
        business_dir: str = "chromaDB/chroma_business",
        wiki_dir: str = "chromaDB/chroma_wiki",
        fs_dir: str = "chromaDB/chroma_fs",
        business_sbert_col: str = "business_sbert",
        business_finbert_col: str = "business_finbert",
        wiki_sbert_col: str = "wiki_sbert",
        wiki_finbert_col: str = "wiki_finbert",
        fs_col: str = "financial_features",
        # 사업/위키/재무 비중 (너가 세팅하는 값, 튜닝 X)
        w_business: float = 0.4,
        w_wiki: float = 0.4,
        w_fs: float = 0.2,
        # fs projection 저장 경로
        fs_proj_path: str = "fs_projection_45to768.npy",
    ):
        # 텍스트 임베딩 모델
        self.sbert = SentenceTransformer(sbert_name)
        self.finbert = SentenceTransformer(finbert_name)

        # Chroma에서 각 소스 로드
        self.biz_sbert = load_chroma_collection(business_dir, business_sbert_col)
        self.biz_finbert = load_chroma_collection(business_dir, business_finbert_col)
        self.wiki_sbert = load_chroma_collection(wiki_dir, wiki_sbert_col)
        self.wiki_finbert = load_chroma_collection(wiki_dir, wiki_finbert_col)
        self.fs_vecs = load_chroma_collection(fs_dir, fs_col)

        self.w_business = w_business
        self.w_wiki = w_wiki
        self.w_fs = w_fs

        # 재무벡터용 고정 projection 로드/생성
        self.W_fs = load_or_init_fs_projection(
            path=fs_proj_path,
            in_dim=next(iter(self.fs_vecs.values())).shape[0],  # 보통 45
            out_dim=768,  # mpnet/finbert hidden dim
        )

    # -------- 텍스트 임베딩 --------
    def embed_texts(
        self,
        texts: List[str],
        text_w_sbert: float = 0.5,
        text_w_finbert: float = 0.5,
        batch_size: int = 32,
    ) -> np.ndarray:
        if abs(text_w_sbert + text_w_finbert - 1.0) > 1e-6:
            total = text_w_sbert + text_w_finbert
            text_w_sbert /= total
            text_w_finbert /= total

        em_sbert = self.sbert.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        em_fin = self.finbert.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        combined = text_w_sbert * em_sbert + text_w_finbert * em_fin
        norms = np.linalg.norm(combined, axis=1, keepdims=True) + 1e-9
        combined = combined / norms
        return combined

    # -------- 기업 임베딩 구성 --------
    def build_company_matrix(
        self,
        comp_w_sbert: float = 0.5,
        comp_w_finbert: float = 0.5,
    ) -> Tuple[List[str], np.ndarray]:
        tickers, mat = merge_sources_for_companies(
            self.biz_sbert,
            self.biz_finbert,
            self.wiki_sbert,
            self.wiki_finbert,
            self.fs_vecs,
            self.W_fs,
            w_business=self.w_business,
            w_wiki=self.w_wiki,
            w_fs=self.w_fs,
            comp_w_sbert=comp_w_sbert,
            comp_w_finbert=comp_w_finbert,
        )
        return tickers, mat

    # -------- 매칭 함수 (forward) --------
    def match(
        self,
        texts: List[str],
        text_w_sbert: float = 0.5,
        text_w_finbert: float = 0.5,
        comp_w_sbert: float = 0.5,
        comp_w_finbert: float = 0.5,
        top_k: int = 10,
        sim_threshold: float = 0.0,
    ) -> List[List[str]]:
        """
        texts: 트윗/기사 리스트
        return: 각 텍스트별 연관기업 ticker 리스트
        """
        # 1) 텍스트 임베딩
        text_embs = self.embed_texts(
            texts,
            text_w_sbert=text_w_sbert,
            text_w_finbert=text_w_finbert,
        )

        # 2) 기업 임베딩
        tickers, comp_mat = self.build_company_matrix(
            comp_w_sbert=comp_w_sbert,
            comp_w_finbert=comp_w_finbert,
        )

        # 3) 코사인 유사도 (이미 normalize 되어 있으므로 dot product)
        sims = text_embs @ comp_mat.T  # (M, N)
        M, N = sims.shape
        k = min(top_k, N)

        # 4) Top-k 추출
        idx_topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        results: List[List[str]] = []

        for i in range(M):
            idxs = idx_topk[i]
            scores = sims[i, idxs]
            order = np.argsort(-scores)
            ordered = idxs[order]

            lst = []
            for j in ordered:
                if sims[i, j] < sim_threshold:
                    continue
                lst.append(tickers[j])
            results.append(lst)

        return results


# -----------------------------
# CLI: forward only
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="트윗/기사에서 연관기업 Top-k 추출 (forward 스크립트, fs 고정 projection 포함)"
    )
    parser.add_argument("--input_csv", required=True, help="id, created_at, text, sentiment 포함 CSV")
    parser.add_argument("--output_csv", required=True, help="predicted_companies 추가해서 저장할 CSV")

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--sim_threshold", type=float, default=0.0)
    parser.add_argument("--text_w_sbert", type=float, default=0.5)
    parser.add_argument("--text_w_finbert", type=float, default=0.5)
    parser.add_argument("--comp_w_sbert", type=float, default=0.5)
    parser.add_argument("--comp_w_finbert", type=float, default=0.5)

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv,sep=";")
    if "text" not in df.columns:
        raise ValueError("입력 CSV에 'text' 컬럼이 필요합니다.")

    texts = df["text"].astype(str).tolist()

    matcher = CompanyMatcher(
        w_business=0.4,
        w_wiki=0.4,
        w_fs=0.2,   # 재무벡터도 반영됨 (projection 통해 768차원에 포함)
        fs_proj_path="out/fs_proj.npy",
    )

    preds = matcher.match(
        texts,
        text_w_sbert=args.text_w_sbert,
        text_w_finbert=args.text_w_finbert,
        comp_w_sbert=args.comp_w_sbert,
        comp_w_finbert=args.comp_w_finbert,
        top_k=args.top_k,
        sim_threshold=args.sim_threshold,
    )

    df["predicted_companies"] = [";".join(p) for p in preds]
    df.to_csv(args.output_csv, index=False)
    print(f"저장 완료: {args.output_csv}")


if __name__ == "__main__":
    main()