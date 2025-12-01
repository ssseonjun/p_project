# classification_demo_template.py
import argparse
import json
import numpy as np
import pandas as pd

# ============= 공통 유틸 =============

def json_to_vec(json_str: str) -> np.ndarray | None:
    """JSON 문자열 -> np.ndarray (float32)"""
    if not isinstance(json_str, str) or not json_str.strip():
        return None
    arr = np.array(json.loads(json_str), dtype=np.float32)
    return arr


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 정규화"""
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def compute_cosine_similarity(doc_vec: np.ndarray, comp_vec: np.ndarray) -> float:
    """이미 L2 정규화된 벡터들 간 dot = cosine"""
    
    return float(np.dot(doc_vec, comp_vec))

# ============= 모드별 설정 =============

def build_config(mode: str):
    """
    mode에 따라 어떤 파일을 쓸지, 어떤 컬럼을 쓸지 정의
    실제 경로/파일명은 선준이가 쓰는 구조에 맞게 수정해도 됨
    """
    if mode == "finbert":
        return {
            "company_csvs": [
                "../file/sp500_wiki_finbert.csv",   # FinBERT 임베딩
            ],
            "company_emb_cols": ["Embedding"],      # 임베딩 컬럼 이름
            "doc_csv": "../file/testDoc_finbert.csv",
            "doc_emb_col": "Embedding",
            "combine": False,   # 한 종류의 임베딩만 사용
        }

    if mode == "bert":
        return {
            "company_csvs": [
                "../file/sp500_wiki_bert.csv",      # BERT 임베딩
            ],
            "company_emb_cols": ["Embedding"],
            "doc_csv": "../file/testDoc_bert.csv",
            "doc_emb_col": "Embedding",
            "combine": False,
        }

    if mode == "finbert_sbert":
        # FinBERT + SBERT 조합 예시
        return {
            "company_csvs": [
                "../file/sp500_wiki_finbert.csv",
                "../file/sp500_wiki_sbert.csv",
            ],
            "company_emb_cols": ["Embedding", "Embedding"],  # 각 파일에서 임베딩 컬럼 이름
            "doc_csv": "../file/testDoc_combo.csv",  # 여기엔 finbert/sbert 둘 다 있다고 가정
            "doc_emb_col": None,   # 아래에서 따로 로딩
            "combine": True,
            "alpha": 0.7,  # semantic(SBERT) 비중
            "beta":  0.3,  # finance(FinBERT) 비중
        }

    raise ValueError(f"Unknown mode: {mode}")

# ============= 메인 로직 =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["finbert", "bert", "finbert_sbert"],
        help="실험 모드 선택",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="상위 몇 개 기업을 볼지",
    )
    args = parser.parse_args()

    cfg = build_config(args.mode)
    topk = args.topk

    # ----- 문서 임베딩 로드 -----
    if args.mode in ("finbert", "bert"):
        doc_df = pd.read_csv(cfg["doc_csv"])
        doc_vec = json_to_vec(doc_df.loc[0, cfg["doc_emb_col"]])
        doc_vec = l2_normalize(doc_vec)

        if doc_vec is None:
            raise RuntimeError("문서 임베딩이 비어있습니다.")

    elif args.mode == "finbert_sbert":
        # 예: test_vec_combo.csv 안에 FinBERT_Embedding, SBERT_Embedding 두 컬럼이 있다고 가정
        doc_df = pd.read_csv(cfg["doc_csv"])
        doc_fin = l2_normalize(json_to_vec(doc_df.loc[0, "FinBERT_Embedding"]))
        doc_sem = l2_normalize(json_to_vec(doc_df.loc[0, "SBERT_Embedding"]))
        if doc_fin is None or doc_sem is None:
            raise RuntimeError("문서 FinBERT/SBERT 임베딩이 비어있습니다.")

    # ----- 회사 임베딩 로드 및 유사도 계산 -----
    results = []

    if not cfg["combine"]:
        # BERT 단독 / FinBERT 단독
        comp_df = pd.read_csv(cfg["company_csvs"][0])
        emb_col = cfg["company_emb_cols"][0]

        for i in range(comp_df.shape[0]):
            # Exists 컬럼이 있을 경우만 필터링 (없으면 그냥 패스)
            if "Exists" in comp_df.columns and comp_df.loc[i, "Exists"] is False:
                continue

            emb_str = comp_df.loc[i, emb_col]
            vec = json_to_vec(emb_str)
            if vec is None:
                continue
            vec = l2_normalize(vec)

            sim = compute_cosine_similarity(doc_vec, vec)
            company_name = comp_df.loc[i, "CorrectedCompany"] if "CorrectedCompany" in comp_df.columns else comp_df.loc[i, "Symbol"]
            results.append([company_name, sim])

    else:
        # FinBERT + SBERT 조합
        fin_df = pd.read_csv(cfg["company_csvs"][0])
        sem_df = pd.read_csv(cfg["company_csvs"][1])

        alpha = cfg["alpha"]  # SBERT 비중
        beta  = cfg["beta"]   # FinBERT 비중

        # 두 데이터프레임이 같은 순서/기업이라고 가정 (동일 CSV 기반에서 모델만 바꿔 임베딩한 경우)
        for i in range(fin_df.shape[0]):
            if "Exists" in fin_df.columns and fin_df.loc[i, "Exists"] is False:
                continue

            fin_str = fin_df.loc[i, cfg["company_emb_cols"][0]]
            sem_str = sem_df.loc[i, cfg["company_emb_cols"][1]]

            v_fin = l2_normalize(json_to_vec(fin_str))
            v_sem = l2_normalize(json_to_vec(sem_str))

            if v_fin is None or v_sem is None:
                continue

            sim_fin = compute_cosine_similarity(doc_fin, v_fin)
            sim_sem = compute_cosine_similarity(doc_sem, v_sem)

            sim_total = alpha * sim_sem + beta * sim_fin

            company_name = fin_df.loc[i, "CorrectedCompany"] if "CorrectedCompany" in fin_df.columns else fin_df.loc[i, "Symbol"]
            results.append([company_name, sim_total])

    # ----- 결과 정렬 및 출력 -----
    res_df = pd.DataFrame(results, columns=["company", "similarity"])
    res_df = res_df.sort_values(by="similarity", ascending=False)

    print("===========================")
    print(f"Mode = {args.mode}, Top-{topk}")
    print(res_df.head(topk))
    print("===========================")


if __name__ == "__main__":
    main()