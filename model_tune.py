# tune_params.py
import argparse
import itertools
from typing import List, Set, Dict, Any, Tuple

import numpy as np
import pandas as pd

from model_match import CompanyMatcher, parse_ticker_list_cell


def evaluate_config(
    matcher: CompanyMatcher,
    df: pd.DataFrame,
    text_w_sbert: float,
    comp_w_sbert: float,
    top_k: int,
    sim_threshold: float,
) -> Tuple[float, float]:
    """
    한 파라미터 조합에 대해:
      - avg_appropriate: 평균 적절한 기업 수 (pred ∩ valid_companies)
      - avg_inappropriate: 평균 부적절한 기업 수 (pred ∩ invalid_companies)
    를 계산해서 반환.
    """
    texts = df["text"].astype(str).tolist()

    preds = matcher.match(
        texts,
        text_w_sbert=text_w_sbert,
        text_w_finbert=1.0 - text_w_sbert,
        comp_w_sbert=comp_w_sbert,
        comp_w_finbert=1.0 - comp_w_sbert,
        top_k=top_k,
        sim_threshold=sim_threshold,
    )

    appropriate_counts: List[int] = []
    inappropriate_counts: List[int] = []

    has_valid = "valid_companies" in df.columns
    has_invalid = "invalid_companies" in df.columns

    for i, pred_list in enumerate(preds):
        pred_set: Set[str] = set(pred_list)

        if has_valid:
            valid_set = set(parse_ticker_list_cell(df.iloc[i]["valid_companies"]))
        else:
            valid_set = set()

        if has_invalid:
            invalid_set = set(parse_ticker_list_cell(df.iloc[i]["invalid_companies"]))
        else:
            invalid_set = set()

        appropriate_counts.append(len(pred_set & valid_set))
        inappropriate_counts.append(len(pred_set & invalid_set))

    avg_appropriate = float(np.mean(appropriate_counts)) if appropriate_counts else 0.0
    avg_inappropriate = float(np.mean(inappropriate_counts)) if inappropriate_counts else 0.0
    return avg_appropriate, avg_inappropriate


def main():
    parser = argparse.ArgumentParser(
        description="Grid search로 FinBERT/SBERT 비중, top-k, threshold 튜닝 (backward 역할)"
    )
    parser.add_argument("--input_csv", required=True, help="valid/invalid 컬럼이 포함된 CSV")
    parser.add_argument("--output_report", type=str, default="grid_search_results.csv")

    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)

    # forward에서 썼던 것과 같은 구조로 matcher 생성
    matcher = CompanyMatcher(
        w_business=0.4,
        w_wiki=0.4,
        w_fs=0.2,
    )

    # ----- Grid 정의 (원하는 대로 바꿔도 됨) -----
    text_w_sbert_list = [0.3, 0.5, 0.7]   # 텍스트 SBERT 비중
    comp_w_sbert_list = [0.3, 0.5, 0.7]   # 기업 SBERT 비중
    top_k_list = [5, 10, 15]
    sim_threshold_list = [0.1, 0.2, 0.3]

    rows: List[Dict[str, Any]] = []

    for text_w_sbert, comp_w_sbert, top_k, sim_th in itertools.product(
        text_w_sbert_list,
        comp_w_sbert_list,
        top_k_list,
        sim_threshold_list,
    ):
        avg_app, avg_inapp = evaluate_config(
            matcher,
            df,
            text_w_sbert=text_w_sbert,
            comp_w_sbert=comp_w_sbert,
            top_k=top_k,
            sim_threshold=sim_th,
        )

        print(
            f"[config] text_sbert={text_w_sbert:.2f}, comp_sbert={comp_w_sbert:.2f}, "
            f"top_k={top_k}, sim_th={sim_th:.2f} "
            f"=> avg_app={avg_app:.3f}, avg_inapp={avg_inapp:.3f}"
        )

        rows.append(
            {
                "text_w_sbert": text_w_sbert,
                "text_w_finbert": 1.0 - text_w_sbert,
                "comp_w_sbert": comp_w_sbert,
                "comp_w_finbert": 1.0 - comp_w_sbert,
                "top_k": top_k,
                "sim_threshold": sim_th,
                "avg_appropriate": avg_app,
                "avg_inappropriate": avg_inapp,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_report, index=False)
    print(f"Grid search 결과 저장: {args.output_report}")


if __name__ == "__main__":
    main()