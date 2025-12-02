#embed_business.py
# 생성되는 벡터: business_sbert, business_finbert
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# --------------------------
# 디렉토리 설정
# --------------------------
BASE_DIR = Path(__file__).resolve().parent

JSON_DIR = BASE_DIR / "json"
BUSINESS_DIR = JSON_DIR / "business_json"     # 사업보고서 json 위치
BUSINESS_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_PATH = BASE_DIR / "chromaDB" / "chroma_business"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)  # Chroma persistent path

# --------------------------
# 텍스트 전처리 / 임베딩 유틸
# --------------------------
def chunk_text(text: str, max_chars: int = 1800) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _encode_chunks_with_pooling(
    chunks: List[str],
    model: SentenceTransformer,
    batch_size: int = 8,
) -> Optional[np.ndarray]:
    if not chunks:
        return None

    emb = model.encode(
        chunks,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.mean(axis=0)


def summarize_item_sections_for_ticker(
    ticker: str,
    sbert_model: SentenceTransformer,
    finbert_model: SentenceTransformer,
    chroma_sbert,
    chroma_finbert,
    max_chars_per_chunk: int = 1800,
    use_items: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    json/business_json/{ticker}.json 파일에서 item_sections를 읽어서
    SBERT / FinBERT 임베딩을 만들고, 바로 ChromaDB에 upsert.
    """

    raw_path = BUSINESS_DIR / f"{ticker}.json"
    if not raw_path.exists():
        print(f"[WARN] Raw JSON not found for ticker={ticker}: {raw_path}")
        return None

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections: Dict[str, str] = data.get("item_sections", {})
    if not sections:
        print(f"[INFO] No item_sections in {raw_path}")
        return None

    all_keys = sorted(sections.keys())
    if use_items is not None:
        keys_used = [k for k in all_keys if k in use_items]
    else:
        keys_used = all_keys

    texts = []
    for k in keys_used:
        txt = sections.get(k, "")
        if txt and txt.strip():
            texts.append(txt.strip())

    if not texts:
        print(f"[INFO] No non-empty items for ticker={ticker}")
        return None

    full_text = "\n\n".join(texts)
    chunks = chunk_text(full_text, max_chars=max_chars_per_chunk)

    sbert_vec = _encode_chunks_with_pooling(chunks, sbert_model)
    finbert_vec = _encode_chunks_with_pooling(chunks, finbert_model)

    if sbert_vec is None or finbert_vec is None:
        print(f"[INFO] Failed to create embeddings for ticker={ticker}")
        return None

    # numpy -> list 변환 (Chroma는 리스트 형태를 기대)
    sbert_list = sbert_vec.tolist()
    finbert_list = finbert_vec.tolist()

    metadata = {
        "ticker": ticker,
        # 리스트 → 문자열로 변환
        "items_used": ",".join(keys_used),
    }

    # --------------------------
    # ChromaDB에 upsert
    # --------------------------
    chroma_sbert.upsert(
        ids=[ticker],
        embeddings=[sbert_list],
        metadatas=[metadata],
    )

    chroma_finbert.upsert(
        ids=[ticker],
        embeddings=[finbert_list],
        metadatas=[metadata],
    )

    # 리턴값은 필요하면 나중에 활용
    result = {
        "ticker": ticker,
        "items_used": keys_used,
        "sbert_vec": sbert_vec,
        "finbert_vec": finbert_vec,
    }
    return result


def main():
    # --------------------------
    # 1) SBERT / FinBERT 로드
    # --------------------------
    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    finbert = SentenceTransformer("yiyanghkust/finbert-tone")

    # --------------------------
    # 2) ChromaDB 클라이언트 / 컬렉션 준비
    # --------------------------
    client_finbert = chromadb.PersistentClient(path=str(CHROMA_PATH))
    client_sbert = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # 컬렉션 이름은 원하는 대로 바꿔도 됨
    chroma_sbert = client_sbert.get_or_create_collection("business_sbert")
    chroma_finbert = client_finbert.get_or_create_collection("business_finbert")

    # --------------------------
    # 3) 각 ticker에 대해 임베딩 → Chroma upsert
    # --------------------------
    raw_files = sorted(BUSINESS_DIR.glob("*.json"))
    print(f"[INFO] Found {len(raw_files)} business_json files")

    for raw_file in raw_files:
        ticker = raw_file.stem  # "AAPL.json" -> "AAPL"
        res = summarize_item_sections_for_ticker(
            ticker=ticker,
            sbert_model=sbert,
            finbert_model=finbert,
            chroma_sbert=chroma_sbert,
            chroma_finbert=chroma_finbert,
            max_chars_per_chunk=1800,
            use_items=None,  # 나중에 ["Item 1", ...]로 좁혀도 됨
        )
        if res is None:
            continue

        print(
            f"[OK] {ticker}: SBERT dim={res['sbert_vec'].shape}, "
            f"FinBERT dim={res['finbert_vec'].shape}"
        )


if __name__ == "__main__":
    main()
