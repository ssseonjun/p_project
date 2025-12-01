"""
file/json 안에 있는 *_business.json 파일을 읽어서
1) business_and_revenue 텍스트 요약
2) 요약문을 SBERT, FinBERT로 임베딩
3) ChromaDB(chromaDB/chroma_business)에 저장

생성되는 Chroma 컬렉션:
- business_sbert   : SBERT 임베딩
- business_finbert : FinBERT 임베딩
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings


# -----------------------------
# 경로 설정
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "file" / "json"
CHROMA_DIR = BASE_DIR / "chromaDB" / "chroma_business"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 0. business.json에서 텍스트 꺼내기 헬퍼
# -----------------------------

def get_business_text_from_data(data: dict) -> str:
    """
    business.json 구조가 여러 가지일 수 있으므로,
    - 루트에 문자열 business_and_revenue가 있으면 그대로 사용
    - 루트에 dict형 business_and_revenue가 있으면 연도별로 합침
    - business_by_year가 있으면 그 안의 business_and_revenue들을 합침
    최종적으로 하나의 긴 문자열을 반환.
    """
    # 1) 루트의 business_and_revenue가 문자열인 경우
    root_text = data.get("business_and_revenue")
    if isinstance(root_text, str) and root_text.strip():
        return root_text.strip()

    # 2) 루트의 business_and_revenue가 dict인 경우 (예: {"2019": "...", "2020": "..."} 형태)
    if isinstance(root_text, dict):
        pieces = []
        for year, txt in sorted(root_text.items(), key=lambda x: x[0]):
            if isinstance(txt, str) and txt.strip():
                pieces.append(f"[{year}]\n{txt.strip()}")
        if pieces:
            return "\n\n".join(pieces)

    # 3) business_by_year 구조가 있는 경우
    by_year = data.get("business_by_year")
    if isinstance(by_year, dict):
        pieces = []
        for year, info in sorted(by_year.items(), key=lambda x: x[0]):
            if not isinstance(info, dict):
                continue
            txt = info.get("business_and_revenue", "")
            if isinstance(txt, str) and txt.strip():
                pieces.append(f"[{year}]\n{txt.strip()}")
        if pieces:
            return "\n\n".join(pieces)

    # 4) 위 케이스 다 안 맞으면 빈 문자열
    return ""


# -----------------------------
# 1. 텍스트 chunking 유틸
# -----------------------------

def chunk_text(text: str, max_chars: int = 3000, overlap: int = 200) -> List[str]:
    """
    긴 텍스트를 summarization 모델용으로 잘라주는 단순 함수.
    - max_chars: 한 chunk 최대 문자 수
    - overlap : 앞 chunk와 겹치는 문자 수 (문맥 끊김 방지용)
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + max_chars
        if end >= n:
            chunks.append(text[start:])
            break

        # 너무 문장 중간에서 끊지 않도록, 뒤쪽에서 가장 가까운 마침표(.) 기준으로 잘라보기
        cut = text.rfind(".", start, end)
        if cut == -1 or cut < start + max_chars * 0.5:
            cut = end

        chunks.append(text[start:cut].strip())
        start = max(cut - overlap, 0)

    return chunks


# -----------------------------
# 2. Summarization 파이프라인
# -----------------------------

def get_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    Hugging Face summarization 파이프라인 준비.
    필요하면 model_name을 바꿔서 다른 요약 모델 사용 가능.
    """
    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
    )
    return summarizer


def summarize_long_text(
    summarizer,
    text: str,
    chunk_max_chars: int = 3000,
    overlap: int = 200,
    max_length: int = 200,
    min_length: int = 60,
) -> str:
    """
    1) 긴 텍스트를 chunk 단위로 나누고
    2) chunk마다 1차 요약
    3) (메모리 이슈 방지를 위해) 1차 요약들을 그냥 합쳐서 최종 요약으로 사용
    """
    if not text or not text.strip():
        return ""

    chunks = chunk_text(text, max_chars=chunk_max_chars, overlap=overlap)

    partial_summaries: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        print(f"    [SUM] chunk {i}/{len(chunks)} (len={len(ch)})")
        result = summarizer(
            ch,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )[0]["summary_text"]
        partial_summaries.append(result.strip())

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    # 🔴 두 번째 요약은 MPS OOM 위험이 크므로 생략
    combined = "\n".join(partial_summaries)
    print(f"    [SUM] combined summary length (no second pass): {len(combined)}")
    return combined.strip()



# -----------------------------
# 3. SBERT / FinBERT 임베딩 준비
# -----------------------------

def get_sbert_model(model_name: str = "sentence-transformers/all-mpnet-base-v2") -> SentenceTransformer:
    """
    SBERT 계열 SentenceTransformer 모델 로드.
    """
    return SentenceTransformer(model_name)


def get_finbert_model(model_name: str = "ProsusAI/finbert"):
    """
    FinBERT(일반 BERT 모델)를 로드하고,
    mean pooling 방식으로 문장 임베딩을 생성할 예정.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def embed_with_sbert(sbert_model: SentenceTransformer, text: str) -> List[float]:
    """
    SBERT로 텍스트 임베딩.
    SentenceTransformer는 내부적으로 mean pooling까지 해 줌.
    """
    emb = sbert_model.encode(text, convert_to_numpy=True)
    return emb.tolist()


def embed_with_finbert(tokenizer, model, device: str, text: str) -> List[float]:
    """
    FinBERT(BERT)로 텍스트 임베딩.
    - last_hidden_state의 mean pooling 사용 (토큰 평균)
    """
    if not text or not text.strip():
        return []

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # 필요시 조정
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        # [batch, seq_len, hidden_size] → [batch, hidden_size]
        token_embeddings = outputs.last_hidden_state  # (1, L, H)
        sentence_embedding = token_embeddings.mean(dim=1)[0]  # (H,)

    return sentence_embedding.cpu().tolist()


# -----------------------------
# 4. ChromaDB 초기화
# -----------------------------

def get_chroma_collections():
    """
    Chroma persistent client 생성 후,
    SBERT / FinBERT용 컬렉션 두 개를 준비.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    coll_sbert = client.get_or_create_collection(
        name="business_sbert",
        metadata={"description": "Summarized business section embeddings (SBERT)"}
    )

    coll_finbert = client.get_or_create_collection(
        name="business_finbert",
        metadata={"description": "Summarized business section embeddings (FinBERT)"}
    )

    return coll_sbert, coll_finbert


# -----------------------------
# 5. main: JSON → 요약 → 임베딩 → Chroma
# -----------------------------

def process_business_files(
    summarizer_model_name: str = "facebook/bart-large-cnn",
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    finbert_model_name: str = "ProsusAI/finbert",
):
    # 모델 로드
    print("[INIT] Loading summarization model...")
    summarizer = get_summarizer(summarizer_model_name)

    print("[INIT] Loading SBERT model...")
    sbert_model = get_sbert_model(sbert_model_name)

    print("[INIT] Loading FinBERT model...")
    finbert_tokenizer, finbert_model, finbert_device = get_finbert_model(finbert_model_name)

    print("[INIT] Connecting to ChromaDB...")
    coll_sbert, coll_finbert = get_chroma_collections()

    business_files = sorted(JSON_DIR.glob("*_business.json"))
    print(f"[INFO] Found {len(business_files)} business.json files in {JSON_DIR}")

    for idx, path in enumerate(business_files, start=1):
        print(f"\n[{idx}/{len(business_files)}] Processing {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        symbol = data.get("symbol") or path.stem.replace("_business", "")
        cik = data.get("cik")
        company_name = data.get("company_name")

        # 루트에 filing_* 정보가 없다면, business_by_year에서 가장 최근 연도 메타데이터 하나 가져오기
        business_by_year = data.get("business_by_year", {})
        filing_type = data.get("filing_type")
        filing_date = data.get("filing_date")
        filing_url = data.get("filing_url")

        if (filing_type is None or filing_date is None or filing_url is None) and isinstance(business_by_year, dict) and business_by_year:
            latest_year = sorted(business_by_year.keys())[-1]
            latest_meta = business_by_year.get(latest_year, {})
            filing_type = filing_type or latest_meta.get("filing_type")
            filing_date = filing_date or latest_meta.get("filing_date")
            filing_url = filing_url or latest_meta.get("filing_url")

        # ---- 여기서부터 수정된 부분: 텍스트 추출 ----
        text = get_business_text_from_data(data)

        if not text or not text.strip():
            print("  [WARN] business text is empty. Skip.")
            continue
        # ---- 수정 끝 ----

        # 1) 요약
        summary = summarize_long_text(summarizer, text)
        print(f"  [OK] Summary length: {len(summary)} chars")

        # 2) SBERT 임베딩
        sbert_vec = embed_with_sbert(sbert_model, summary)
        # 3) FinBERT 임베딩
        finbert_vec = embed_with_finbert(finbert_tokenizer, finbert_model, finbert_device, summary)

        # 4) 공통 메타데이터
        metadata: Dict[str, Optional[str]] = {
            "symbol": symbol,
            "cik": cik,
            "company_name": company_name,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "filing_url": filing_url,
            "source": "10-K Item 1 Business (summarized)",
        }

        # id는 symbol+filing_date 조합으로 유니크하게
        doc_id = f"{symbol}_{filing_date}" if filing_date else symbol

        # SBERT 컬렉션에 추가
        coll_sbert.add(
            ids=[doc_id],
            documents=[summary],
            embeddings=[sbert_vec],
            metadatas=[metadata],
        )
        print("  [CHROMA] Upserted into business_sbert")

        # FinBERT 컬렉션에 추가
        if finbert_vec:
            coll_finbert.add(
                ids=[doc_id],
                documents=[summary],
                embeddings=[finbert_vec],
                metadatas=[metadata],
            )
            print("  [CHROMA] Upserted into business_finbert")
        else:
            print("  [WARN] FinBERT embedding is empty. Skipped.")

    # ---------- JSON 누락 기업 체크 ----------
    numerical_files = sorted(JSON_DIR.glob("*_numerical.json"))

    business_symbols = {p.stem.replace("_business", "") for p in business_files}
    numerical_symbols = {p.stem.replace("_numerical", "") for p in numerical_files}

    missing_business = sorted(numerical_symbols - business_symbols)

    print("\n[CHECK] JSON consistency check")
    print(f"  numerical.json count : {len(numerical_symbols)}")
    print(f"  business.json count  : {len(business_symbols)}")

    if missing_business:
        print(f"  [WARN] {len(missing_business)} companies have numerical JSON but NO business JSON:")
        for sym in missing_business:
            print(f"    - {sym}")
    else:
        print("  [INFO] All companies with numerical JSON also have business JSON.")

    # ---------- ChromaDB 저장 완료 ----------
    print("\n[INFO] Done. All embeddings stored in", CHROMA_DIR)


if __name__ == "__main__":
    # 필요하면 아래 모델 이름을 프로젝트에 맞게 바꿔도 됨
    SUMMARIZER_MODEL = "facebook/bart-large-cnn"
    SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    FINBERT_MODEL = "ProsusAI/finbert"

    process_business_files(
        summarizer_model_name=SUMMARIZER_MODEL,
        sbert_model_name=SBERT_MODEL,
        finbert_model_name=FINBERT_MODEL,
    )
