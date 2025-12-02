#embed_wiki_sbert.py
#벡터이름: wiki_sbert
import json
from pathlib import Path
from typing import List

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent

# 입력: 위키 json
JSON_DIR = BASE_DIR / "json" / "wiki_json"

# 출력: ChromaDB persistent path
CHROMA_PATH = BASE_DIR / "chromaDB" / "chroma_wiki"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)


def load_text_from_json(path: Path, field_candidates: List[str]) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for field in field_candidates:
        if field in data and isinstance(data[field], str):
            return data[field]
    # 재무제표 같이 dict 구조일 경우, 필요하면 여기서 텍스트 구성 로직 추가
    return ""


def main():
    input_dir = JSON_DIR  # 또는 나중에 "sec/item1_summary" 등으로 바꿔서 재사용 가능

    # 어떤 필드를 텍스트로 쓸지 입력 디렉토리별로 바꾸면 됨
    field_candidates = ["clean_text", "item1_summary", "item1_text"]

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    # Chroma 클라이언트 & 컬렉션 준비
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection("wiki_sbert")

    json_files = sorted(p for p in input_dir.glob("*.json") if not p.name.startswith("_"))
    print(f"[INFO] SBERT: Found {len(json_files)} wiki json files in {input_dir}")

    for path in tqdm(json_files, desc=f"SBERT embedding ({input_dir.name})"):
        text = load_text_from_json(path, field_candidates)
        if not text.strip():
            print(f"[WARN] empty text in {path}")
            continue

        emb = model.encode(text, show_progress_bar=False)
        emb = emb.tolist()

        doc_id = path.stem  # 예: "AAPL.json" -> "AAPL"
        metadata = {
            "file_name": path.name,
            "source": "wiki",
        }

        collection.upsert(
            ids=[doc_id],
            embeddings=[emb],
            metadatas=[metadata],
        )

    print("[INFO] SBERT wiki embeddings stored in Chroma collection 'wiki_sbert'")


if __name__ == "__main__":
    main()