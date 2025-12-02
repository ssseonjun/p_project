#embed_wiki_finbert.py
#벡터이름: wiki_finbert
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
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
    return ""


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / (
        input_mask_expanded.sum(1) + 1e-9
    )


def main():
    input_dir = JSON_DIR  # 또는 나중에 "sec/item1_summary" 등으로 바꿔서 재사용 가능

    field_candidates = ["clean_text", "item1_summary", "item1_text"]

    # FinBERT 모델
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Chroma 클라이언트 & 컬렉션 준비
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection("wiki_finbert")

    json_files = sorted(p for p in input_dir.glob("*.json") if not p.name.startswith("_"))
    print(f"[INFO] FinBERT: Found {len(json_files)} wiki json files in {input_dir}")

    for path in tqdm(json_files, desc=f"FinBERT embedding ({input_dir.name})"):
        text = load_text_from_json(path, field_candidates)
        if not text.strip():
            print(f"[WARN] empty text in {path}")
            continue

        enc = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            emb = mean_pooling(out, enc["attention_mask"])
        emb = emb.detach().cpu().numpy()[0].tolist()  # Chroma는 list 형태 기대

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

    print("[INFO] FinBERT wiki embeddings stored in Chroma collection 'wiki_finbert'")


if __name__ == "__main__":
    main()