# embed_sbert.py
import json
from pathlib import Path
from typing import List

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")

def load_text_from_json(path: Path, field_candidates: List[str]) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for field in field_candidates:
        if field in data and isinstance(data[field], str):
            return data[field]
    # 재무제표 같이 dict 구조일 경우, 필요하면 여기서 텍스트 구성 로직 추가
    return ""


def main():
    # 예시: 위키, Item1 요약, 재무 요약 등을 각각 다른 디렉토리로 호출 가능
    input_dir = DATA_DIR / "wiki_json"          # 또는 "sec/item1_summary", ...
    output_dir = DATA_DIR / "embeddings_sbert"  # 공통 저장소
    output_dir.mkdir(parents=True, exist_ok=True)

    # 어떤 필드를 텍스트로 쓸지 입력 디렉토리별로 바꾸면 됨
    # - wiki_json: "text"
    # - sec/item1_summary: "item1_summary"
    # - sec/financials: 나중에 "fin_text" 같은 필드 추가 후 사용
    field_candidates = ["text", "item1_summary", "item1_text"]

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    json_files = sorted(p for p in input_dir.glob("*.json") if not p.name.startswith("_"))

    for path in tqdm(json_files, desc=f"SBERT embedding ({input_dir.name})"):
        text = load_text_from_json(path, field_candidates)
        if not text.strip():
            print(f"[WARN] empty text in {path}")
            continue

        emb = model.encode(text, show_progress_bar=False)
        emb = emb.tolist()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["embedding_sbert"] = emb

        out_path = output_dir / path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()