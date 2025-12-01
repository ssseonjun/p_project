# embed_finbert.py
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

JSON_DIR = Path("json")
VEC_DIR = Path("chromaDB")


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
    input_dir = JSON_DIR / "wiki_json"           # 또는 "sec/item1_summary", ...
    output_dir = VEC_DIR / "embeddings_finbert"
    output_dir.mkdir(parents=True, exist_ok=True)

    field_candidates = ["text", "item1_summary", "item1_text"]

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    json_files = sorted(p for p in input_dir.glob("*.json") if not p.name.startswith("_"))

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
        emb = emb.detach().cpu().numpy()[0].tolist()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["embedding_finbert"] = emb

        out_path = output_dir / path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
