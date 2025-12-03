from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    finbert_model: str = "yiyanghkust/finbert-tone"
    device: Optional[str] = None


class SummaryEmbeddingPipeline:
    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()):
        self.sbert = SentenceTransformer(config.sbert_model, device=config.device)
        self.finbert = SentenceTransformer(config.finbert_model, device=config.device)

    def embed_summary(self, summary_text: str) -> Dict[str, list]:
        return {
            "sbert": self.sbert.encode(summary_text, convert_to_numpy=True).tolist(),
            "finbert": self.finbert.encode(summary_text, convert_to_numpy=True).tolist(),
        }

    def embed_summary_payload(self, summaries: Dict[str, Optional[str]]) -> Dict[str, Dict[str, list]]:
        payload: Dict[str, Dict[str, list]] = {}
        for section_name, content in summaries.items():
            if not content:
                continue
            payload[section_name] = self.embed_summary(content)
        return payload

    def save_embeddings(self, embeddings: Dict[str, Dict[str, list]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self._format_json(embeddings), encoding="utf-8"
        )

    @staticmethod
    def _format_json(embeddings: Dict[str, Dict[str, list]]) -> str:
        import json

        return json.dumps(embeddings, ensure_ascii=False, indent=2)
