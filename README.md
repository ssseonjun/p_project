# p_project

SEC 10-K/10-Q 섹션을 구조화하고, 요약 후 SBERT/FinBERT 임베딩까지 이어지는 파이프라인 샘플입니다.

## 준비물
- `sec_filings/` 디렉터리에 10-K/10-Q 원문을 `.txt` 또는 `.html` 형태로 저장합니다. (예: `AAPL_2024_10K.txt`)
- Python 의존성 설치: `pip install -r requirements.txt`

## 섹션 추출
`sec_items_to_json.py`는 원문에서 Item 1/7/2(10-Q) 섹션을 찾아 `json/` 디렉터리에 저장합니다.

```bash
python sec_items_to_json.py --input-dir sec_filings --output-dir json
```

생성 예시 (`json/AAPL_2024_10K_sections.json`):
```json
{
  "source_file": "AAPL_2024_10K.txt",
  "form_type": "10-K",
  "item1": "Item 1. Business ...",
  "item7": "Item 7. Management's Discussion and Analysis ...",
  "item2": null
}
```

### 최신 10-K/10-Q를 SEC API로 내려받아 섹션 추출하기
`embed_SEC.py`는 위키백과에서 S&P 500 티커 목록을 받아 CIK로 변환한 뒤, 각 기업의 최신 10-K/10-Q를 내려받아 Item 1/7/2를 추출하고 요약·임베딩까지 한 번에 처리합니다.

```bash
python embed_SEC.py --user-agent "YourName Contact@example.com" --output-dir json --max-companies 5
```

- 섹션 JSON: `json/<TICKER>_latest_10K_sections.json`, `json/<TICKER>_latest_10Q_sections.json`
- 요약 JSON: `json/<TICKER>_latest_summaries.json` (전체 단락을 롱 컨텍스트 요약기로 병렬 처리)
- 임베딩 JSON: `chromaDB/sec/<TICKER>_latest_embeddings.json` (SBERT/FinBERT)

기본 chunk 예산은 약 4096 토큰(문자 기준 16384)이며, `--chunk-char-budget`와 `--summary-workers`로 조정해 롱 컨텍스트 요약 및 병렬 처리 속도를 맞출 수 있습니다.

## 요약 전처리
`sec_preprocess_for_summary.py`는 추출된 섹션을 깨끗하게 정제하고, 요약에 적합하도록 길이를 줄여 `json_clean/`에 저장합니다.

```bash
python sec_preprocess_for_summary.py --input-dir json --output-dir json_clean
```

각 섹션에 대해 HTML 잔여 태그 제거, 단락 필터링, 키워드 기반 점수화 후, 섹션별 문자 예산(예: Item 7 약 16k자) 내에서 상위 단락만 남깁니다.

## 요약 템플릿 및 파이프라인
- `sec_pipeline/summary_templates.py`: Item 1/7/2 요약 지시문 템플릿과, 임의의 `summarizer` 콜백을 받아 섹션별 요약을 생성하는 헬퍼.
- `sec_pipeline/workflow.py`: 단순 문장 추출 기반 `naive_summarizer`를 예시로 연결하고, 요약 JSON/임베딩 파일을 저장하는 유틸리티.

실제 서비스에서는 `summarizer` 자리에 LLM 호출 또는 커스텀 요약 함수를 주입하면 됩니다.

## 임베딩 파이프라인
`sec_pipeline/embedding_pipeline.py`는 SBERT(`all-MiniLM-L6-v2`)와 FinBERT(`yiyanghkust/finbert-tone`) 모델을 이용해 요약 텍스트를 임베딩합니다.

```python
from pathlib import Path
from sec_pipeline.workflow import build_summaries, embed_summaries
from sec_pipeline.sections import load_sections_from_json

sections = load_sections_from_json(Path("json/AAPL_2024_10K_sections.json"))
summaries = build_summaries(sections)
embeddings = embed_summaries(summaries, Path("embeddings/AAPL_2024_10K.json"))
```

출력 JSON 구조 예시:
```json
{
  "item1": {"sbert": [...], "finbert": [...]},
  "item7": {"sbert": [...], "finbert": [...]}
}
```

## 추천 필드 매핑 (예시)
- `raw_sections`: 추출된 원문 섹션
- `summary.item1|item7|item2`: 템플릿 기반 요약 결과
- `embeddings.item*.sbert/finbert`: 각 요약의 SBERT/FinBERT 벡터

이 구조를 벡터 DB 혹은 내부 노션/JSON 문서에 저장하면, 기업별/아이템별 검색·분석에 활용할 수 있습니다.
