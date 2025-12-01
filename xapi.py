# main.py
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv

# .env 파일에서 환경변수 불러오기
load_dotenv()

# .env에 넣어둔 Bearer Token 읽어오기
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")
if not BEARER_TOKEN:
    raise RuntimeError("X_BEARER_TOKEN 환경변수가 설정되어 있지 않습니다.")

# X API v2 최근 트윗 검색 엔드포인트
BASE_URL = "https://api.x.com/2/tweets/search/recent"
# 만약 이 주소가 안 되면 아래 주석 풀고 위는 주석 처리해서 테스트:
# BASE_URL = "https://api.twitter.com/2/tweets/search/recent"

app = FastAPI(
    title="Xtock Xignal Backend",
    description="X API 연동 테스트용 백엔드",
)


async def call_x_recent_search(
    query: str,
    max_results: int = 10,
    next_token: Optional[str] = None,
):
    """
    실제로 X API를 호출하는 함수
    """
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
    }

    params = {
        "query": query,
        "max_results": max_results,
        # 필요한 필드는 여기서 추가 가능
        "tweet.fields": "created_at,author_id,public_metrics,lang",
    }
    if next_token:
        params["next_token"] = next_token  # 페이지네이션 용

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(BASE_URL, headers=headers, params=params)

    # 200이 아니면 에러로 보내기
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail={"msg": "X API error", "body": resp.json()},
        )

    return resp.json()


@app.get("/api/tweets")
async def get_tweets(
    q: str = Query(..., description="검색 쿼리 (예: $TSLA, from:elonmusk 등)"),
    max_results: int = Query(10, ge=10, le=100),
    next_token: Optional[str] = Query(None, description="페이지네이션용 next_token"),
):
    """
    프론트엔드에서 사용할 엔드포인트

    예시:
    - /api/tweets?q=$TSLA
    - /api/tweets?q=from:elonmusk
    - /api/tweets?q="NVIDIA stock"
    """
    data = await call_x_recent_search(q, max_results=max_results, next_token=next_token)

    # 나중에 여기서 필요한 필드만 뽑아서 리턴해도 됨
    return {
        "query": q,
        "max_results": max_results,
        "raw": data,
    }