from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from cache_engine import SemanticCacheEngine

app = FastAPI(
    title="Semantic LLM Cache Gateway",
    description="High-throughput semantic caching middleware that intercepts redundant LLM queries to cut API spend and sub-millisecond response latency.",
    version="1.0.0",
)

cache = SemanticCacheEngine(
    similarity_threshold=0.75,
    max_entries=1000,
    default_ttl_seconds=3600,
)


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=2, example="How do I reset my workspace API key?")
    model: str = Field(default="gpt-4o", example="gpt-4o")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.1, le=1.0)


class StoreCacheRequest(BaseModel):
    prompt: str = Field(..., min_length=2)
    response: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4o")
    tokens_used: int = Field(default=350, ge=1)
    ttl_seconds: Optional[int] = Field(default=None, ge=10)


def _mock_upstream_llm(prompt: str, model: str) -> tuple[str, int]:
    """Simulates an upstream LLM completion when a cache miss occurs."""
    simulated_answer = (
        f"[{model} Generated Response] To handle '{prompt}', navigate to Settings > Developer > API Keys, "
        "revoke the compromised token, and generate a scoped replacement key."
    )
    estimated_tokens = max(120, len(prompt.split()) * 18)
    return simulated_answer, estimated_tokens


@app.post("/v1/gateway/query", summary="Query LLM Through Semantic Cache")
def query_with_cache(payload: QueryRequest):
    lookup_result = cache.lookup(
        prompt=payload.prompt,
        threshold_override=payload.similarity_threshold,
    )

    if lookup_result["hit"]:
        return {
            "status": "success",
            "source": "semantic_cache",
            "cache_meta": {
                "match_type": lookup_result["match_type"],
                "similarity_score": lookup_result["similarity_score"],
                "matched_prompt": lookup_result["cached_prompt"],
                "lookup_latency_ms": lookup_result["latency_ms"],
            },
            "response": lookup_result["response"],
        }

    # Cache Miss -> Call upstream LLM and populate semantic cache
    answer, tokens_used = _mock_upstream_llm(payload.prompt, payload.model)
    cache.store(
        prompt=payload.prompt,
        response=answer,
        model=payload.model,
        tokens_used=tokens_used,
    )

    return {
        "status": "success",
        "source": "upstream_llm",
        "cache_meta": {
            "match_type": "miss",
            "similarity_score": 0.0,
            "stored_for_future": True,
            "lookup_latency_ms": lookup_result["latency_ms"],
        },
        "response": answer,
    }


@app.post("/v1/cache/store", summary="Manually Seed Prompt-Response Pair")
def seed_cache_entry(payload: StoreCacheRequest):
    return cache.store(
        prompt=payload.prompt,
        response=payload.response,
        model=payload.model,
        tokens_used=payload.tokens_used,
        ttl_seconds=payload.ttl_seconds,
    )


@app.get("/v1/cache/stats", summary="Inspect Hit Rate & Cost Savings Telemetry")
def get_cache_telemetry():
    return {"status": "success", "telemetry": cache.get_stats()}


@app.delete("/v1/cache/flush", summary="Flush All Cached Entries")
def flush_cache():
    cache.clear()
    return {"status": "success", "message": "Semantic cache flushed."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
