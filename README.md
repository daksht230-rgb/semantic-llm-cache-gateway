# Semantic LLM Cache Gateway

A lightweight, sub-millisecond **Semantic Caching Middleware** built with FastAPI and Scikit-Learn that intercepts redundant or paraphrased LLM prompts before they hit expensive upstream model APIs.

## Why This Exists
In production RAG and customer-facing AI assistants, **30%–45% of incoming queries are semantic paraphrases** of previously answered questions (e.g., *"How do I reset my API key?"* vs. *"Steps to regenerate workspace API token"*). Standard key-value caches (Redis exact-match) miss these completely.

`Semantic LLM Cache Gateway` bridges that gap using vectorized cosine similarity, TTL expiration, and O(1) LRU eviction—cutting token spend and response latency from ~1.2s down to **< 3ms**.

## Key Capabilities
- **Dual-Stage Lookup:** Fast SHA-256 exact-hash check followed by TF-IDF n-gram cosine similarity search.
- **LRU + TTL Memory Management:** Automatic expiration purging and bounded memory eviction (`OrderedDict`).
- **Live Savings Telemetry:** Tracks `hit_rate_pct`, `estimated_tokens_saved`, and `estimated_cost_saved_usd`.
- **Drop-In REST API:** Clean FastAPI + Pydantic v2 endpoints ready to sit in front of OpenAI, Anthropic, or local vLLM endpoints.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --reload
curl -X POST [http://127.0.0.1:8000/v1/gateway/query](http://127.0.0.1:8000/v1/gateway/query) \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset my workspace API key?"}'
curl -X POST [http://127.0.0.1:8000/v1/gateway/query](http://127.0.0.1:8000/v1/gateway/query) \
  -H "Content-Type: application/json" \
  -d '{"prompt": "reset workspace API key steps", "similarity_threshold": 0.70}'
curl [http://127.0.0.1:8000/v1/cache/stats](http://127.0.0.1:8000/v1/cache/stats)
