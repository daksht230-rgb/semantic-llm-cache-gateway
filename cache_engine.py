import time
import hashlib
from typing import Dict, Any, Optional, List
from collections import OrderedDict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCacheEngine:
    """
    Production-ready Semantic Caching Engine for LLM APIs.
    Intercepts semantically similar prompts using TF-IDF n-gram cosine similarity,
    enforces TTL expiration, manages LRU eviction, and tracks token/cost savings.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        max_entries: int = 500,
        default_ttl_seconds: int = 3600,
        cost_per_1k_tokens_usd: float = 0.005,
    ):
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if default_ttl_seconds < 1:
            raise ValueError("default_ttl_seconds must be at least 1")

        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.cost_per_1k_tokens_usd = cost_per_1k_tokens_usd

        # OrderedDict maintains insertion/access order for O(1) LRU eviction
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self._matrix = None
        self._keys_index: List[str] = []
        self._index_dirty: bool = False

        # Telemetry metrics
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "expired_purged": 0,
            "estimated_tokens_saved": 0,
            "estimated_cost_saved_usd": 0.0,
        }

    def _generate_id(self, prompt: str) -> str:
        normalized = " ".join(prompt.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _purge_expired(self) -> int:
        now = time.time()
        expired_keys = [
            key for key, entry in self._store.items() if now > entry["expires_at"]
        ]
        for key in expired_keys:
            del self._store[key]

        if expired_keys:
            self.metrics["expired_purged"] += len(expired_keys)
            self._index_dirty = True

        return len(expired_keys)

    def _ensure_index(self) -> None:
        if not self._index_dirty and self._matrix is not None:
            return

        self._keys_index = list(self._store.keys())
        if not self._keys_index:
            self._matrix = None
            self._index_dirty = False
            return

        corpus = [self._store[key]["prompt"] for key in self._keys_index]
        try:
            self._matrix = self._vectorizer.fit_transform(corpus)
        except ValueError:
            self._matrix = None

        self._index_dirty = False

    def lookup(
        self, prompt: str, threshold_override: Optional[float] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self.metrics["total_requests"] += 1
        self._purge_expired()

        threshold = (
            threshold_override if threshold_override is not None else self.similarity_threshold
        )
        clean_prompt = " ".join(prompt.strip().split())

        if not clean_prompt:
            self.metrics["cache_misses"] += 1
            latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
            return {
                "hit": False,
                "match_type": None,
                "similarity_score": 0.0,
                "cached_prompt": None,
                "response": None,
                "model": None,
                "latency_ms": latency_ms,
            }

        # 1. Fast Exact-Hash Check (normalized whitespace & casing)
        exact_key = self._generate_id(clean_prompt)
        if exact_key in self._store:
            entry = self._store[exact_key]
            self._store.move_to_end(exact_key)
            entry["hit_count"] += 1
            self._record_hit(entry["tokens_used"])
            latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
            return {
                "hit": True,
                "match_type": "exact",
                "similarity_score": 1.0,
                "cached_prompt": entry["prompt"],
                "response": entry["response"],
                "model": entry["model"],
                "latency_ms": latency_ms,
            }

        # 2. Semantic Vector Similarity Search
        self._ensure_index()
        if self._matrix is not None and len(self._keys_index) > 0:
            try:
                query_vec = self._vectorizer.transform([clean_prompt])
                similarities = cosine_similarity(query_vec, self._matrix).flatten()
                if similarities.size > 0:
                    best_idx = int(np.argmax(similarities))
                    best_score = float(similarities[best_idx])

                    if best_score >= threshold:
                        matched_key = self._keys_index[best_idx]
                        entry = self._store[matched_key]
                        self._store.move_to_end(matched_key)
                        entry["hit_count"] += 1
                        self._record_hit(entry["tokens_used"])
                        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
                        return {
                            "hit": True,
                            "match_type": "semantic",
                            "similarity_score": round(best_score, 4),
                            "cached_prompt": entry["prompt"],
                            "response": entry["response"],
                            "model": entry["model"],
                            "latency_ms": latency_ms,
                        }
            except ValueError:
                pass

        self.metrics["cache_misses"] += 1
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        return {
            "hit": False,
            "match_type": None,
            "similarity_score": 0.0,
            "cached_prompt": None,
            "response": None,
            "model": None,
            "latency_ms": latency_ms,
        }

    def store(
        self,
        prompt: str,
        response: str,
        model: str = "gpt-4o",
        tokens_used: int = 350,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._purge_expired()
        clean_prompt = " ".join(prompt.strip().split())
        if not clean_prompt:
            raise ValueError("Prompt cannot be empty.")

        key = self._generate_id(clean_prompt)

        if key not in self._store and len(self._store) >= self.max_entries:
            self._store.popitem(last=False)
            self.metrics["evictions"] += 1

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        now = time.time()

        self._store[key] = {
            "prompt": clean_prompt,
            "response": response,
            "model": model,
            "tokens_used": max(0, tokens_used),
            "hit_count": 0,
            "created_at": now,
            "expires_at": now + ttl,
        }
        self._store.move_to_end(key)
        self._index_dirty = True

        return {
            "cache_id": key,
            "status": "stored",
            "ttl_seconds": ttl,
            "active_entries": len(self._store),
        }

    def _record_hit(self, tokens_saved: int) -> None:
        self.metrics["cache_hits"] += 1
        self.metrics["estimated_tokens_saved"] += tokens_saved
        saved_cost = (tokens_saved / 1000.0) * self.cost_per_1k_tokens_usd
        self.metrics["estimated_cost_saved_usd"] = round(
            self.metrics["estimated_cost_saved_usd"] + saved_cost, 6
        )

    def get_stats(self) -> Dict[str, Any]:
        self._purge_expired()
        total = self.metrics["total_requests"]
        hit_rate = round((self.metrics["cache_hits"] / total) * 100, 2) if total > 0 else 0.0
        return {
            "active_entries": len(self._store),
            "max_entries": self.max_entries,
            "similarity_threshold": self.similarity_threshold,
            "hit_rate_pct": hit_rate,
            **self.metrics,
        }

    def clear(self) -> None:
        self._store.clear()
        self._keys_index = []
        self._matrix = None
        self._index_dirty = False
