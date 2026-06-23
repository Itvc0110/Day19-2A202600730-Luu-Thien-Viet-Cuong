"""Minimal hybrid memory agent for the Day 19 bonus challenge.

The class keeps the POC runnable on a clean laptop by falling back to local
in-memory retrieval. When qdrant-client and fastembed are installed, callers
can opt into the same vector-store pattern used by the core lab.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


@dataclass
class Memory:
    memory_id: int
    user_id: str
    text: str
    created_at: datetime
    topic: str
    source: str = "saved_note"
    vector: list[float] = field(default_factory=list)


class HybridMemoryAgent:
    """POC assistant memory with episodic recall plus stable profile features."""

    def __init__(self, use_external_stores: bool = True, vector_dim: int = 64) -> None:
        self.vector_dim = vector_dim
        self.memories: list[Memory] = []
        self.recent_queries: dict[str, list[str]] = {}
        self.user_profiles: dict[str, dict[str, Any]] = {
            "u_001": {
                "preferred_language": "vi/en mix",
                "reading_speed_wpm": 230,
                "topic_affinity": "cloud, AI, security",
                "active_hours": "20:00-23:00",
            }
        }
        self._client = None
        self._embedder = None
        self._collection = "bonus_memory"
        self._external_ready = False
        if use_external_stores:
            self._init_external_vector_store()

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        for chunk in self._chunk_text(text):
            memory = Memory(
                memory_id=len(self.memories),
                user_id=user_id,
                text=chunk,
                created_at=datetime.now(timezone.utc),
                topic=self._infer_topic(chunk),
                vector=self._embed_text(chunk),
            )
            self.memories.append(memory)
            self._upsert_external(memory)

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top memories and profile features, then assemble LLM context."""
        self.recent_queries.setdefault(user_id, []).append(query)
        ranked = self._hybrid_rank(query, user_id)[:3]
        profile = self.user_profiles.get(user_id, self._default_profile(user_id))
        recent = self.recent_queries.get(user_id, [])[-5:]

        memory_lines = [
            f"{i}. [{m.topic}] {m.text}" for i, m in enumerate(ranked, start=1)
        ] or ["1. No saved memories matched this query yet."]

        return "\n".join(
            [
                "User profile:",
                (
                    f"- language={profile['preferred_language']}; "
                    f"reading_speed={profile['reading_speed_wpm']}wpm; "
                    f"topic_affinity={profile['topic_affinity']}; "
                    f"active_hours={profile['active_hours']}"
                ),
                "Recent activity:",
                f"- queries_last_hour={len(recent)}; latest_queries={recent}",
                "Top memories:",
                *memory_lines,
                "Assembled context for LLM:",
                (
                    "Use the profile as stable personalization, recent activity as freshness "
                    "signal, and top memories as evidence. Do not answer beyond this context."
                ),
            ]
        )

    def _init_external_vector_store(self) -> None:
        try:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._client = QdrantClient(":memory:")
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            self._external_ready = True
        except Exception:
            self._client = None
            self._embedder = None
            self._external_ready = False

    def _upsert_external(self, memory: Memory) -> None:
        if not self._external_ready or self._client is None:
            return
        try:
            from qdrant_client.models import PointStruct

            vector = next(self._embedder.embed([memory.text])).tolist()
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=memory.memory_id,
                        vector=vector,
                        payload={
                            "user_id": memory.user_id,
                            "text": memory.text,
                            "topic": memory.topic,
                        },
                    )
                ],
            )
        except Exception:
            self._external_ready = False

    def _chunk_text(self, text: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return []
        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            words = sentence.split()
            if current and current_words + len(words) > 90:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            current.append(sentence)
            current_words += len(words)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _hybrid_rank(self, query: str, user_id: str, rrf_k: int = 60) -> list[Memory]:
        candidates = [m for m in self.memories if m.user_id == user_id]
        if not candidates:
            return []

        keyword_rank = self._keyword_rank(query, candidates)
        semantic_rank = self._semantic_rank(query, candidates)
        scores: dict[int, float] = {}
        by_id = {m.memory_id: m for m in candidates}
        for ranked in (keyword_rank, semantic_rank):
            for rank, memory in enumerate(ranked, start=1):
                scores[memory.memory_id] = scores.get(memory.memory_id, 0.0) + 1.0 / (
                    rrf_k + rank
                )
        return [by_id[mid] for mid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    def _keyword_rank(self, query: str, memories: list[Memory]) -> list[Memory]:
        q_tokens = set(self._tokens(query))
        scored = []
        for memory in memories:
            m_tokens = self._tokens(memory.text)
            overlap = sum(1 for token in m_tokens if token in q_tokens)
            scored.append((overlap / max(len(m_tokens), 1), memory))
        return [m for score, m in sorted(scored, key=lambda item: -item[0]) if score > 0] or memories

    def _semantic_rank(self, query: str, memories: list[Memory]) -> list[Memory]:
        q_vec = self._embed_text(query)
        scored = [(self._cosine(q_vec, memory.vector), memory) for memory in memories]
        return [m for _, m in sorted(scored, key=lambda item: -item[0])]

    def _embed_text(self, text: str) -> list[float]:
        tokens = self._tokens(text)
        vector = [0.0] * self.vector_dim
        for token in tokens:
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.vector_dim
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _tokens(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def _cosine(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _infer_topic(self, text: str) -> str:
        lowered = text.lower()
        topic_terms = {
            "cloud": ["cloud", "kubernetes", "autoscaling", "hạ tầng", "serverless"],
            "security": ["security", "mã hoá", "zero-trust", "jwt", "tls"],
            "ai": ["ai", "llm", "rag", "embedding", "mô hình"],
            "mobile": ["flutter", "mobile", "offline", "android", "ios"],
        }
        for topic, terms in topic_terms.items():
            if any(term in lowered for term in terms):
                return topic
        return "general"

    def _default_profile(self, user_id: str) -> dict[str, Any]:
        profile = {
            "preferred_language": "vi",
            "reading_speed_wpm": 210,
            "topic_affinity": "cloud",
            "active_hours": "19:00-22:00",
        }
        self.user_profiles[user_id] = profile
        return profile
