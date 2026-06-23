from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonus.agent import HybridMemoryAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_agent() -> HybridMemoryAgent:
    agent = HybridMemoryAgent(use_external_stores=False)
    memories = [
        "I read a Kubernetes guide about pod lifecycle, autoscaling, and cluster cost control.",
        "Cloud security notes: zero-trust access, TLS everywhere, and least privilege for services.",
        "Tài liệu AI giải thích RAG, embedding retrieval, và cách đánh giá hallucination.",
        "Vietnamese users often code-switch, so memory search should handle tiếng Việt and English terms together.",
        "Flutter offline-first apps need local cache, background sync, and careful push notification handling.",
        "Recent reading focused on scaling infrastructure with Kubernetes HPA and serverless workloads.",
    ]
    for text in memories:
        agent.remember(text, user_id="u_001")
    return agent


def main() -> int:
    agent = build_agent()
    queries = [
        "What have I read about Kubernetes?",
        "Recommend what to read next",
        "What am I focused on lately?",
        "Documents about scaling infrastructure?",
        "Give me a cloud security summary",
    ]
    for i, query in enumerate(queries, start=1):
        print("=" * 72)
        print(f"Query {i}: {query}")
        print(agent.recall(query, user_id="u_001"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
