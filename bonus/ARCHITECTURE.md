# Hybrid Memory Assistant Architecture

Contributors: Luu Thien Viet Cuong, Nguyen Hoai Bao Ngoc

## Architecture Diagram

```text
User notes / chats / read documents
        |
        v
Memory ingestion
  - chunk text
  - infer topic metadata
  - embed chunk
        |
        +---------------------> Vector store: episodic memories
        |                       filtered by user_id, ranked by semantic match
        |
User profile events
  - language preference
  - reading speed
  - topic affinity
  - active hours
        |
        +---------------------> Feature store: stable profile features
        |
Recent query events
  - queries_last_hour
  - distinct topics
  - latest query text
        |
        +---------------------> Streaming-style recent activity features

Recall request
        |
        v
Hybrid retrieval: BM25-style keyword rank + vector rank + RRF
        |
        v
Context assembler: profile + activity + top memories
        |
        v
LLM final answer
```

## Decision 1: Chunking Strategy

The POC chunks episodic memory as short semantic notes, usually one saved note
or paragraph at a time. For a real assistant, the target chunk size would be
around 300-500 tokens with metadata such as `user_id`, timestamp, source, and
inferred topic. I chose this instead of per-conversation storage because recall
quality is better when each chunk has one clear idea. A whole conversation may
contain Kubernetes, security, and mobile topics together; embedding it as one
large vector makes retrieval blurry. The tradeoff is storage cost and indexing
overhead: paragraph chunks create more vectors than conversation chunks.

I also considered per-message chunks. They are cheap and easy to update, but
Vietnamese users often mix short English technical terms with Vietnamese
explanations, so a single message can be too thin to retrieve well. Short
semantic chunks keep enough context for vector search while staying small
enough for an LLM context window. This follows the lab lesson from the vector
store notebooks: embeddings are useful only when the text unit has enough
semantic signal.

## Decision 2: Feature Schema

Stable profile data belongs in a feature store, not in the vector store. The POC
uses tabular features: preferred language, reading speed, topic affinity, active
hours, queries in the last hour, and distinct topics in the last day. Each row is
keyed by `user_id`. Stable fields such as language and reading speed can have a
long TTL, while recent activity should have a short TTL because stale activity
would personalize the answer incorrectly.

I chose simple tabular features instead of embedding features for v1. Embedding
features can represent latent preference, but they are harder to debug and less
transparent for an instructor or product owner. If the assistant recommends a
cloud security article, it should be easy to inspect that the user has
`topic_affinity=cloud, security` and recent queries about Kubernetes. This is
the same reason Feast is useful in the core lab: it gives consistent offline and
online feature lookup, and point-in-time joins help avoid using future data
during training.

## Decision 3: Freshness Strategy

Freshness should depend on the type of memory. New saved notes and newly read
documents should be searchable immediately, so `remember()` upserts them into
episodic memory at once. Recent activity should update near real time, because a
query like "What am I focused on lately?" depends on the last few actions. Stable
profile fields can refresh daily or weekly because reading speed and language
preference do not change minute by minute.

This is a tradeoff between latency, infrastructure complexity, and correctness.
Sub-second streaming is best for recent queries and fraud-like signals, but it is
unnecessary for stable profile features. A five-minute batch can be acceptable
for document-read summaries. A daily refresh is enough for slow preferences. The
POC simulates the streaming feature view with an in-memory recent-query list,
while the architecture keeps the boundary clear so a real Feast online store or
Redis-backed stream can replace it.

## Rejected Alternative

I considered storing episodic memory inside the feature store as an embedding
feature view, but rejected it. Episodic memory and profile features have
different access patterns. Memories need top-k nearest-neighbor retrieval,
payload filtering, and frequent inserts. Feature views need keyed lookup,
materialization, TTL, and point-in-time correctness. Combining them would make
both systems harder to reason about. The cleaner design is to keep episodic
recall in a vector store and stable profile in a feature store, then join their
outputs only when assembling context.

## Vietnamese Context

For Vietnamese users, whitespace tokenization is only a baseline. The POC keeps
it simple, but a production version should compare whitespace, `pyvi`, and
`underthesea` tokenization. Code-switching is common: a query might say "summary
cloud security" while the notes say "bảo mật đám mây". Hybrid search helps
because BM25 catches exact English technical terms and vector search catches
paraphrased Vietnamese meaning. The system also needs user isolation by
`user_id`, and a real product should add privacy controls such as deletion,
encryption at rest, and clear consent for storing personal memories.

## What This POC Does Not Handle Yet

This POC does not implement encryption, memory deletion, multi-device sync,
production authentication, or true streaming ingestion. It also uses a small
hash embedding fallback so the demo can run without downloads. When the lab
environment is installed, the same interface can use fastembed and Qdrant
in-memory mode, matching the vector-store pattern from the core notebooks.
