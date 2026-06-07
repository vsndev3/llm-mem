# The memory pyramid

llm-mem doesn't just dump everything into a flat pile. It organizes memories in **layers**, similar to how human memory works — from concrete facts up to abstract understanding.

| Layer | Name | What it is |
|---|---|---|
| **L0** | Raw content | Your conversations, notes, and documents as-is |
| **L1** | Summaries | AI-generated summaries and structural outlines |
| **L2** | Connections | Cross-document links and thematic relationships |
| **L3** | Concepts | Domain principles and theories |
| **L4** | Insights | Mental models, paradigms, and high-level patterns |

Background workers automatically create higher layers over time. You can search at any layer, zoom in to see sources of a concept, or zoom out to see what abstractions were built from something you stored.

## Why layers?

Flat search works for keyword matching but fails on questions like *"What pattern connects these accidental discoveries?"*. Pyramid search across layers can synthesize the answer.

The pyramid is also how human memory works:

| Human cognition | llm-mem layer | Example |
|---|---|---|
| Sensory input | **L0** raw | "Meeting at 3pm, Alice said PostgreSQL, see #arch-2026-02-15" |
| Episodic | **L1** summary | "Feb 15 architecture meeting" |
| Semantic | **L2** connection | "PostgreSQL chosen over MongoDB for analytics service" |
| Conceptual | **L3** concept | "Relational vs. document databases for join-heavy workloads" |
| Wisdom | **L4** insight | "Schema-first design reduces migration cost at scale" |

The L0 content is verbatim — exact original phrasing, file paths, line numbers, timestamps. Higher layers compress and connect, but you can always zoom back to the source.

## How layers are created

**L0 — Raw content**: created by you (or your AI assistant) when you call `add_content_memory`, `store_memories`, or `upload_document`. These are immutable entries.

**L1 — Summaries**: created automatically by a background worker that watches L0 content. When enough L0 content accumulates (controlled by `auto_summary_threshold`), the worker picks related memories and asks the LLM to produce a summary.

**L2 — Connections**: created by another background worker that looks at L1 summaries, finds semantic overlap, and emits `related_to`, `references`, `extends`, or `contradicts` relations.

**L3 — Concepts**: created by an L2→L3 worker that clusters L2 connections into domain concepts.

**L4 — Insights**: created by an L3→L4 worker that finds cross-domain patterns and abstracts them into principles.

You don't manage this pipeline. It runs in the background and adds higher layers as material accumulates. You can query at any layer, force a specific layer to be built (`create_abstraction`), or pause the pipeline (`stop_abstraction_pipeline` / `start_abstraction_pipeline`).

## Navigation

The `navigate_memory` tool moves through the pyramid:

- **zoom out** from a concrete L0 memory → see what L1+ insights it contributed to
- **zoom in** from an abstract L3 concept → see the L0 source evidence
- **search at layer** → restrict results to a specific abstraction level

This is the killer feature for research workflows. Ask "what did we decide about auth in February?" and you get raw meeting notes. Ask "what's our auth architecture?" and you get a synthesized concept with source links.

## The two timestamps

Every memory carries two distinct timestamps:

- **`event_at`** — *when the event actually happened* (caller-supplied for L0; derived as a range from source memories for L1+)
- **`created_at`** — *when the memory was stored* (set automatically)

This separation lets you answer "what happened in the last two days" without confusing it with "what was stored recently". Old memories without an explicit `event_at` fall back to `created_at` so the timeline stays complete.

## Next

Ready to run it? See [What you need](./what-you-need.md) and then [Installation](./installation.md).
