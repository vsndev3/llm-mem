#!/usr/bin/env python3
"""
LongMemEval benchmark runner for llm-mem.

CRITICAL: The full history is NEVER pasted into the LLM context.
Only the top-K retrieved memories are fed in. This isolates
retrieval quality — the actual challenge llm-mem is solving.

Supports two modes:
  --pyramid-mode none    Flat L0 vector search (NaiveRAG baseline)
  --pyramid-mode bottom_heavy|balanced|top_heavy   Pyramid abstraction search
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import config, mcp_client, utils

logger = logging.getLogger("longmemeval")

STORE_TIMEOUT = 900
QUERY_TIMEOUT = 60
JUDGE_TIMEOUT = 300
ABSTRACTION_POLL_INTERVAL = 10
ABSTRACTION_MAX_WAIT = 1800
# A frozen layer state (no changes for this long) means the pipeline has
# settled: leftover items are stranded (e.g. an L1 without enough siblings
# to form an L2 group) and will never abstract. Proceed instead of aborting.
ABSTRACTION_SETTLE_GRACE = 180


def build_answer_prompt(question: str, memories: list[dict]) -> str:
    if not memories:
        return (
            f"Question: {question}\n\n"
            "You have no relevant context. If you cannot answer, say 'I don't know'.\n"
            "Answer:"
        )

    lines = ["Relevant memories:"]
    for i, mem in enumerate(memories):
        content = mem.get("content", mem.get("memory", str(mem)))
        lines.append(f"\n[{i + 1}] {content}")

    lines.append(f"\n\nQuestion: {question}")
    lines.append(
        "\nBased ONLY on the context above, answer the question concisely. "
        "If the context does not contain the answer, say 'I don't know'."
    )
    return "\n".join(lines)


async def judge_answer(
    question: str,
    retrieved_memories: list[dict],
    model: str,
    base_url: str,
    api_key: str,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    prompt = build_answer_prompt(question, retrieved_memories)

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
            ),
            timeout=JUDGE_TIMEOUT,
        )
        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            logger.warning("Judge returned empty content, finish_reason=%s",
                           choice.finish_reason)
            return "I don't know"
        return content.strip()
    except Exception as e:
        logger.error("Judge API call failed: %s", e)
        return "I don't know"


async def wait_for_abstraction(client: mcp_client.LlmMemClient, bank_name: str):
    """Poll pending abstraction counts until all layers settle.
    Timer resets to 0 whenever pending count drops (progress was made).
    PIPELINE_DEAD_GRACE seconds (pipeline likely crashed).
    """
    waited = 0
    prev_state = None
    no_progress_since = 0
    while waited < ABSTRACTION_MAX_WAIT:
        status = await client.system_status()
        pipeline = status.get("abstraction_pipeline", {})
        l0 = pipeline.get("pending_l0_count", 0)
        l1 = pipeline.get("pending_l1_count", 0)
        l2 = pipeline.get("pending_l2_count", 0)
        pending = l0 + l1 + l2

        if pending == 0 and waited >= 30:
            logger.info("Abstraction complete after %ds", waited)
            return

        # Progress = ANY layer-count change. An L0→L1 promotion keeps the
        # pending sum constant, so the sum alone cannot detect progress.
        if prev_state is not None and (l0, l1, l2) != prev_state:
            waited = 0
            no_progress_since = 0

        logger.info("Abstraction pending=%d (L0=%d L1=%d L2=%d), waited=%ds",
                    pending, l0, l1, l2, waited)

        if pending > 0 and no_progress_since >= ABSTRACTION_SETTLE_GRACE:
            logger.info(
                "Abstraction settled with pending L0=%d L1=%d L2=%d "
                "(no layer changes for %ds — stranded items). Proceeding.",
                l0, l1, l2, no_progress_since,
            )
            return

        await asyncio.sleep(ABSTRACTION_POLL_INTERVAL)
        prev_state = (l0, l1, l2)
        waited += ABSTRACTION_POLL_INTERVAL
        no_progress_since += ABSTRACTION_POLL_INTERVAL

    logger.warning("Abstraction did not settle after %ds, proceeding anyway", waited)


async def process_question(
    client: mcp_client.LlmMemClient,
    instance: dict,
    question_idx: int,
    total: int,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    pyramid_mode: str = "none",
    max_sessions: int = 0,
    retrieval_dump_path: Path | None = None,
) -> dict:
    question_id = instance["question_id"]
    question_text = instance["question"]
    question_type = instance.get("question_type", "unknown")
    bank_name = f"{config.BANK_PREFIX}_lme_{question_idx}"

    logger.info("[%d/%d] q=%s type=%s mode=%s",
                question_idx + 1, total, question_id, question_type, pyramid_mode)

    await client.create_bank(bank_name, f"LongMemEval question {question_id}")

    haystack_sessions = instance.get("haystack_sessions", [])
    if max_sessions > 0:
        haystack_sessions = haystack_sessions[:max_sessions]
    haystack_dates = instance.get("haystack_dates", [])

    for sess_idx, session in enumerate(haystack_sessions):
        ts = haystack_dates[sess_idx] if sess_idx < len(haystack_dates) else ""
        content = mcp_client.format_session(session, sess_idx, ts)
        preview = content[:120].replace("\n", " ").strip()
        t0 = time.time()
        try:
            logger.info("  Storing session %d/%d (%d bytes) %.70s...",
                        sess_idx + 1, len(haystack_sessions), len(content), preview)
            await asyncio.wait_for(
                client.store_content(content, memory_type="conversational"),
                timeout=STORE_TIMEOUT,
            )
            elapsed = time.time() - t0
            tok_sec = len(content) / 4 / elapsed if elapsed > 0 else 0
            logger.info("           session %d done in %.1fs (~%.0f tok/s)",
                        sess_idx + 1, elapsed, tok_sec)
        except asyncio.TimeoutError:
            logger.warning("Store timeout for session %d (%ds)", sess_idx, STORE_TIMEOUT)
        except Exception as e:
            # Near-duplicate rejections (repeated filler sessions in the
            # haystack) and other per-session store errors must not kill the
            # whole question — skip the session and keep storing.
            logger.warning("Store rejected for session %d: %.200s", sess_idx, e)

    if pyramid_mode != "none":
        await wait_for_abstraction(client, bank_name)

    query_kwargs: dict = {
        "query": question_text,
        "k": config.QUERY_K,
    }
    if config.SIMILARITY_THRESHOLD > 0:
        query_kwargs["similarity_threshold"] = config.SIMILARITY_THRESHOLD
    if pyramid_mode not in ("none", ""):
        query_kwargs["pyramid_mode"] = pyramid_mode
    granularity = os.environ.get("LLM_MEM_QUERY_GRANULARITY", "").strip().lower()
    if granularity:
        query_kwargs["granularity"] = granularity
        if os.environ.get("LLM_MEM_EXCERPT_MAX_CHARS", "").strip():
            query_kwargs["excerpt_max_chars"] = int(os.environ["LLM_MEM_EXCERPT_MAX_CHARS"])

    retrieved = await asyncio.wait_for(
        client.query(**query_kwargs),
        timeout=QUERY_TIMEOUT,
    )

    memories = []
    if isinstance(retrieved, dict):
        data = retrieved.get("data", retrieved)
        memories = data.get("memories", [])

    for i, mem in enumerate(memories):
        layer = mem.get("layer_name", "?")
        score = mem.get("score", 0)
        content = mem.get("content", mem.get("memory", ""))
        content_preview = (content or "")[:200].replace("\n", " ")
        logger.info("  Retrieved [%d] L=%s score=%.4f: %s",
                    i, layer, score, content_preview)

    # Sidecar dump for retrieval-grounding analysis
    if retrieval_dump_path is not None:
        rec = {
            "question_id": question_id,
            "question": question_text,
            "retrieved": [
                {
                    "i": i,
                    "layer": mem.get("layer_name", "?"),
                    "score": mem.get("score", 0),
                    "content": mem.get("content", mem.get("memory", "")),
                    **({"memory_id": mem["memory_id"]} if "memory_id" in mem else {}),
                }
                for i, mem in enumerate(memories)
            ],
        }
        with open(retrieval_dump_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    hypothesis = await judge_answer(
        question_text,
        memories,
        model=judge_model,
        base_url=judge_base_url,
        api_key=judge_api_key,
    )

    # Keep bank for post-run inspection with llm-mem CLI.
    # await client.cleanup_bank(bank_name)

    layer_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    for mem in memories:
        ln = mem.get("layer_name", "").replace("raw_content", "L0").replace("structural", "L1") \
            .replace("semantic", "L2").replace("concept", "L3").replace("wisdom", "L4")
        layer_counts[ln] = layer_counts.get(ln, 0) + 1

    return {
        "question_id": question_id,
        "hypothesis": hypothesis,
        "task_type": question_type,
        "num_retrieved": len(memories),
        "pyramid_mode": pyramid_mode,
        "layers": layer_counts,
    }


async def run_longmemeval(
    dataset_path: Path,
    output_path: Path,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    pyramid_mode: str = "none",
    start_from: int = 0,
    limit: int = 0,
    max_sessions: int = 0,
):
    with open(dataset_path) as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError("Expected JSON array of question instances")

    if limit > 0:
        questions = questions[start_from : start_from + limit]
    elif start_from > 0:
        questions = questions[start_from:]

    total = len(questions)
    logger.info("Loaded %d questions (pyramid=%s)", total, pyramid_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with mcp_client.mcp_session(str(config.CONFIG_PATH)) as client:
        logger.info("Warming up models (lazy-load triggers)...")
        await client.create_bank("_warmup", "model loading warm-up")
        try:
            await asyncio.wait_for(
                client.store_content("Warm-up. Ignore this.", memory_type="conversational"),
                timeout=120,
            )
        except asyncio.TimeoutError:
            logger.warning("Warm-up store timed out (120s), models may not be fully loaded")
        await client.cleanup_bank("_warmup")

        status = await client.system_status()
        sys_st = status.get("system_status", {})
        logger.info("System: embed=%s llm=%s backend=%s state=%s",
                    sys_st.get("embedding_available"),
                    sys_st.get("llm_available"),
                    sys_st.get("backend"),
                    sys_st.get("state"))

        if pyramid_mode != "none":
            await client.start_abstraction_pipeline()
            logger.info("Abstraction pipeline started")

        results = []
        retrieval_dump_path = output_path.parent / (output_path.name + ".retrieval.jsonl")
        for idx, instance in enumerate(questions):
            try:
                result = await process_question(
                    client, instance, idx, total,
                    judge_model, judge_base_url, judge_api_key,
                    pyramid_mode=pyramid_mode,
                    max_sessions=max_sessions,
                    retrieval_dump_path=retrieval_dump_path,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed question %s: %s", instance.get("question_id", "?"), e)
                results.append({
                    "question_id": instance.get("question_id", f"unknown_{idx}"),
                    "hypothesis": "I don't know",
                    "task_type": instance.get("question_type", "unknown"),
                    "pyramid_mode": pyramid_mode,
                    "error": str(e),
                })

            if (idx + 1) % 10 == 0 or pyramid_mode != "none":
                utils.write_jsonl(output_path, results)
                logger.info("Checkpoint: %d results → %s", len(results), output_path)

        utils.write_jsonl(output_path, results)
        logger.info("Done. %d results → %s", len(results), output_path)


def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark runner for llm-mem")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=config.JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=config.JUDGE_API_KEY)
    parser.add_argument("--pyramid-mode", default="none",
                        choices=["none", "bottom_heavy", "balanced", "top_heavy"])
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()

    utils.setup_logging(config.LOGS_DIR, f"longmemeval_{args.pyramid_mode}")

    if not args.judge_api_key:
        logger.error("No judge API key. Set LLM_MEM_JUDGE_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)

    asyncio.run(run_longmemeval(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        pyramid_mode=args.pyramid_mode,
        start_from=args.start_from,
        limit=args.limit,
        max_sessions=args.max_sessions,
    ))


if __name__ == "__main__":
    main()
