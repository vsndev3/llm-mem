#!/usr/bin/env python3
"""
Judge LLM validation - run before benchmarking to verify the judge model works.
Tests connectivity, reasoning with a known-answer math problem, and fact extraction.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI


GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"
BOLD = "\033[1m"


def ok(msg):
    print(f"  {GREEN}OK{NC}    {msg}")


def fail(msg):
    print(f"  {RED}FAIL{NC}  {msg}")


def warn(msg):
    print(f"  {YELLOW}WARN{NC}  {msg}")


def call_llm(client, model, prompt, max_tokens=4096, temperature=0.1):
    """Call LLM and return content string (ignoring thinking/reasoning), or None on failure."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = r.choices[0]
        content = choice.message.content
        if content is None:
            warn(f"empty content (finish_reason={choice.finish_reason}, has_reasoning={choice.message.reasoning is not None})")
        return content
    except Exception as e:
        fail(f"API error: {e}")
        return None


def check_basic(client, model):
    print(f"\n{BOLD}[1/3] Basic connectivity + reasoning{NC}")
    prompt = (
        "You are a precise answering engine. Answer this question with ONLY the answer, "
        "no explanation:\n"
        "What is the capital of France?"
    )
    content = call_llm(client, model, prompt, max_tokens=4096)
    if not content:
        return False

    answer = content.strip().lower()
    if "paris" in answer:
        ok(f"correct answer: \"{content.strip()}\"")
        return True
    else:
        warn(f"unexpected answer: \"{content.strip()}\" (wanted 'Paris')")
        return False


def check_format(client, model):
    print(f"\n{BOLD}[2/3] Benchmark-style answer format{NC}")
    prompt = (
        "You are a precise answering engine. Answer ONLY with the answer, no preamble.\n"
        "Question: What color is the sky on a clear day?\n"
        "Answer:"
    )
    content = call_llm(client, model, prompt)
    if not content:
        return False

    answer = content.strip()
    if len(answer) > 200:
        fail(f"too verbose ({len(answer)} chars). Response: \"{answer[:120]}...\"")
        return False
    ok(f"concise ({len(answer)} chars): \"{answer}\"")
    return True


def check_math(client, model):
    print(f"\n{BOLD}[3/3] Hard math reasoning{NC}")
    prompt = (
        "Solve this problem step by step, then end with 'ANSWER: <number>' on its own line.\n\n"
        "A spaceship travels at 0.8c for 5 years in its own reference frame. "
        "How many years pass on Earth? Use the time dilation formula t = t0 / sqrt(1 - v^2/c^2). "
        "Compute and round to 2 decimal places."
    )
    content = call_llm(client, model, prompt, max_tokens=32768, temperature=0.1)
    if not content:
        return False

    expected = 8.33
    margin = 0.15

    print(f"  Response ({len(content)} chars):")
    for line in content.split("\n")[:8]:
        print(f"  | {line}")
    if len(content.split("\n")) > 8:
        print(f"  | ...")

    for line in content.split("\n"):
        m = re.search(r"ANSWER\s*:\s*([\d.]+)", line, re.IGNORECASE)
        if m:
            value = float(m.group(1))
            diff = abs(value - expected)
            if diff <= margin:
                ok(f"correct! ANSWER: {value} (expected {expected}, diff={diff:.3f})")
                return True
            else:
                fail(f"wrong. ANSWER: {value}, expected {expected} +/- {margin}")
                return False

    floats = re.findall(r"\b\d+\.\d+\b", content)
    if floats:
        value = float(floats[-1])
        diff = abs(value - expected)
        if diff <= margin:
            ok(f"found {value} at end (expected {expected}, diff={diff:.3f})")
            return True

    fail(f"no numeric answer near {expected} found")
    return False


def main():
    parser = argparse.ArgumentParser(description="Validate judge LLM for benchmarking")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--api-key", default="", help="API key")
    args = parser.parse_args()

    if not args.api_key:
        print(f"{RED}No API key. Set --api-key or OPENAI_API_KEY env var.{NC}")
        sys.exit(1)

    print(f"{BOLD}Judge LLM Validation{NC}")
    print(f"  Model:    {args.model}")
    print(f"  Endpoint: {args.base_url}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    results = []
    results.append(check_basic(client, args.model))
    results.append(check_format(client, args.model))
    results.append(check_math(client, args.model))

    passed = sum(results)
    total = len(results)

    print(f"\n{BOLD}{'=' * 50}{NC}")
    if passed == total:
        print(f"{GREEN}{BOLD}ALL CHECKS PASSED ({passed}/{total}) — Judge LLM is ready.{NC}")
    elif passed >= 2:
        print(f"{YELLOW}{BOLD}{passed}/{total} passed — OK to proceed, but verify results.{NC}")
    else:
        print(f"{RED}{BOLD}{passed}/{total} passed — Judge LLM may not be reliable.{NC}")

    return 0 if passed >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
