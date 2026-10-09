#!/usr/bin/env python3
"""Reproduce Mia's published decode numbers with plain Python — a port of sparkDash's DecodeBench.

The recipe's "Performance" table is measured with `sparkDash` (github.com/MiaAI-Lab/sparkDash):
`server/collectors/DecodeBench.js` for the protocol and `src/shared/llmPrompts.js` for the prompts. Running that
Node dashboard just to verify a number is heavy, so this is a faithful port of both — same prompts, same timing,
same sampling — with no dependencies beyond the standard library.

Protocol, copied from the source:
  * concurrency levels run one after another; every type uses the same sampling: temperature 0, top_p 1, thinking off
  * a 32-token warmup request first; default reply budget 400 tokens
  * decode tok/s = (completion_tokens - 1) / (last content chunk - first content chunk)   [streaming; excludes TTFT]
  * concurrent streams get a unique suffix " (stream i/n)" so they cannot share a prefix-cache block; the code type
    uses a different task per stream, exactly as the source does

Usage:
  python3 sparkdash_decode_bench.py --base-url http://127.0.0.1:8888/v1 --model Qwen3.8-Flash-Next \
      [--api-key KEY] [--type prose|structured|code|json] [--max-tokens 400] [--levels 1,2,4,8,16] [--out FILE]
"""
import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

PROSE = ("Write a detailed step-by-step explanation of how a hash map works, including collision "
         "handling, resizing, and time complexity. Be thorough.")
STRUCTURED = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
CODE_TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
             "Then add tests and the helpers this needs. Keep writing code.")
CODE_TASKS = [
    "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.",
    "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.",
    "lru_cache\nclass LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used.",
    "token_bucket\nclass TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False.",
    "ring_buffer\nclass RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow.",
    "dijkstra\ndef dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph.",
    "edit_distance\ndef edit_distance(a, b) -> int: Levenshtein distance between two strings.",
    "semver_cmp\ndef semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b.",
]
JSON_PROMPT = ("Emit a long JSON catalog of GPU metrics with many entries: gpu, memory, temperature, power, "
               "utilization. JSON only. Keep adding entries.")
PROMPTS = {"prose": PROSE, "structured": STRUCTURED, "json": JSON_PROMPT}
WARMUP_TOKENS = 32


def prompts_for(kind, n):
    if kind == "code":
        out = []
        for i in range(n):
            task = CODE_TASKS[i % len(CODE_TASKS)]
            spec = f"{task}\n{CODE_TAIL}"
            out.append(spec if i < len(CODE_TASKS) else f"[stream {i + 1}]\n{spec}")
        return out
    base = PROMPTS[kind]
    return [base] if n <= 1 else [f"{base} (stream {i + 1}/{n})" for i in range(n)]


def one(url, model, key, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "chat_template_kwargs": {"thinking": False},
            "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    t0 = time.time()
    first = last = None
    deltas = 0
    usage_tokens = None
    with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers),
                                timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens")
            for choice in chunk.get("choices") or []:
                if (choice.get("delta") or {}).get("content"):
                    deltas += 1
                    now = time.time()
                    first = first or now
                    last = now
    tokens = usage_tokens or deltas
    if not first or not last or last <= first or tokens < 2:
        return tokens, 0.0, (first - t0) if first else 0.0
    return tokens, (tokens - 1) / (last - first), first - t0


def wave(url, model, key, kind, n, max_tokens):
    prompts = prompts_for(kind, n)
    out, lock, barrier = {}, threading.Lock(), threading.Barrier(n)

    def run(i):
        barrier.wait()
        r = one(url, model, key, prompts[i], max_tokens)
        with lock:
            out[i] = r
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    tokens = sum(v[0] for v in out.values())
    per = [v[1] for v in out.values() if v[1] > 0]
    ttfts = [v[2] for v in out.values() if v[2] > 0]
    return {"requests": n, "aggregate_tok_s": round(tokens / wall, 1),
            "per_request_median_tok_s": round(statistics.median(per), 1) if per else 0.0,
            "ttft_median_ms": round(statistics.median(ttfts) * 1000) if ttfts else 0,
            "tokens": tokens, "wall_s": round(wall, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--type", default="prose", choices=sorted(PROMPTS) + ["code"])
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    url = a.base_url.rstrip("/") + "/chat/completions"
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    print(f"== sparkDash DecodeBench port: type={a.type} max_tokens={a.max_tokens} levels={levels}")
    print("   temperature 0 / top_p 1 / thinking off / 32-token warmup; decode tok/s excludes TTFT")
    one(url, a.model, a.api_key, prompts_for(a.type, 1)[0], WARMUP_TOKENS)
    rows = []
    for n in levels:
        r = wave(url, a.model, a.api_key, a.type, n, a.max_tokens)
        rows.append(r)
        print(f"   N={n:2d}: aggregate {r['aggregate_tok_s']:7.1f} tok/s | per request (median) "
              f"{r['per_request_median_tok_s']:6.1f} | TTFT (median) {r['ttft_median_ms']:5d} ms | "
              f"{r['tokens']} tok / {r['wall_s']}s", flush=True)
        time.sleep(3)
    if a.out:
        json.dump({"type": a.type, "max_tokens": a.max_tokens, "base_url": a.base_url, "rows": rows},
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
