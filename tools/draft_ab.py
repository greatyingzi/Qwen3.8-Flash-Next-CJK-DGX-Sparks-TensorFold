#!/usr/bin/env python3
"""Where the speed comes from: draft acceptance, measured per language.

The engine's drafts are proposals — the target model verifies every token, so replies are unchanged either way. That
means decode speed here is (almost) entirely a function of **draft acceptance**: with drafts off the engine runs at
the bandwidth floor (one round per ~6B active parameters a token), and every accepted draft multiplies that.

This asks the same three prompts twice, once with drafts as served and once with the request-level `"draft": false`
knob the engine exposes, and prints tok/s plus the accepted/drafted counters the reply carries.

Usage:
  python3 draft_ab.py --base-url http://127.0.0.1:8888/v1 --model Qwen3.8-Flash-Next [--api-key KEY] [--reps 2]
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

PROMPTS = {
    "zh_prose": "写一段200字的秋天散文，要求文笔细腻。",
    "en_prose": "Write 200 words of prose about autumn, with fine imagery.",
    "en_code": "Write a Python quicksort implementation with comments.",
    "zh_code": "用 Python 写一个快速排序函数，并解释思路。",
}


def one(url, model, key, prompt, drafts, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    if not drafts:
        body["draft"] = False
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers), timeout=1800).read())
    dt = time.time() - t0
    tokens = r["usage"]["completion_tokens"]
    # the Python engine reports a "tensorfold" block, the Zig engine a "speculative" one; accept either
    stats = r.get("tensorfold") or r.get("speculative") or {}
    return {"tok_s": tokens / dt, "tokens": tokens,
            "drafted": stats.get("drafted"), "accepted": stats.get("accepted")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    url = a.base_url.rstrip("/") + "/chat/completions"
    print(f"== draft A/B  (max_tokens={a.max_tokens}, reps={a.reps}, temperature 0, thinking off)")
    for name, prompt in PROMPTS.items():
        for label, drafts in (("drafts ON ", True), ("draft:false", False)):
            tps, drafted, accepted = [], 0, 0
            for _ in range(a.reps):
                r = one(url, a.model, a.api_key, prompt, drafts, a.max_tokens)
                tps.append(r["tok_s"])
                drafted += r["drafted"] or 0
                accepted += r["accepted"] or 0
                time.sleep(3)
            acc = f"{accepted / drafted * 100:.1f}%" if drafted else "-"
            print(f"  {name:9s} {label}: {statistics.median(tps):6.1f} tok/s | "
                  f"drafted {drafted:5d} accepted {accepted:5d} ({acc})", flush=True)


if __name__ == "__main__":
    main()
