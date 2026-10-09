# A CJK draft vocabulary for Qwen3.8-Flash-Next on two DGX Sparks

This is **Mia's AI Lab's recipe for Qwen3.8-Flash-Next on two DGX Sparks** (TensorFold's Zig engine, NVFP4, 1,048,576-token
window, 16 concurrent, fp8 KV, vision) with **one change**: the engine's embedded *draft vocabulary* is replaced with a
CJK-extended list, and the three width pins that carry it are widened. The change is a fork pinned to the same engine
commit the recipe ships, so no other behaviour moves.

**Measured on two DGX Sparks (GB10, TP2), through the recipe's own API, greedy, thinking off:**

| | stock (upstream recipe) | this fork |
|---|---|---|
| Chinese draft acceptance | 16.7% | **49.7%** |
| Chinese prose, single stream | 22.1 tok/s | **45.8 tok/s** (+107%) |
| Chinese prose / code / chat (six-scene harness) | 32.9 / 56.1 / 33.0 | **50.3 / 77.6 / 49.4** (+38 … +53%) |
| Six-scene mean | 53.5 | **61.8** |
| Four concurrent streams (Chinese prompt) | 97.5 | **140.0 tok/s** (+44%) |
| English prose / code / chat | 51.4 / 89.1 / 58.5 | 50.1 / 86.1 / 57.1 (−2 to −3%) |
| Mia's own published protocol (prose, N=1/4/8/16) | 70.2 / 158.4 / 235.4 / 363.5 | 68.3 / 156.9 / 230.9 / 378.3 (within noise) |
| `tools/exact.py` — drafted == `"draft": false` == concurrent | PASS | **PASS** (text and token sha identical) |

Chinese gets **38–107% faster**; English costs **2–3%**; every reply is **byte-identical** to stock.

## Why the shipped engine cannot draft Chinese

The Zig engine's draft head scores only the ids in a shipped list, and that list is English/code-tuned: `79,591` ids from a
Python-stdlib corpus. Decoding every id with the checkpoint's own tokenizer:

```
ids in the shipped draft list whose text holds CJK:   122 / 79,591  = 0.15%   (and they are punctuation: 。，：「」【】？！)
CJK tokens in the whole vocabulary:                   55,571
…of which the draft list covers:                      122          = 0.2%
的 / 了 / 是 / 深度 / 秋天                            none of them are in it
```

A token outside the list **cannot be proposed at all**, so Chinese drafting is not weak — it is off. Drafts are proposals
and the target verifies every token, which makes decode speed almost purely a function of acceptance: with drafts off the
engine sits at the memory-bandwidth floor (~30 tok/s for ~6B active parameters a token, measured), and each accepted draft
multiplies that.

Measured acceptance on this pair, stock list: **16.7%** Chinese prose, **47.9%** English prose, **61.0%** English code —
and the decode rates follow it exactly: 22 / 33 / 89 tok/s.

## What changed

| file | change |
|---|---|
| `patches/0003-config-weights.patch` | the embedded list `79,591 → 135,040` ids (default ∪ every id whose `decode()` holds CJK: +55,449 Chinese ids, **41.15% CJK**), plus the test that asserts the list's length |
| `patches/0004-triton-replay-torch-ops.patch` | the replay fixture widths: `WIDTHS` and the `drafts` list gain `135040` and the TP=2 half `67520` |
| `patches/0008-kernel-spec-tools.patch` | the AOT kernel-spec generator's `for vocab in (…)` tuple gains the same two widths |

Nothing else is touched: the engine, the scripts, the checkpoint pin and the API are upstream's. The wider head needs **no
new hand-written kernels** — the AOT spec build (which the image does anyway) covers it, and it adds no measurable English
cost at one request per stream (the list is read once per draft step; one rank reads 79,591 → 135,040 rows, two ranks
39,796 → 67,520 each).

## Reproduce it

```bash
# 1. build the union list (needs the checkpoint's tokenizer; `tools/draft_vocab_extend.py` upstream does the same thing)
python3 tools/draft_vocab_extend.py <snapshot>/tokenizer.json <stock draft list> draft_vocab_cjk.txt --script cjk

# 2. build and serve exactly as the recipe says (the image's kernel set is built during the build; no GPU needed for it)
cp scripts/local.sh.example scripts/local.sh   # set WORKER=user@<worker>
./start.sh                                     # ~85 s to LIVE on a warm pair

# 3. verify: exactness first, then the point of the fork
PORT=8888 MODEL=Qwen3.8-Flash-Next python3 tools/exact.py            # drafted == serial, concurrent == alone
python3 tools/draft_ab.py --base-url http://127.0.0.1:8888/v1 --model Qwen3.8-Flash-Next
```

`tools/draft_ab.py` and `tools/sparkdash_decode_bench.py` in this repository are ours, not upstream's:

- **`draft_ab.py`** — the same prompts with drafts as served and with the request-level `"draft": false` knob, printing
  tok/s and the accepted/drafted counters the reply carries. That pairing is what shows a speed change is draft
  acceptance and not something else.
- **`sparkdash_decode_bench.py`** — a dependency-free port of `sparkDash`'s `DecodeBench` (`server/collectors/DecodeBench.js`
  and `src/shared/llmPrompts.js`): the same prompts, the same sampling (temperature 0, top_p 1, thinking off, 32-token
  warmup), the same timing — `decode tok/s = (completion_tokens − 1) / (last content chunk − first content chunk)`, and a
  unique suffix per concurrent stream so streams cannot share a prefix-cache block. It exists so anyone can re-measure the
  recipe's published table without standing up the dashboard. On our pair it reproduced the published numbers within a few
  percent (70.2 / 158.4 / 235.4 / 363.5 against 67.4 / 152.2 / 231.2 / 390.1 at N=1/4/8/16).

## Honest notes

- **Different rulers disagree, and that is expected.** On Mia's protocol (English hash-map prose) the fork is within noise
  of stock; on a six-scene harness that includes 1200-token Chinese continuations the Chinese gain dominates. Quote the
  ruler with the number.
- **One measurement caveat.** In our probe runs the `"draft": false` (serial) baselines differed between sessions
  (Chinese 16.4 vs 31.0 tok/s), so the "English prose +46%" a single probe run showed is not trustworthy — the acceptance
  counters, the published protocol, and the multi-prompt harness (all showing English flat) are what we report.
- **This fork is for the build the recipe currently pins** (`TF_REF=db281878`, image tag `zig-db28187-f3e1300f4f62`). Newer
  TensorFold engines write the draft list to `{cache_dir}/draft_vocab.txt` and build the pack from it, so there the list
  is a file and needs no rebuild — but that line does not yet qualify Qwen-Flash-Next on GB10 CUDA in its platform table.
  When it does, the list below is all you need.
- **Upstream already took the tool.** `tools/draft_vocab_extend.py` here is the one from our PR
  [ashhart/TensorFold#416](https://github.com/ashhart/TensorFold/pull/416), which the TensorFold author merged into the
  repository when he rewrote the fix for the Zig engine — the same analysis (a token outside the list can never be drafted)
  is what made the case.

## Credits and licence

- **The recipe, scripts and this README's basis**: [Mia's AI Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold),
  Apache-2.0 — `LICENSE` and `NOTICE` are kept as they ship, with our modification recorded in `NOTICE`.
- **TensorFold** and its Zig engine, by [ashhart](https://github.com/ashhart/TensorFold) and contributors, MIT; the Zig CUDA
  serving path and family registry by Jürgen Schmied ([PR #443](https://github.com/ashhart/TensorFold/pull/443)).
- **Qwen3.8-Flash-Next** by the Qwen team; the NVFP4 checkpoint by NVIDIA (`nvidia/Qwen3.8-Flash-Next-NVFP4`, made with
  NVIDIA Model Optimizer).
- The CJK draft list is derived data: the tokenizer's ids filtered by script, unioned with the shipped list — no weights are
  redistributed here.
- Our changes in this fork are Apache-2.0, same as the recipe.
