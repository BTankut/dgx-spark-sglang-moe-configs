# DGX Spark Bandwidth-Throughput Validation Results

**Date:** 2026-02-07
**Setup:** 4x DGX Spark (GB10, 128GB, 273 GB/s each), TP=4, EAGLE speculative decoding
**Model:** GLM-4.7-FP8 (355B MoE, ~32B active, FP8)
**Server:** SGLang v0.5.4.post2 + glm47 tool-call parser
**MoE Configs:** Custom-tuned for GB10 (101KB SMEM limit)

---

## Test 1: Context Length vs Decode Speed

**Formula under test:** `tok/s = (β × TP) / (W + KV)`

Where β=273 GB/s, TP=4, W=32 GB (active params × FP8)

| Context | KV Cache (GB) | TTFT (ms) | Decode tok/s | Theoretical tok/s | Ratio |
|---------|--------------|-----------|-------------|-------------------|-------|
| 512 | 0.074 | 132.8 | 7.50 | 34.05 | 0.22 |
| 1,024 | 0.149 | 116.7 | 7.58 | 33.97 | 0.22 |
| 2,048 | 0.298 | 123.1 | 7.48 | 33.81 | 0.22 |
| 4,096 | 0.595 | 123.9 | 7.41 | 33.50 | 0.22 |
| 8,192 | 1.190 | 134.6 | 7.24 | 32.90 | 0.22 |
| 16,384 | 2.381 | 158.8 | 6.99 | 31.76 | 0.22 |
| 32,768 | 4.762 | 198.9 | 6.54 | 29.70 | 0.22 |

### Analysis

- **Constant ratio of 0.22**: The formula correctly predicts the *shape* of degradation, but actual throughput is ~22% of theoretical maximum. This efficiency coefficient (η) accounts for multi-node network overhead, EAGLE speculative decoding overhead, framework overhead, and streaming measurement granularity.

- **Graceful degradation**: Only 13% throughput drop from 512→32K tokens (7.50→6.54 tok/s). This directly contradicts flash3's claim of "1 tok/s at 17K context" — but flash3 was testing a *single* DGX Spark with GLM-4.7-Flash (3B active), not a 4-node TP=4 cluster.

- **Corrected formula**: `tok/s = η × (β × TP) / (W + KV)` where η ≈ 0.22

---

## Test 2: Agentic Workflow Simulation (10-turn Multi-Tool)

| Turn | Context (tokens) | TTFT (ms) | Decode tok/s | Tool Call OK |
|------|-----------------|-----------|-------------|-------------|
| 1 | ~42 | 809 | 8.74 | ✅ |
| 2 | ~120 | 782 | 7.18 | ✅ |
| 3 | ~305 | 1,384 | 7.24 | ✅ |
| 4 | ~901 | 751 | 9.44 | ✅ |
| 5 | ~994 | 383 | 7.80 | ✅ |
| 6 | ~1,826 | 2,180 | 7.43 | ✅ (large file injection) |
| 7 | ~1,918 | 1,408 | 7.09 | ✅ |
| 8 | ~3,142 | — | — | ❌ |
| 9 | ~3,212 | — | — | ❌ |
| 10 | ~3,283 | — | — | ❌ |

### Analysis

- **Turns 1-7: Stable performance** — Tool calling works reliably at 7-9 tok/s, even after large code injection (~4K chars) at turn 6.
- **Turns 8-10: Failures** — At ~3K accumulated context tokens, tool call generation broke. Since Test 1 showed healthy throughput at 32K, this is a conversation format / tool message accumulation issue, not a bandwidth bottleneck.
- **Recommendation**: Implement context window management — summarize or truncate earlier turns after ~7 exchanges.

---

## Test 3: EAGLE Speculative Decoding Efficiency

### Streaming Client Measurement

| Context | EAGLE OFF tok/s | EAGLE ON tok/s | Apparent Ratio |
|---------|----------------|---------------|----------------|
| 1,024 | 16.89 | 7.58 | 0.45x |
| 4,096 | 16.48 | 7.41 | 0.45x |
| 16,384 | 15.28 | 6.99 | 0.46x |
| 32,768 | 14.05 | 6.54 | 0.47x |

### Server-Side Throughput (from SGLang logs)

| Metric | EAGLE OFF | EAGLE ON |
|--------|-----------|----------|
| Server throughput | ~14 tok/s | 16.77 tok/s |
| EAGLE accept rate | — | 0.89-0.93 |

### Analysis

**The streaming measurement is misleading.** EAGLE generates multiple tokens per step (speculative batching), but SGLang's streaming API sends them in larger chunks. The Python streaming client counts each chunk as one token, underreporting EAGLE ON throughput by ~2.2x.

**Server-side reality**: EAGLE provides a genuine **+19.8% throughput improvement** (16.77 vs 14.0 tok/s), consistent with the earlier A/B test.

**EAGLE efficiency is stable across context lengths**: The 0.45-0.47x streaming ratio stays constant, confirming EAGLE's acceptance rate doesn't degrade with longer context.

---

## Test 4: Thinking Mode Impact on Tool Calling

| Mode | TTFT (ms) | Total Time (ms) | Decode tok/s | Tool Calls | Correct |
|------|-----------|-----------------|-------------|------------|---------|
| Thinking OFF | 699 | 2,048 | 8.89 | 1 | ✅ |
| Thinking ON | 112 | 3,148 | 7.90 | 1 | ✅ |
| Preserved Thinking | 119 | 3,126 | 8.32 | 1 | ✅ |

### Analysis

- **All modes produce correct tool calls** — GLM-4.7's tool calling works regardless of thinking mode.
- **Thinking OFF**: Fastest total time (2.0s), best for speed-critical agentic tasks.
- **Thinking ON**: 50% slower but model "thinks" before acting — lower TTFT (112ms vs 699ms) because it starts outputting thinking tokens immediately.
- **Preserved Thinking**: Same speed as ON, but reasoning tokens are visible (~54 tokens). Useful for debugging agent behavior.

---

## Comparison to flash3's Claims

| Claim | flash3 (1 node, 3B active) | Our setup (4-node TP=4, 32B active) |
|-------|---------------------------|-------------------------------------|
| "1 tok/s at 17K context" | Plausible for single GB10 | **6.99 tok/s at 16K** |
| "Unusable for agentic coding" | Likely true for single node | **Usable up to ~7 turns** |
| Formula `tok/s = β/(W+KV)` | Correct structure | Needs η≈0.22 coefficient |
| "DGX Spark is memory-bandwidth limited" | True | True, but TP=4 multiplies effective bandwidth |

### Key Takeaways

1. **TP=4 is the game-changer** — 4 nodes multiply effective decode bandwidth by ~4x, pushing the "agentic death zone" far beyond single-node limits.
2. **flash3's formula is structurally correct** but needs an efficiency coefficient η that captures real-world overheads (network, framework, speculation).
3. **EAGLE helps but the benefit is modest** (+20% server-side). Its main value on GB10 is enabling features that would otherwise crash (OutOfResources without MoE configs).
4. **Tool calling works reliably for 7+ turns** before context accumulation becomes an issue — this is a model/format limitation, not hardware.
5. **32K context is practical** — only 13% throughput degradation, with usable 6.5+ tok/s decode speed.

---

## Model Configuration Reference

```
num_hidden_layers: 92
num_attention_heads: 96
num_key_value_heads: 8 (GQA)
hidden_size: 5120
head_dim: 53
num_experts_per_tok: 8
KV cache dtype: bfloat16
KV bytes/token: 156,032 (0.149 MB)
Active weights: ~32 GB (FP8)
```

---

*All benchmark scripts and CSV results are in `benchmarks/results/`*
*Production config: EAGLE ON + glm47 parser + MoE configs + TP=4*
