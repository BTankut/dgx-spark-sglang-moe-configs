#!/usr/bin/env python3
"""Test 1: Context Length vs Decode Speed — validates flash3's bandwidth formula"""
import time
import json
import csv
import os
import statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")
MODEL = "zai-org/GLM-4.7-FP8"

# Model config for theoretical calculation
NUM_LAYERS = 92
NUM_KV_HEADS = 8
HEAD_DIM = 53  # 5120 / 96
KV_DTYPE_BYTES = 2  # bf16
ACTIVE_WEIGHTS_GB = 32  # ~32B active params × 1 byte FP8
BANDWIDTH_PER_NODE = 273  # GB/s
TP = 4
# With TP=4, effective bandwidth for decode = TP * bandwidth (each node reads its shard in parallel)
# But there's network overhead for all-reduce after each layer
EFFECTIVE_BANDWIDTH = BANDWIDTH_PER_NODE * TP  # theoretical max = 1092 GB/s

CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
MAX_OUTPUT_TOKENS = 128
REPEATS = 3

# Generate filler text (~1.3 chars per token for English, but let's use a known ratio)
FILLER_BLOCK = "The quick brown fox jumps over the lazy dog. " * 20  # ~200 tokens worth

def kv_cache_gb(context_len):
    """Calculate KV cache size in GB for given context length"""
    kv_bytes = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * KV_DTYPE_BYTES * context_len
    return kv_bytes / (1024**3)

def theoretical_toks(context_len):
    """Calculate theoretical tok/s using flash3's formula adapted for TP"""
    kv = kv_cache_gb(context_len)
    # Each node processes W/TP weights + KV/TP cache
    # tok/s = bandwidth / ((W + KV) / TP)  = (bandwidth * TP) / (W + KV)
    return EFFECTIVE_BANDWIDTH / (ACTIVE_WEIGHTS_GB + kv)

def build_prompt(target_tokens):
    """Build a prompt approximately target_tokens long"""
    # Rough estimate: 1 token ≈ 4 chars for English
    target_chars = target_tokens * 4
    prompt = ""
    while len(prompt) < target_chars:
        prompt += FILLER_BLOCK
    prompt = prompt[:target_chars]
    return prompt

def measure_streaming(context_len):
    """Send a streaming request and measure TTFT and decode tok/s"""
    prompt = build_prompt(context_len)
    system_msg = "You are a helpful assistant. Continue the text naturally."
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt + "\n\nPlease continue writing naturally:"}
    ]
    
    t_start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
            temperature=0.7,
        )
        
        for chunk in stream:
            now = time.perf_counter()
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                token_count += 1  # approximate: 1 chunk ≈ 1 token for streaming
        
        if first_token_time is None:
            return None, None, 0
            
        ttft_ms = (first_token_time - t_start) * 1000
        if token_count > 1 and last_token_time > first_token_time:
            decode_toks = (token_count - 1) / (last_token_time - first_token_time)
        else:
            decode_toks = 0
            
        return ttft_ms, decode_toks, token_count
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None, 0

def main():
    os.makedirs("results", exist_ok=True)
    
    print("=" * 90)
    print("  Test 1: Context Length vs Decode Speed")
    print("  Formula: tok/s = β×TP / (W + KV)")
    print(f"  β={BANDWIDTH_PER_NODE} GB/s, TP={TP}, W={ACTIVE_WEIGHTS_GB} GB")
    print("=" * 90)
    
    results = []
    
    for ctx_len in CONTEXT_LENGTHS:
        kv = kv_cache_gb(ctx_len)
        theo = theoretical_toks(ctx_len)
        
        print(f"\n--- Context: {ctx_len} tokens (KV={kv:.2f} GB, Theoretical={theo:.1f} tok/s) ---")
        
        ttfts = []
        decode_rates = []
        
        for r in range(REPEATS):
            print(f"  Round {r+1}/{REPEATS}...", end=" ", flush=True)
            ttft, decode, tokens = measure_streaming(ctx_len)
            if ttft is not None:
                ttfts.append(ttft)
                decode_rates.append(decode)
                print(f"TTFT={ttft:.0f}ms, Decode={decode:.1f} tok/s ({tokens} tokens)")
            else:
                print("FAILED")
        
        if ttfts:
            med_ttft = statistics.median(ttfts)
            med_decode = statistics.median(decode_rates)
        else:
            med_ttft = 0
            med_decode = 0
        
        results.append({
            "context_length": ctx_len,
            "kv_cache_gb": round(kv, 3),
            "ttft_ms": round(med_ttft, 1),
            "decode_toks": round(med_decode, 2),
            "theoretical_toks": round(theo, 2),
            "ratio": round(med_decode / theo, 2) if theo > 0 and med_decode > 0 else 0,
        })
    
    # Print summary
    print("\n" + "=" * 90)
    print(f"{'Context':>8} | {'KV (GB)':>8} | {'TTFT (ms)':>10} | {'Decode tok/s':>13} | {'Theory tok/s':>13} | {'Ratio':>6}")
    print("-" * 90)
    for r in results:
        print(f"{r['context_length']:>8} | {r['kv_cache_gb']:>8.3f} | {r['ttft_ms']:>10.1f} | {r['decode_toks']:>13.2f} | {r['theoretical_toks']:>13.2f} | {r['ratio']:>6.2f}")
    
    # Save CSV
    csv_path = "results/test1_context_vs_speed.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")
    
    # Save JSON too
    with open("results/test1_context_vs_speed.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
