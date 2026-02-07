#!/usr/bin/env python3
"""Test 3: EAGLE efficiency — measure tok/s WITHOUT EAGLE for comparison"""
import time
import json
import csv
import os
import statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")
MODEL = "zai-org/GLM-4.7-FP8"

CONTEXT_LENGTHS = [1024, 4096, 16384, 32768]
MAX_OUTPUT_TOKENS = 128
REPEATS = 3

FILLER_BLOCK = "The quick brown fox jumps over the lazy dog. " * 20

def build_prompt(target_tokens):
    target_chars = target_tokens * 4
    prompt = ""
    while len(prompt) < target_chars:
        prompt += FILLER_BLOCK
    return prompt[:target_chars]

def measure_streaming(context_len):
    prompt = build_prompt(context_len)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Continue the text naturally."},
        {"role": "user", "content": prompt + "\n\nPlease continue writing naturally:"}
    ]
    t_start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=MAX_OUTPUT_TOKENS,
            stream=True, temperature=0.7,
        )
        for chunk in stream:
            now = time.perf_counter()
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now
                token_count += 1
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
    print("=" * 70)
    print("  Test 3: EAGLE OFF — Decode Speed Baseline")
    print("=" * 70)
    results = []
    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n--- Context: {ctx_len} tokens ---")
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
        med_ttft = statistics.median(ttfts) if ttfts else 0
        med_decode = statistics.median(decode_rates) if decode_rates else 0
        results.append({
            "context_length": ctx_len,
            "ttft_ms": round(med_ttft, 1),
            "eagle_off_toks": round(med_decode, 2),
        })
    
    # Load EAGLE ON results from Test 1
    eagle_on = {}
    try:
        with open("results/test1_context_vs_speed.json") as f:
            for r in json.load(f):
                eagle_on[r["context_length"]] = r["decode_toks"]
    except:
        pass
    
    print("\n" + "=" * 70)
    print(f"{'Context':>8} | {'EAGLE OFF':>12} | {'EAGLE ON':>12} | {'Speedup':>8}")
    print("-" * 50)
    for r in results:
        on = eagle_on.get(r["context_length"], 0)
        speedup = on / r["eagle_off_toks"] if r["eagle_off_toks"] > 0 else 0
        r["eagle_on_toks"] = on
        r["speedup"] = round(speedup, 2)
        print(f"{r['context_length']:>8} | {r['eagle_off_toks']:>10.2f} | {on:>10.2f} | {speedup:>7.2f}x")
    
    csv_path = "results/test3_eagle_efficiency.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["context_length", "ttft_ms", "eagle_off_toks", "eagle_on_toks", "speedup"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")
    with open("results/test3_eagle_efficiency.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
