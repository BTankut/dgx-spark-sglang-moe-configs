#!/usr/bin/env python3
"""Test 4: Thinking Mode Impact on Agentic Performance"""
import time
import json
import csv
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")
MODEL = "zai-org/GLM-4.7-FP8"

tools = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read contents of a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
    }},
]

PROMPT = """You have access to tools for reading files and running commands.

Task: Read the file /etc/hostname, then write a Python script called system_info.py that prints the hostname and current date. After writing it, run the script and show me the output.

Start by reading /etc/hostname."""

MODES = [
    {"name": "Thinking OFF", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    {"name": "Thinking ON", "extra_body": {}},  # default
    {"name": "Preserved Thinking", "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}}},
]

def measure_mode(mode_config):
    """Measure a single mode"""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": PROMPT},
    ]
    
    t_start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    tool_calls_acc = {}
    content_acc = ""
    reasoning_acc = ""
    
    try:
        kwargs = dict(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto",
            max_tokens=1024, stream=True, temperature=0.7,
        )
        if mode_config.get("extra_body"):
            kwargs["extra_body"] = mode_config["extra_body"]
        
        stream = client.chat.completions.create(**kwargs)
        
        for chunk in stream:
            now = time.perf_counter()
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
                    token_count += 1
                    content_acc += delta.content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
                    token_count += 1
                    reasoning_acc += delta.reasoning_content
                if delta.tool_calls:
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"name": "", "arguments": ""}
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments
                        token_count += 1
        
        t_end = time.perf_counter()
        
        if first_token_time is None:
            return {"ttft_ms": 0, "total_ms": 0, "decode_toks": 0, "tool_calls": 0, "correct": False, "reasoning_tokens": 0}
        
        ttft_ms = (first_token_time - t_start) * 1000
        total_ms = (t_end - t_start) * 1000
        if token_count > 1 and last_token_time > first_token_time:
            decode_toks = (token_count - 1) / (last_token_time - first_token_time)
        else:
            decode_toks = 0
        
        # Check correctness: should call read_file with /etc/hostname
        correct = False
        for tc in tool_calls_acc.values():
            if tc["name"] == "read_file" and "hostname" in tc["arguments"]:
                correct = True
                break
        
        return {
            "ttft_ms": round(ttft_ms, 1),
            "total_ms": round(total_ms, 1),
            "decode_toks": round(decode_toks, 2),
            "tool_calls": len(tool_calls_acc),
            "correct": correct,
            "reasoning_tokens": len(reasoning_acc) // 4,  # rough estimate
            "tools": tool_calls_acc,
            "content": content_acc[:200],
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"ttft_ms": 0, "total_ms": 0, "decode_toks": 0, "tool_calls": 0, "correct": False, "reasoning_tokens": 0}

def main():
    os.makedirs("results", exist_ok=True)
    
    print("=" * 90)
    print("  Test 4: Thinking Mode Impact on Agentic Performance")
    print("=" * 90)
    
    results = []
    
    for mode in MODES:
        print(f"\n--- {mode['name']} ---")
        result = measure_mode(mode)
        
        icon = "✅" if result["correct"] else "❌"
        print(f"  TTFT={result['ttft_ms']:.0f}ms, Total={result['total_ms']:.0f}ms, Decode={result['decode_toks']:.1f} tok/s")
        print(f"  Tool calls: {result['tool_calls']}, Correct: {icon}")
        if result.get("tools"):
            for idx, tc in result["tools"].items():
                print(f"    [{idx}] {tc['name']}({tc['arguments'][:60]})")
        if result["reasoning_tokens"] > 0:
            print(f"  Reasoning tokens: ~{result['reasoning_tokens']}")
        if result.get("content"):
            print(f"  Content: {result['content'][:100]}...")
        
        results.append({
            "mode": mode["name"],
            "ttft_ms": result["ttft_ms"],
            "total_ms": result["total_ms"],
            "decode_toks": result["decode_toks"],
            "tool_calls": result["tool_calls"],
            "correct": result["correct"],
            "reasoning_tokens": result.get("reasoning_tokens", 0),
        })
    
    # Summary
    print("\n" + "=" * 90)
    print(f"{'Mode':>22} | {'TTFT (ms)':>10} | {'Total (ms)':>11} | {'Decode tok/s':>13} | {'Tools':>6} | {'OK':>4}")
    print("-" * 80)
    for r in results:
        icon = "✅" if r["correct"] else "❌"
        print(f"{r['mode']:>22} | {r['ttft_ms']:>10.1f} | {r['total_ms']:>11.1f} | {r['decode_toks']:>13.2f} | {r['tool_calls']:>6} | {icon:>4}")
    
    # Save
    csv_path = "results/test4_thinking_mode.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")
    
    with open("results/test4_thinking_mode.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
