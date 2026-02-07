#!/usr/bin/env python3
"""Test 2: Agentic Workflow Simulation — multi-turn tool calling with growing context"""
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
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write content to a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}, "content": {"type": "string", "description": "File content"}}, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command"}}, "required": ["command"]}
    }},
    {"type": "function", "function": {
        "name": "list_directory", "description": "List contents of a directory",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
]

# Simulated tool responses
FAKE_HOSTNAME = "dgxnode1"
FAKE_OS_RELEASE = """NAME="Ubuntu"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 22.04.3 LTS"
VERSION_ID="22.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=jammy
UBUNTU_CODENAME=jammy"""

# Large code file for stress test at turn 6
LARGE_CODE = '''#!/usr/bin/env python3
"""System monitoring script for DGX Spark cluster"""
import subprocess
import json
import time
import socket
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

@dataclass
class GPUInfo:
    index: int
    name: str
    temperature: float
    utilization: float
    memory_used: float
    memory_total: float
    power_draw: float
    power_limit: float

@dataclass
class NodeInfo:
    hostname: str
    ip: str
    gpu: GPUInfo
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    network_rx_mb: float
    network_tx_mb: float
    timestamp: str

def get_gpu_info() -> GPUInfo:
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return GPUInfo(
                index=int(parts[0]), name=parts[1].strip(),
                temperature=float(parts[2]), utilization=float(parts[3]),
                memory_used=float(parts[4]), memory_total=float(parts[5]),
                power_draw=float(parts[6]), power_limit=float(parts[7])
            )
    except Exception as e:
        print(f"GPU info error: {e}")
    return GPUInfo(0, "Unknown", 0, 0, 0, 0, 0, 0)

def get_network_stats(interface="enP2p1s0f1np1"):
    """Get network throughput stats"""
    try:
        with open(f"/sys/class/net/{interface}/statistics/rx_bytes") as f:
            rx = int(f.read().strip())
        with open(f"/sys/class/net/{interface}/statistics/tx_bytes") as f:
            tx = int(f.read().strip())
        return rx / 1024 / 1024, tx / 1024 / 1024
    except:
        return 0.0, 0.0

def collect_node_info() -> NodeInfo:
    """Collect all monitoring info for this node"""
    gpu = get_gpu_info()
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    rx, tx = get_network_stats()
    
    import psutil
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    return NodeInfo(
        hostname=hostname, ip=ip, gpu=gpu,
        cpu_percent=cpu,
        memory_used_gb=mem.used / 1024**3,
        memory_total_gb=mem.total / 1024**3,
        network_rx_mb=rx, network_tx_mb=tx,
        timestamp=datetime.now().isoformat()
    )

def main():
    """Main monitoring loop"""
    print(f"DGX Monitor started on {socket.gethostname()}")
    while True:
        info = collect_node_info()
        print(json.dumps(asdict(info), indent=2))
        time.sleep(5)

if __name__ == "__main__":
    main()
'''

# Define conversation turns
TURNS = [
    {"user": "Read the file /etc/hostname", "tool_result": FAKE_HOSTNAME},
    {"user": "Good. Now read /etc/os-release to check the OS version.", "tool_result": FAKE_OS_RELEASE},
    {"user": "Write a Python script called system_info.py that prints hostname and OS info.", "tool_result": "File written successfully: system_info.py"},
    {"user": "Run the script: python3 system_info.py", "tool_result": "Hostname: dgxnode1\nOS: Ubuntu 22.04.3 LTS"},
    {"user": "Now read the monitoring script at /home/btankut/dgx-monitor/monitor.py", "tool_result": LARGE_CODE},
    {"user": "Analyze that monitoring script. What libraries does it use and what could be improved?", "tool_result": None},  # No tool call expected
    {"user": "Write an improved version of the monitoring script with async support and better error handling.", "tool_result": "File written successfully: monitor_v2.py"},
    {"user": "Run the improved script: python3 monitor_v2.py", "tool_result": "DGX Monitor v2 started\n{\"hostname\": \"dgxnode1\", \"gpu\": {\"temp\": 45, \"util\": 23}}"},
    {"user": "List the current directory to see all files we created.", "tool_result": "system_info.py\nmonitor_v2.py\n__pycache__/"},
    {"user": "Great work! Now create a README.md documenting both scripts.", "tool_result": "File written successfully: README.md"},
]

def estimate_tokens(messages):
    """Rough token count estimate: ~1 token per 4 chars"""
    total_chars = sum(len(json.dumps(m)) for m in messages)
    return total_chars // 4

def measure_turn(messages, expect_tool_call=True):
    """Measure a single turn with streaming"""
    t_start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    tool_calls_acc = {}
    content_acc = ""
    
    try:
        kwargs = dict(
            model=MODEL, messages=messages, max_tokens=1024, stream=True, temperature=0.7,
        )
        if expect_tool_call:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
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
        
        if first_token_time is None:
            return {"ttft_ms": 0, "decode_toks": 0, "tool_ok": False, "tools": {}, "content": ""}
        
        ttft_ms = (first_token_time - t_start) * 1000
        if token_count > 1 and last_token_time > first_token_time:
            decode_toks = (token_count - 1) / (last_token_time - first_token_time)
        else:
            decode_toks = 0
        
        tool_ok = len(tool_calls_acc) > 0 if expect_tool_call else True
        
        return {
            "ttft_ms": round(ttft_ms, 1),
            "decode_toks": round(decode_toks, 2),
            "tool_ok": tool_ok,
            "tools": tool_calls_acc,
            "content": content_acc[:200],
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"ttft_ms": 0, "decode_toks": 0, "tool_ok": False, "tools": {}, "content": str(e)}

def main():
    os.makedirs("results", exist_ok=True)
    
    print("=" * 90)
    print("  Test 2: Agentic Workflow Simulation (10-turn multi-tool)")
    print("=" * 90)
    
    messages = [{"role": "system", "content": "You are a helpful coding assistant. Use the provided tools to accomplish tasks."}]
    results = []
    
    for i, turn in enumerate(TURNS):
        turn_num = i + 1
        expect_tool = turn["tool_result"] is not None
        
        # Add user message
        messages.append({"role": "user", "content": turn["user"]})
        approx_tokens = estimate_tokens(messages)
        
        print(f"\n--- Turn {turn_num}: ~{approx_tokens} tokens ---")
        print(f"  User: {turn['user'][:80]}...")
        
        result = measure_turn(messages, expect_tool_call=expect_tool)
        
        tool_str = ""
        if result["tools"]:
            for idx, tc in result["tools"].items():
                tool_str = f"{tc['name']}({tc['arguments'][:60]})"
                print(f"  Tool: {tool_str}")
        elif result["content"]:
            print(f"  Response: {result['content'][:100]}...")
        
        icon = "✅" if result["tool_ok"] else "❌"
        print(f"  {icon} TTFT={result['ttft_ms']:.0f}ms, Decode={result['decode_toks']:.1f} tok/s")
        
        results.append({
            "turn": turn_num,
            "approx_tokens": approx_tokens,
            "ttft_ms": result["ttft_ms"],
            "decode_toks": result["decode_toks"],
            "tool_ok": result["tool_ok"],
        })
        
        # Add assistant response to conversation
        if result["tools"]:
            # Add tool call message
            tc_list = []
            for idx, tc in result["tools"].items():
                tc_list.append({"id": f"call_{turn_num}_{idx}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}})
            messages.append({"role": "assistant", "tool_calls": tc_list})
            # Add tool result
            for tc in tc_list:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": turn["tool_result"] or ""})
        else:
            messages.append({"role": "assistant", "content": result["content"]})
    
    # Print summary
    print("\n" + "=" * 90)
    print(f"{'Turn':>5} | {'Tokens':>8} | {'TTFT (ms)':>10} | {'Decode tok/s':>13} | {'Tool OK':>8}")
    print("-" * 60)
    for r in results:
        icon = "✅" if r["tool_ok"] else "❌"
        print(f"{r['turn']:>5} | {r['approx_tokens']:>8} | {r['ttft_ms']:>10.1f} | {r['decode_toks']:>13.2f} | {icon:>8}")
    
    # Save
    csv_path = "results/test2_agentic_workflow.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")
    
    with open("results/test2_agentic_workflow.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
