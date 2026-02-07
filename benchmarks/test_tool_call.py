#!/usr/bin/env python3
"""Test GLM-4.7 tool calling"""
import json
import sys
import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

tests = [
    {
        "name": "Test 1: Simple tool call with params",
        "messages": [{"role": "user", "content": "Read the file /etc/hostname"}],
        "stream": False,
    },
    {
        "name": "Test 2: No-param tool call",
        "messages": [{"role": "user", "content": "List the current directory"}],
        "stream": False,
    },
    {
        "name": "Test 3: Multi-param tool call",
        "messages": [{"role": "user", "content": "What's the weather in Istanbul in celsius?"}],
        "stream": False,
    },
    {
        "name": "Test 4: Streaming tool call",
        "messages": [{"role": "user", "content": "Read the file /etc/hostname"}],
        "stream": True,
    },
]

results = []
for test in tests:
    print(f"\n{'='*60}")
    print(f"  {test['name']}")
    print(f"{'='*60}")
    try:
        if test["stream"]:
            stream = client.chat.completions.create(
                model="zai-org/GLM-4.7-FP8",
                messages=test["messages"],
                tools=tools,
                tool_choice="auto",
                stream=True,
                max_tokens=512,
            )
            tool_calls_acc = {}
            content_acc = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    content_acc += delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"name": "", "arguments": ""}
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments
            if tool_calls_acc:
                print(f"  ✅ Streaming tool calls:")
                for idx, tc in tool_calls_acc.items():
                    print(f"     [{idx}] {tc['name']}({tc['arguments']})")
                results.append(("PASS", test["name"]))
            elif content_acc:
                print(f"  ⚠️  Got content instead: {content_acc[:200]}")
                results.append(("WARN", test["name"]))
            else:
                print(f"  ❌ No tool calls or content")
                results.append(("FAIL", test["name"]))
        else:
            response = client.chat.completions.create(
                model="zai-org/GLM-4.7-FP8",
                messages=test["messages"],
                tools=tools,
                tool_choice="auto",
                max_tokens=512,
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                print(f"  ✅ Tool calls:")
                for tc in msg.tool_calls:
                    print(f"     {tc.function.name}({tc.function.arguments})")
                results.append(("PASS", test["name"]))
            elif msg.content:
                print(f"  ⚠️  Got content instead: {msg.content[:200]}")
                results.append(("WARN", test["name"]))
            else:
                print(f"  ❌ No tool calls or content")
                results.append(("FAIL", test["name"]))
    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append(("FAIL", test["name"]))

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
for status, name in results:
    icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
    print(f"  {icon} {name}")
