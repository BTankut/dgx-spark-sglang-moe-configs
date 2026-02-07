# SGLang for DGX Spark (GB10) - GLM-4.7-FP8

Ready-to-use Docker setup for running **GLM-4.7-FP8** (355B MoE) on NVIDIA DGX Spark with GB10 GPUs.

## Why is this needed?

The GB10 GPU has a shared memory limit of **101,376 bytes**. SGLang's default MoE kernel settings require **147,456 bytes**, causing `OutOfResources` errors. Without tuned configs, EAGLE speculative decoding crashes entirely. This repo provides:

1. **Optimized MoE kernel configs** — Tuned for GB10's memory constraints
2. **GLM-4.7 tool call parser** — Backported `glm47` parser for SGLang v0.5.4
3. **Pre-configured Dockerfile** — Build once, deploy everywhere
4. **Benchmark suite** — Validates bandwidth-throughput formula and agentic performance

## Key Results

### MoE Config Impact (A/B/C Test)

| Scenario | MoE Configs | EAGLE | Result |
|----------|-------------|-------|--------|
| A | ✅ Optimized | ❌ Off | 16.77 tok/s |
| B | ❌ Default | ❌ Off | 15.77 tok/s (-6.3%) |
| C | ❌ Default | ✅ On | 💥 `OutOfResources` crash |
| **Production** | **✅ Optimized** | **✅ On** | **20-27 tok/s** |

### Context Length vs Decode Speed

| Context | KV Cache | Decode tok/s | Degradation |
|---------|----------|-------------|-------------|
| 512 | 0.07 GB | 7.50 | baseline |
| 4K | 0.60 GB | 7.41 | -1.2% |
| 16K | 2.38 GB | 6.99 | -6.8% |
| 32K | 4.76 GB | 6.54 | -12.8% |

Only 13% degradation from 512→32K tokens. See [RESULTS.md](RESULTS.md) for full analysis including agentic workflow tests, EAGLE efficiency, and thinking mode comparison.

## Quick Start

### Option 1: Pull Pre-built Image (Fastest)

```bash
docker pull ghcr.io/btankut/sglang-spark-glm47:latest

docker run -d --name sglang_node \
  --network host --ipc=host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/btankut/sglang-spark-glm47:latest sleep infinity
```

### Option 2: Use Base Image + Manual Setup

```bash
docker run -d --name sglang_node \
  --network host --ipc=host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:spark sleep infinity

# Copy MoE configs
CONFIG_DIR="/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0"
docker exec sglang_node mkdir -p "$CONFIG_DIR"
docker cp configs/triton_3_5_0/. sglang_node:"$CONFIG_DIR/"

# Apply glm47 parser patch (for tool calling)
FUNC_DIR="/sgl-workspace/sglang/python/sglang/srt/function_call"
docker cp patches/glm47_moe_detector.py sglang_node:"$FUNC_DIR/"
docker cp patches/patch_utils.py sglang_node:/tmp/
docker cp patches/patch_parser.py sglang_node:/tmp/
docker exec sglang_node python3 /tmp/patch_utils.py "$FUNC_DIR/utils.py"
docker exec sglang_node python3 /tmp/patch_parser.py "$FUNC_DIR/function_call_parser.py"
```

### Launch (4-node TP=4 + EAGLE)

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 4 --nnodes 4 --node-rank 0 \
  --dist-init-addr 192.168.101.11:50000 \
  --dist-timeout 600 \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code \
  --max-running-requests 64 \
  --context-length 32768 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 8 \
  --speculative-eagle-topk 2
```

## Repository Structure

```
.
├── README.md                  # This file
├── RESULTS.md                 # Full benchmark results & analysis
├── Dockerfile                 # Ready-to-build container
├── build-and-deploy.sh        # Build + deploy to cluster
├── configs/
│   ├── triton_3_5_0/          # MoE configs for Triton 3.5.0
│   └── triton_3_3_0/          # MoE configs for Triton 3.3.0
├── patches/
│   ├── glm47_moe_detector.py  # GLM-4.7 tool call parser (backport)
│   ├── patch_utils.py         # Adds infer_type_from_json_schema
│   └── patch_parser.py        # Registers glm47 parser
├── benchmarks/
│   ├── benchmark_context_vs_speed.py   # Test 1: flash3 formula validation
│   ├── benchmark_agentic_workflow.py   # Test 2: Multi-turn tool calling
│   ├── benchmark_eagle_efficiency.py   # Test 3: EAGLE ON vs OFF
│   ├── benchmark_thinking_mode.py      # Test 4: Thinking mode impact
│   ├── benchmark_ab.py                 # MoE config A/B test
│   ├── test_tool_call.py               # Tool calling validation
│   └── results/                        # CSV & JSON benchmark data
├── MULTI_NODE_SETUP.md        # 4-node cluster guide
├── TUNING.md                  # How to tune for other models
└── FORUM_POST.md              # NVIDIA forum post template
```

## Documentation

- [MULTI_NODE_SETUP.md](MULTI_NODE_SETUP.md) — Complete 4-node cluster setup with RoCE/RDMA
- [TUNING.md](TUNING.md) — How to tune MoE kernels for other models on GB10
- [RESULTS.md](RESULTS.md) — Full benchmark results, flash3 formula validation, agentic workflow analysis
- [FORUM_POST.md](FORUM_POST.md) — NVIDIA Developer Forum post template

## Why `sglang:spark` instead of `sglang:latest`?

| Issue | `:latest` | `:spark` |
|-------|-----------|----------|
| Triton sm_121a | `ptxas fatal error` | Works |
| sgl-kernel | Not compiled for GB10 | Compiled for sm_121 |
| MoE kernels | OutOfResources | Works with configs |

## Tested Configuration

- **Hardware:** 4x NVIDIA DGX Spark (GB10, 128GB each)
- **Network:** 200Gbps RoCE/RDMA (dedicated fabric, MikroTik CRS812 DDQ)
- **Container:** `lmsysorg/sglang:spark` (v0.5.4.post2)
- **Model:** `zai-org/GLM-4.7-FP8` (355B MoE, 32B active)
- **CUDA:** 13.1 + CUTLASS 4.4.0

## References

- [SGLang Project](https://github.com/sgl-project/sglang)
- [DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [GLM-4.7-FP8 Model](https://huggingface.co/zai-org/GLM-4.7-FP8)
- [CUTLASS Issue #2800](https://github.com/NVIDIA/cutlass/issues/2800) — SM121 Python DSL fix

## License

MIT

---

*Tested and working — February 2026*
