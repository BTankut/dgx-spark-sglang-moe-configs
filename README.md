# SGLang for DGX Spark (GB10) - GLM-4.7-FP8

Ready-to-use Docker setup for running **GLM-4.7-FP8** (355B MoE) on NVIDIA DGX Spark with GB10 GPUs.

## Why is this needed?

The GB10 GPU has a shared memory limit of **101,376 bytes**. SGLang's default MoE kernel settings require **147,456 bytes**, causing `OutOfResources` errors. This repo provides:

1. **Optimized MoE kernel configs** - Tuned for GB10's memory constraints
2. **Tool call parser patch** - Fixes GLM-4.7 function calling compatibility
3. **Pre-configured Dockerfile** - Build once, deploy everywhere

## Quick Start

### Option 1: Build from Dockerfile (Recommended)

```bash
# Clone repo
git clone https://github.com/BTankut/dgx-spark-glm47-config.git
cd dgx-spark-glm47-config

# Build image
docker build -t sglang-spark-glm47 .

# Deploy to all nodes (builds + transfers)
./build-and-deploy.sh --deploy
```

### Option 2: Use Base Image + Manual Setup

```bash
# Start container
docker run -d --name sglang_node \
  --network host --ipc=host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:spark sleep infinity

# Copy MoE configs
CONFIG_DIR="/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0"
docker exec sglang_node mkdir -p "$CONFIG_DIR"
docker cp configs/triton_3_5_0/. sglang_node:"$CONFIG_DIR/"
```

## Benchmark Results (4x DGX Spark, TP=4, EAGLE)

| Metric | Value |
|--------|-------|
| Throughput | **20-30 tok/s** |
| Context Window | 202,752 tokens |
| Model Memory | ~82 GB per node |
| GPU Utilization | 94-95% |
| Speculative Accept Rate | 75-88% |

## Repository Structure

```
.
├── Dockerfile                 # Ready-to-build container
├── build-and-deploy.sh       # Build + deploy to cluster
├── configs/
│   ├── triton_3_5_0/         # MoE configs for Triton 3.5.0
│   │   ├── E=160,N=384,...fp8_w8a8.json
│   │   └── E=160,N=384,...fp8_w8a8_down.json
│   └── triton_3_3_0/         # MoE configs for Triton 3.3.0
├── MULTI_NODE_SETUP.md       # 4-node cluster guide
├── TUNING.md                 # How to tune for other models
└── FORUM_POST.md             # NVIDIA forum template
```

## Multi-Node Setup (4x DGX Spark)

See [MULTI_NODE_SETUP.md](MULTI_NODE_SETUP.md) for complete 4-node cluster setup with:
- Tensor Parallelism (TP=4)
- EAGLE Speculative Decoding
- 200Gbps RoCE/RDMA networking
- Tool calling support

**Quick launch after deployment:**
```bash
# Head node (rank 0)
docker exec sglang_node python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 4 --nnodes 4 --node-rank 0 \
  --dist-init-addr 192.168.101.11:50000 \
  --tool-call-parser glm \
  --speculative-algorithm EAGLE \
  --trust-remote-code
```

## What the Dockerfile Includes

1. **Base:** `lmsysorg/sglang:spark` (v0.5.4 with GB10 support)
2. **MoE Configs:** Optimized kernel parameters for GB10
3. **Parser Patch:** Fixed regex for GLM-4.7 tool calls

### Why `sglang:spark` instead of `sglang:latest`?

| Issue | `:latest` | `:spark` |
|-------|-----------|----------|
| Triton sm_121a | `ptxas fatal error` | Works |
| sgl-kernel | Not compiled for GB10 | Compiled for sm_121 |
| MoE kernels | OutOfResources | Works with configs |

## Tested Configuration

- **Hardware:** 4x NVIDIA DGX Spark (GB10, 128GB each)
- **Network:** 200Gbps RoCE/RDMA (dedicated fabric)
- **Container:** `lmsysorg/sglang:spark` (v0.5.4.post2)
- **Model:** `zai-org/GLM-4.7-FP8` (355B MoE, 32B active)

## How Configs Were Generated

MoE kernel tuning with Ray across 4 nodes (~9 hours):

```bash
python3 tuning_fused_moe_triton.py \
  --model zai-org/GLM-4.7-FP8 \
  --tp-size 4 --dtype fp8_w8a8 --tune
```

See [TUNING.md](TUNING.md) for tuning other models.

## References

- [SGLang Project](https://github.com/sgl-project/sglang)
- [DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [GLM-4.7-FP8 Model](https://huggingface.co/zai-org/GLM-4.7-FP8)

## License

MIT

---

*Tested and working - February 2026*
