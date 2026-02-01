# SGLang MoE Kernel Configs for DGX Spark (GB10)

Optimized MoE (Mixture of Experts) kernel configurations for running **GLM-4.7-FP8** on NVIDIA DGX Spark with GB10 GPUs.

## Why is this needed?

The GB10 GPU has a shared memory limit of **101,376 bytes** (same as RTX 4090). SGLang's default MoE kernel settings (128×128 tile, 4 stages) require **147,456 bytes**, causing `OutOfResources` errors.

These tuned configs provide optimized parameters that work within GB10's constraints.

## Benchmark Results (4x DGX Spark, TP=4, EAGLE Speculative Decoding)

| Metric | Value |
|--------|-------|
| Throughput | **20-27 tok/s** |
| Context Window | 202,752 tokens |
| Model Memory | ~82 GB per node |
| GPU Utilization | 94-95% (active) |

## Quick Setup

### 1. Use the Spark Container (REQUIRED for GB10)

**IMPORTANT:** You MUST use `lmsysorg/sglang:spark` container for DGX Spark (GB10). The standard `:latest` container does NOT work because `sgl-kernel` is not compiled for sm_121 architecture.

```bash
docker run -d --name sglang_node \
  --network host --ipc=host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:spark sleep infinity
```

### 2. Install MoE Kernel Configs (REQUIRED)

The spark container does NOT include these configs by default. You must copy them manually:

```bash
# Create config directory (spark container uses Triton 3.5.0)
CONFIG_DIR="/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0"
docker exec sglang_node mkdir -p "$CONFIG_DIR"

# Copy configs
docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json sglang_node:"$CONFIG_DIR/"
docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json sglang_node:"$CONFIG_DIR/"

# Verify
docker exec sglang_node ls -la "$CONFIG_DIR"
```

Without these configs, you will get `OutOfResources: shared memory` errors.

### 3. Launch SGLang Server

**Single Node:**
```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --tp 1
```

**Multi-Node (4x DGX Spark, TP=4) with EAGLE Speculative Decoding:**

See [MULTI_NODE_SETUP.md](MULTI_NODE_SETUP.md) for detailed instructions.

## Files

| File | Description |
|------|-------------|
| `E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json` | MoE up-projection config |
| `E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json` | MoE down-projection config |
| `MULTI_NODE_SETUP.md` | Multi-node TP=4 setup guide |
| `TUNING.md` | **How to tune MoE kernels for other models** |
| `FORUM_POST.md` | NVIDIA Developer Forum post template |

## Tested Configuration

- **Hardware:** 4x NVIDIA DGX Spark (GB10, 128GB each)
- **Network:** 200Gbps RoCE/RDMA (dedicated fabric network)
- **Container:** `lmsysorg/sglang:spark` (v0.5.4.post2, sgl-kernel 0.3.16.post5)
- **Model:** `zai-org/GLM-4.7-FP8` (355B MoE, 32B active params)
- **Features:** EAGLE Speculative Decoding enabled
- **Context Length:** 202,752 tokens

### Why `sglang:spark` instead of `sglang:latest`?

The standard `:latest` container fails on GB10 with:
- `ptxas fatal: Value 'sm_121a' is not defined` (Triton issue)
- `no kernel image is available for execution` (sgl-kernel not compiled for sm_121)

The `:spark` container includes GB10-specific builds of `sgl-kernel` that support sm_121 architecture.

## Network Architecture

For optimal multi-node performance, use a **dedicated fabric network** for NCCL traffic:

```
┌─────────────────────────────────────────────────────────┐
│              200Gbps Fabric Network                      │
│              (NCCL/RDMA Traffic)                         │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Node 0   │  │ Node 1   │  │ Node 2   │  │ Node 3   │ │
│  │ .101.11  │  │ .101.12  │  │ .101.13  │  │ .101.14  │ │
│  │ (Head)   │  │ (Worker) │  │ (Worker) │  │ (Worker) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────┘
```

## How these configs were generated

Using SGLang's tuning script with Ray distributed across 4 nodes (~9 hours):

```bash
python3 tuning_fused_moe_triton.py \
  --model zai-org/GLM-4.7-FP8 \
  --tp-size 4 \
  --dtype fp8_w8a8 \
  --tune
```

**Want to tune for a different model?** See [TUNING.md](TUNING.md) for the complete guide.

## Contributing

If you generate configs for other MoE models on GB10, please share them!

## References

- [SGLang Project](https://github.com/sgl-project/sglang)
- [DGX Spark SGLang Playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/sglang)
- [GLM-4.7-FP8 Model](https://huggingface.co/zai-org/GLM-4.7-FP8)
- [GLM-4.7-Flash vLLM Guide](https://forums.developer.nvidia.com/t/glm-4-7-flash-on-pgx-dgx-vllm-guide/358874) - Alternative setup using vLLM
- [GLM-4.7 Model Card](https://huggingface.co/zai-org/GLM-4.7) - Official deployment examples

## License

MIT

---

*Tested and working as of February 2026*
