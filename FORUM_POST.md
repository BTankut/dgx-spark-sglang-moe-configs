# Forum Post: NVIDIA Developer DGX Spark

---

**Title:** Running GLM-4.7-FP8 (355B MoE) on 4x DGX Spark with SGLang + EAGLE Speculative Decoding

---

Hi everyone!

I wanted to share my experience running **GLM-4.7-FP8** (355B parameters, 32B active) on a 4-node DGX Spark cluster. After some trial and error, I got it working smoothly with SGLang and EAGLE speculative decoding.

## The Challenge

When I first tried to run GLM-4.7-FP8 on DGX Spark with SGLang, I hit this error:

```
OutOfResources: out of resource: shared memory
Required: 147456, Hardware limit: 101376
```

The GB10's shared memory limit (101,376 bytes) is the same as the RTX 4090. SGLang's default MoE kernel settings exceed this limit.

## The Solution

I ran SGLang's MoE kernel tuning script to generate optimized configurations specifically for the GB10. The tuning took about 9 hours across 4 nodes, but the resulting configs work perfectly.

**Key insight:** You must use `lmsysorg/sglang:spark` container for GB10 - the standard `:latest` does NOT work (sgl-kernel not compiled for sm_121).

I've also published a **pre-built Docker image** with all configs and patches applied:
```bash
docker pull ghcr.io/btankut/sglang-spark-glm47:latest
```

## Results

With optimized configs + EAGLE speculative decoding:

| Metric | Value |
|--------|-------|
| **Throughput** | 20-27 tok/s |
| **Context Window** | 202,752 tokens |
| **GPU Memory** | ~82 GB per node |
| **GPU Utilization** | 94-95% |

## Hardware Setup

- 4x DGX Spark (GB10, 128GB each)
- 200Gbps RoCE network (dedicated fabric)
- Container: `lmsysorg/sglang:spark` or `ghcr.io/btankut/sglang-spark-glm47:latest`

## Network Architecture (Important!)

For multi-node inference, I recommend using a **dedicated fabric network** for NCCL traffic:

```
┌─────────────────────────────────────────────────────────┐
│              200Gbps Fabric Network                      │
│              (NCCL/RDMA Traffic Only)                    │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Node 0   │  │ Node 1   │  │ Node 2   │  │ Node 3   │ │
│  │ .101.11  │  │ .101.12  │  │ .101.13  │  │ .101.14  │ │
│  │ (Head)   │  │ (Worker) │  │ (Worker) │  │ (Worker) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────┘
```

This separates high-bandwidth GPU-to-GPU communication from regular LAN traffic.

## Quick Start

```bash
# Option A: Use pre-built image (recommended)
docker pull ghcr.io/btankut/sglang-spark-glm47:latest

# Option B: Use base spark image + manual config setup
# See GitHub repo for config installation steps

# Start container on each node
docker run -d --name sglang_node \
  --network host --ipc=host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/uverbs1 \
  --device=/dev/infiniband/uverbs2 \
  --device=/dev/infiniband/uverbs3 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/btankut/sglang-spark-glm47:latest sleep infinity

# Launch on head node (rank 0)
docker exec -d sglang_node bash -c '
export NCCL_SOCKET_IFNAME=enP2p1s0f1np1
export GLOO_SOCKET_IFNAME=enP2p1s0f1np1
export NCCL_IB_HCA=mlx5_1
export VLLM_HOST_IP=192.168.101.11

python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 4 --nnodes 4 --node-rank 0 \
  --dist-init-addr 192.168.101.11:50000 \
  --dist-timeout 600 \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code \
  --tool-call-parser glm \
  --reasoning-parser glm45 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 8 \
  --speculative-eagle-topk 2 \
  --context-length 202752 \
  > /tmp/sglang.log 2>&1
'

# Similar for workers (change --node-rank and VLLM_HOST_IP)
```

## Config Files & Full Instructions

I've uploaded everything to GitHub:

**🔗 https://github.com/BTankut/dgx-spark-sglang-moe-configs**

The repo includes:
- **Dockerfile** - Build your own optimized container
- **Pre-built image** - `ghcr.io/btankut/sglang-spark-glm47:latest`
- Pre-tuned MoE kernel configs for GB10
- Tool call parser patch for GLM-4.7
- Complete multi-node setup guide
- Step-by-step tuning guide (for other models)

## Tips That Helped Me

1. **Use dedicated fabric network** - Separating NCCL traffic from LAN improved stability
2. **Enable EAGLE speculative decoding** - Noticeable throughput improvement
3. **Set `--dist-timeout 600`** - Prevents timeouts during model loading
4. **Clean old processes** - `pkill -9 -f sglang` before restarting

## Common Issues

| Problem | Solution |
|---------|----------|
| "Init torch distributed begin" hangs | Kill old sglang processes on all nodes |
| OutOfResources error | Ensure MoE configs are in Triton directory |
| Slow performance | Verify NCCL is using RoCE, not sockets |

## Contributing

If you generate configs for other MoE models on GB10, please share them! It would be great to build a collection for the DGX Spark community.

Hope this helps someone! Happy to answer questions.

---

**Tags:** DGX Spark, GB10, SGLang, GLM-4, MoE, EAGLE, Multi-Node, Inference
