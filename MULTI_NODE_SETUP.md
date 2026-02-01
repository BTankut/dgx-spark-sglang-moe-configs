# Multi-Node Setup Guide (4x DGX Spark, TP=4)

Complete guide for running GLM-4.7-FP8 across 4 DGX Spark nodes with Tensor Parallelism and EAGLE Speculative Decoding.

## Prerequisites

- 4x NVIDIA DGX Spark with GB10 GPUs (128GB each)
- 200Gbps network with RoCE/RDMA support
- SSH access between all nodes
- Docker with NVIDIA Container Toolkit

## Network Architecture

**Critical:** Use a dedicated fabric network for NCCL traffic, separate from your LAN.

```
┌─────────────────────────────────────────────────────────────────┐
│                    200Gbps Fabric Network                        │
│                    (NCCL/RDMA - enP2p1s0f1np1)                  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  │   Node 0     │  │   Node 1     │  │   Node 2     │  │   Node 3     │
│  │ 192.168.101.11│  │192.168.101.12│  │192.168.101.13│  │192.168.101.14│
│  │   (Head)     │  │  (Worker)    │  │  (Worker)    │  │  (Worker)    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
└─────────────────────────────────────────────────────────────────┘
```

| Node | Fabric IP | LAN IP | Role |
|------|-----------|--------|------|
| Node 0 | 192.168.101.11 | 10.0.0.11 | Head (rank 0) |
| Node 1 | 192.168.101.12 | 10.0.0.12 | Worker (rank 1) |
| Node 2 | 192.168.101.13 | 10.0.0.13 | Worker (rank 2) |
| Node 3 | 192.168.101.14 | 10.0.0.14 | Worker (rank 3) |

> **Note:** Adjust IPs to match your network. The fabric interface is typically `enP2p1s0f1np1` on DGX Spark.

## Step 1: Start Containers (All Nodes)

Run on each node with RDMA support:

```bash
docker run -d --name sglang_node \
  --network host \
  --ipc=host \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/infiniband/rdma_cm \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/uverbs1 \
  --device=/dev/infiniband/uverbs2 \
  --device=/dev/infiniband/uverbs3 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest sleep infinity
```

## Step 2: Environment Variables

These must be set on ALL nodes before starting SGLang:

```bash
export NCCL_SOCKET_IFNAME=enP2p1s0f1np1   # Fabric network interface
export GLOO_SOCKET_IFNAME=enP2p1s0f1np1   # Same for Gloo
export NCCL_IB_HCA=mlx5_1                  # InfiniBand HCA device
export VLLM_HOST_IP=192.168.101.XX         # This node's fabric IP
export MASTER_ADDR=192.168.101.11          # Head node's fabric IP
export MASTER_PORT=50000                   # Distributed init port
export NCCL_DEBUG=INFO                     # For debugging (optional)
```

## Step 3: Start Head Node (Rank 0)

On the head node (192.168.101.11):

```bash
docker exec -d sglang_node bash -c '
export NCCL_SOCKET_IFNAME=enP2p1s0f1np1
export GLOO_SOCKET_IFNAME=enP2p1s0f1np1
export NCCL_IB_HCA=mlx5_1
export VLLM_HOST_IP=192.168.101.11
export MASTER_ADDR=192.168.101.11
export MASTER_PORT=50000

python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 4 --nnodes 4 --node-rank 0 \
  --dist-init-addr 192.168.101.11:50000 \
  --dist-timeout 600 \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 8 \
  --speculative-eagle-topk 2 \
  --max-running-requests 64 \
  --context-length 202752 \
  > /tmp/sglang.log 2>&1
'
```

## Step 4: Start Worker Nodes (Rank 1, 2, 3)

Start workers after the head node. Run on each worker:

**Node 1 (Rank 1 - 192.168.101.12):**
```bash
docker exec -d sglang_node bash -c '
export NCCL_SOCKET_IFNAME=enP2p1s0f1np1
export GLOO_SOCKET_IFNAME=enP2p1s0f1np1
export NCCL_IB_HCA=mlx5_1
export VLLM_HOST_IP=192.168.101.12
export MASTER_ADDR=192.168.101.11
export MASTER_PORT=50000

python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp 4 --nnodes 4 --node-rank 1 \
  --dist-init-addr 192.168.101.11:50000 \
  --dist-timeout 600 \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 8 \
  --speculative-eagle-topk 2 \
  --max-running-requests 64 \
  --context-length 202752 \
  > /tmp/sglang.log 2>&1
'
```

**Node 2 (Rank 2 - 192.168.101.13):**
```bash
# Same as above, but with:
# --node-rank 2
# VLLM_HOST_IP=192.168.101.13
```

**Node 3 (Rank 3 - 192.168.101.14):**
```bash
# Same as above, but with:
# --node-rank 3
# VLLM_HOST_IP=192.168.101.14
```

## Step 5: Wait for Model Loading (~10-12 minutes)

Monitor the head node logs:

```bash
docker exec sglang_node tail -f /tmp/sglang.log
```

You'll see:
1. `Loading safetensors checkpoint shards: 100%`
2. `Capture cuda graph end`
3. `Capture draft cuda graph end` (EAGLE)
4. **"The server is fired up and ready to roll!"**

## Step 6: Test the API

```bash
curl http://192.168.101.11:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Launch Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--tp 4` | 4 | Tensor Parallelism (1 GPU per node) |
| `--nnodes 4` | 4 | Total number of nodes |
| `--node-rank N` | 0-3 | This node's rank (0 = head) |
| `--dist-init-addr` | IP:PORT | Head node's fabric IP and port |
| `--dist-timeout` | 600 | Timeout in seconds (important!) |
| `--speculative-algorithm EAGLE` | - | Enable EAGLE speculative decoding |
| `--speculative-num-steps` | 3 | Draft speculation steps |
| `--speculative-num-draft-tokens` | 8 | Tokens per draft step |
| `--context-length` | 202752 | Max context window |

## Environment Variables Explained

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_SOCKET_IFNAME` | `enP2p1s0f1np1` | Network interface for NCCL |
| `GLOO_SOCKET_IFNAME` | `enP2p1s0f1np1` | Network interface for Gloo |
| `NCCL_IB_HCA` | `mlx5_1` | InfiniBand HCA device |
| `VLLM_HOST_IP` | Node's fabric IP | This node's IP for binding |
| `MASTER_ADDR` | Head's fabric IP | Coordination address |
| `MASTER_PORT` | 50000 | Coordination port |

## Troubleshooting

### "Init torch distributed begin" hangs
- Old processes holding port 50000
- **Fix:** `docker exec sglang_node pkill -9 -f sglang` on all nodes

### "Connection refused" error
- Head node not started or wrong IP
- **Fix:** Verify head node is running and fabric IPs are correct

### Slow performance (not using RoCE)
- NCCL falling back to sockets
- **Fix:** Check `NCCL_DEBUG=INFO` logs for "NET/IB" vs "NET/Socket"

### OutOfResources error
- MoE kernel configs not loaded
- **Fix:** Verify configs exist in Triton config directory

### GPU memory error
- Context too long or too many concurrent requests
- **Fix:** Reduce `--context-length` or `--max-running-requests`

## Stopping the Cluster

```bash
# On all nodes
docker exec sglang_node pkill -9 -f sglang
```

## Performance Tips

1. **Use dedicated fabric network** - Separate NCCL traffic from LAN
2. **Enable EAGLE** - Speculative decoding improves throughput
3. **Set dist-timeout** - 600s prevents premature timeouts during loading
4. **Check GPU utilization** - Should be 94-95% during inference

---

*Tested on 4x DGX Spark with 200Gbps RoCE network - February 2026*
