# Multi-Node Setup Guide (4x DGX Spark, TP=4)

Complete guide for running GLM-4.7-FP8 across 4 DGX Spark nodes with Tensor Parallelism.

## Prerequisites

- 4x NVIDIA DGX Spark with GB10 GPUs
- 200Gbps network with RoCE/RDMA support
- SSH access between all nodes
- Docker with NVIDIA Container Toolkit

## Network Configuration

| Node | IP Address | Role |
|------|------------|------|
| Node 0 | 192.168.12.227 | Head (rank 0) |
| Node 1 | 192.168.12.224 | Worker (rank 1) |
| Node 2 | 192.168.12.225 | Worker (rank 2) |
| Node 3 | 192.168.12.226 | Worker (rank 3) |

**Adjust IPs to match your setup.**

## Step 1: Start Containers (All Nodes)

**Critical:** Must include RDMA parameters for RoCE support.

```bash
# Run on each node
docker run -d --name sglang_node \
  --network host \
  --ipc=host \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/infiniband \
  --cap-add=IPC_LOCK \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:spark sleep infinity
```

## Step 2: Install MoE Configs (All Nodes)

```bash
CONFIG_DIR="/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0"

# On each node
docker exec sglang_node mkdir -p "$CONFIG_DIR"
docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json sglang_node:"$CONFIG_DIR/"
docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json sglang_node:"$CONFIG_DIR/"
```

## Step 3: Start Workers (Rank 3, 2, 1)

**Important:** Start workers BEFORE head node!

```bash
# Rank 3 (192.168.12.226)
ssh 192.168.12.226 "docker exec -d sglang_node bash -c '\
  export NCCL_SOCKET_IFNAME=enp1s0f1np1 && \
  export GLOO_SOCKET_IFNAME=enp1s0f1np1 && \
  export NCCL_IB_HCA=mlx5_1 && \
  export NCCL_DEBUG=WARN && \
  python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-FP8 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --tp 4 --nnodes 4 \
    --node-rank 3 \
    --dist-init-addr 192.168.12.227:50000 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.85 \
    --context-length 32000 \
    --disable-cuda-graph \
    > /tmp/sglang.log 2>&1'"

# Rank 2 (192.168.12.225)
ssh 192.168.12.225 "docker exec -d sglang_node bash -c '\
  export NCCL_SOCKET_IFNAME=enp1s0f1np1 && \
  export GLOO_SOCKET_IFNAME=enp1s0f1np1 && \
  export NCCL_IB_HCA=mlx5_1 && \
  export NCCL_DEBUG=WARN && \
  python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-FP8 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --tp 4 --nnodes 4 \
    --node-rank 2 \
    --dist-init-addr 192.168.12.227:50000 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.85 \
    --context-length 32000 \
    --disable-cuda-graph \
    > /tmp/sglang.log 2>&1'"

# Rank 1 (192.168.12.224)
ssh 192.168.12.224 "docker exec -d sglang_node bash -c '\
  export NCCL_SOCKET_IFNAME=enp1s0f1np1 && \
  export GLOO_SOCKET_IFNAME=enp1s0f1np1 && \
  export NCCL_IB_HCA=mlx5_1 && \
  export NCCL_DEBUG=WARN && \
  python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-FP8 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --tp 4 --nnodes 4 \
    --node-rank 1 \
    --dist-init-addr 192.168.12.227:50000 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.85 \
    --context-length 32000 \
    --disable-cuda-graph \
    > /tmp/sglang.log 2>&1'"
```

## Step 4: Start Head Node (Rank 0)

```bash
# On head node (192.168.12.227)
docker exec -d sglang_node bash -c "\
  export NCCL_SOCKET_IFNAME=enp1s0f1np1 && \
  export GLOO_SOCKET_IFNAME=enp1s0f1np1 && \
  export NCCL_IB_HCA=mlx5_1 && \
  export NCCL_DEBUG=WARN && \
  python3 -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-FP8 \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --tp 4 --nnodes 4 \
    --node-rank 0 \
    --dist-init-addr 192.168.12.227:50000 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.85 \
    --context-length 32000 \
    --disable-cuda-graph \
    > /tmp/sglang.log 2>&1"
```

## Step 5: Wait for Model Loading (~10 minutes)

```bash
# Monitor progress
docker exec sglang_node tail -f /tmp/sglang.log

# Wait for this message:
# "The server is fired up and ready to roll!"
```

## Step 6: Test the API

```bash
curl http://<HEAD_IP>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-4.7-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_SOCKET_IFNAME` | `enp1s0f1np1` | Network interface for NCCL |
| `GLOO_SOCKET_IFNAME` | `enp1s0f1np1` | Network interface for Gloo |
| `NCCL_IB_HCA` | `mlx5_1` | InfiniBand HCA device |
| `NCCL_DEBUG` | `WARN` | Debug level |

**Note:** Adjust `enp1s0f1np1` and `mlx5_1` to match your network configuration.

## Troubleshooting

### "Connection refused" error
- Check that workers started before head node
- Verify firewall allows port 50000

### "No RDMA device found"
- Ensure `--device=/dev/infiniband` in docker run
- Check `ls /dev/infiniband/` shows devices

### Slow performance (using Socket instead of RoCE)
- Verify NCCL_IB_HCA is set correctly
- Check `NCCL_DEBUG=INFO` logs for "NET/IB"

### OutOfResources error
- Ensure MoE config files are installed
- Check `--disable-cuda-graph` is set

## Stopping the Server

```bash
# On all nodes
docker exec sglang_node pkill -f sglang.launch_server
```
