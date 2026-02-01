# MoE Kernel Tuning Guide for DGX Spark (GB10)

This guide explains how to generate optimized MoE kernel configurations for any model on the GB10 GPU.

## Why Tuning is Needed

The GB10 GPU has a shared memory limit of **101,376 bytes** (same as RTX 4090). Many MoE models use kernel configurations that exceed this limit, causing:

```
OutOfResources: out of resource: shared memory
Required: 147456, Hardware limit: 101376
```

Tuning finds optimal kernel parameters that work within GB10's constraints.

## Prerequisites

- DGX Spark with GB10 GPU(s)
- SGLang installed (or use Docker container)
- Ray (for multi-node distributed tuning)
- ~9 hours for full tuning (4 nodes)

## Step 1: Set Up Environment

### Using Docker (Recommended)

```bash
docker run -it --name sglang_tuning \
  --network host --ipc=host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest bash
```

### Find the Tuning Script

```bash
# Inside container
SGLANG_PATH=$(python3 -c "import sglang; print(sglang.__path__[0])")
ls $SGLANG_PATH/../benchmark/kernels/fused_moe_triton/

# You should see:
# tuning_fused_moe_triton.py
```

Or clone the SGLang repo:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/benchmark/kernels/fused_moe_triton/
```

## Step 2: Single-Node Tuning (Quick Test)

For a quick test on a single GPU:

```bash
cd /sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/

python3 tuning_fused_moe_triton.py \
  --model zai-org/GLM-4.7-FP8 \
  --tp-size 1 \
  --dtype fp8_w8a8 \
  --tune
```

This will take several hours and generate configs for single-GPU use.

## Step 3: Multi-Node Tuning (Recommended)

For production use with TP=4, tune across all nodes using Ray.

### Start Ray Cluster

**Head Node:**
```bash
ray start --head --node-ip-address=192.168.101.11 --port=6379
```

**Worker Nodes:**
```bash
ray start --address=192.168.101.11:6379 --node-ip-address=192.168.101.XX
```

### Run Distributed Tuning

```bash
python3 tuning_fused_moe_triton.py \
  --model zai-org/GLM-4.7-FP8 \
  --tp-size 4 \
  --dtype fp8_w8a8 \
  --tune
```

### Tuning Parameters

| Parameter | Description |
|-----------|-------------|
| `--model` | HuggingFace model path |
| `--tp-size` | Tensor parallelism size (match your deployment) |
| `--dtype` | Data type: `fp8_w8a8`, `fp16`, `bf16` |
| `--tune` | Enable tuning mode |

## Step 4: Tuning Output

The script generates JSON config files:

```
E=<experts>,N=<intermediate_size>,device_name=NVIDIA_GB10,dtype=<dtype>.json
E=<experts>,N=<intermediate_size>,device_name=NVIDIA_GB10,dtype=<dtype>_down.json
```

For GLM-4.7-FP8:
- `E=160` (160 experts)
- `N=384` (intermediate size / TP)
- Two files: one for up-projection, one for down-projection

## Step 5: Install Generated Configs

Copy the generated configs to SGLang's config directory:

```bash
# Find the config directory
CONFIG_DIR=$(python3 -c "import sglang; print(sglang.__path__[0])")/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0

# Create if needed
mkdir -p $CONFIG_DIR

# Copy configs
cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8*.json $CONFIG_DIR/
```

### For Docker Deployment

```bash
docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json \
  sglang_node:/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0/

docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json \
  sglang_node:/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0/
```

## Step 6: Verify Configs Are Used

When you start SGLang, check the logs:

```bash
grep "Using MoE kernel config" /tmp/sglang.log
```

Expected output:
```
[TP0] Using MoE kernel config from .../E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json
[TP0] Using MoE kernel config from .../E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json
```

## Understanding the Config Format

Each JSON file contains batch size mappings:

```json
{
    "1": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2
    },
    "2": { ... },
    "4": { ... },
    ...
}
```

| Parameter | Description |
|-----------|-------------|
| `BLOCK_SIZE_M/N/K` | Tile dimensions for matrix multiply |
| `GROUP_SIZE_M` | Thread group size |
| `num_warps` | Number of GPU warps |
| `num_stages` | Pipeline stages (affects shared memory) |

The key constraint: `BLOCK_SIZE * num_stages` must fit in 101,376 bytes.

## Tuning for Other Models

To tune for a different MoE model:

1. Identify the model's MoE parameters:
   - Number of experts (E)
   - Intermediate size (N)
   - Data type

2. Run tuning:
   ```bash
   python3 tuning_fused_moe_triton.py \
     --model <your-model-path> \
     --tp-size <your-tp-size> \
     --dtype <fp8_w8a8|fp16|bf16> \
     --tune
   ```

3. Copy generated configs to SGLang

### Models That May Need Tuning on GB10

- Mixtral 8x7B / 8x22B
- DeepSeek-MoE
- Qwen-MoE
- Any large MoE model with >8 experts

## Troubleshooting

### "CUDA out of memory" during tuning
- Reduce batch sizes being tested
- Use fewer parallel tuning jobs

### Tuning takes too long
- Use more nodes with Ray
- Reduce the search space

### Generated configs still cause OutOfResources
- Check `num_stages` values (should be 2-3 for GB10)
- Verify configs are in correct directory

## Contributing

If you generate configs for other models on GB10, please share them with the community!

---

*Based on SGLang's MoE tuning framework*
