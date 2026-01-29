# SGLang MoE Kernel Configs for DGX Spark (GB10)

Optimized MoE (Mixture of Experts) kernel configurations for running **GLM-4.7-FP8** on NVIDIA DGX Spark with GB10 GPUs.

## Why is this needed?

The GB10 GPU has a shared memory limit of **101,376 bytes** (same as RTX 4090). SGLang's default MoE kernel settings (128×128 tile, 4 stages) require **147,456 bytes**, causing `OutOfResources` errors.

These tuned configs provide optimized parameters that work within GB10's constraints.

## Benchmark Results (4x DGX Spark, TP=4)

| Test | Tokens | Time | Speed |
|------|--------|------|-------|
| Single - Short (50 tok) | 50 | 4.0s | **12.5 tok/s** |
| Single - Long (500 tok) | 500 | 38.3s | **13.1 tok/s** |
| Single - Heavy (1000 tok) | 1000 | 75.8s | **13.2 tok/s** |
| 2 Concurrent | 300 | 15.0s | **20.0 tok/s** |
| 4 Concurrent | 592 | 16.9s | **35.0 tok/s** |
| 8 Concurrent | 1200 | 21.3s | **56.4 tok/s** |

## Quick Setup

### 1. Install config files

```bash
# Find your SGLang installation path
SGLANG_PATH=$(python3 -c "import sglang; print(sglang.__path__[0])")
CONFIG_DIR="$SGLANG_PATH/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0"

# Create directory and copy configs
mkdir -p "$CONFIG_DIR"
cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json "$CONFIG_DIR/"
cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json "$CONFIG_DIR/"
```

### 2. For Docker (lmsysorg/sglang:spark)

```bash
CONFIG_DIR="/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0"

docker exec sglang_node mkdir -p "$CONFIG_DIR"
docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json sglang_node:"$CONFIG_DIR/"
docker cp E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json sglang_node:"$CONFIG_DIR/"
```

### 3. Launch SGLang Server

**Single Node (TP=1):**
```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  --tp 1 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph
```

**Multi-Node (4x DGX Spark, TP=4):**

See [MULTI_NODE_SETUP.md](MULTI_NODE_SETUP.md) for detailed instructions.

## Verification

Check that configs are being used:
```bash
grep "Using MoE kernel config" /tmp/sglang.log
```

Expected output:
```
Using MoE kernel config from .../E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json
Using MoE kernel config from .../E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json
```

## Files

| File | Description |
|------|-------------|
| `E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json` | MoE up-projection config |
| `E=160,N=384,device_name=NVIDIA_GB10,dtype=fp8_w8a8_down.json` | MoE down-projection config |
| `MULTI_NODE_SETUP.md` | Multi-node TP=4 setup guide |

## Tested Configuration

- **Hardware:** 4x NVIDIA DGX Spark (GB10, 128GB each)
- **Network:** 200Gbps RoCE/RDMA
- **Container:** `lmsysorg/sglang:spark`
- **Model:** `zai-org/GLM-4.7-FP8` (355B MoE, 32B active params)
- **Triton:** 3.5.0
- **PyTorch:** 2.9.0+cu130

## How these configs were generated

Using SGLang's tuning script with Ray distributed across 4 nodes (~9 hours):

```bash
python3 tuning_fused_moe_triton.py \
  --model zai-org/GLM-4.7-FP8 \
  --tp-size 4 \
  --dtype fp8_w8a8 \
  --tune
```

## Contributing

If you generate configs for other models on GB10, please share them!

## References

- [SGLang MoE Tuning Guide](https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton)
- [DGX Spark SGLang Playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/sglang)
- [Triton Shared Memory Issue](https://github.com/triton-lang/triton/issues/8182)

## License

MIT

---

*Tested and working as of January 2026*
