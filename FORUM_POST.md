# Forum Post: NVIDIA Developer DGX Spark

---

**Title:** Running GLM-4.7-FP8 (355B MoE) on DGX Spark with SGLang - Optimized MoE Kernel Configs

---

Hi everyone! 👋

I wanted to share something that might help fellow DGX Spark owners who want to run large MoE (Mixture of Experts) models locally.

## The Challenge

I recently tried running **GLM-4.7-FP8** (355B parameters, 32B active) on a 4-node DGX Spark cluster using SGLang. The model loaded fine, but inference would crash with this error:

```
OutOfResources: out of resource: shared memory
Required: 147456, Hardware limit: 101376
```

After some digging, I found that the GB10's shared memory limit (101,376 bytes) is the same as the RTX 4090. SGLang's default MoE kernel settings exceed this limit.

## The Solution

I ran SGLang's MoE kernel tuning script to generate optimized configurations specifically for the GB10. The tuning took about 9 hours across 4 nodes using Ray, but the result is a set of config files that work perfectly.

## Results

With the optimized configs, the model runs smoothly:

| Test | Speed |
|------|-------|
| Single request | ~13 tok/s |
| 4 concurrent | 35 tok/s |
| 8 concurrent | 56 tok/s |

## Sharing the Configs

I've uploaded the config files and setup instructions to GitHub:

**🔗 [GitHub Repository Link]**

The repo includes:
- Pre-tuned MoE kernel configs for GB10
- Single-node setup instructions
- Multi-node (TP=4) setup guide
- Benchmark results

## Quick Start

1. Download the config files
2. Copy them to SGLang's config directory:
   ```
   .../sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0/
   ```
3. Launch SGLang with `--disable-cuda-graph`

Full instructions are in the README.

## Hardware Setup

My cluster:
- 4x DGX Spark (GB10, 128GB each)
- 200Gbps RoCE network
- Container: `lmsysorg/sglang:spark`

## Notes

- These configs are specifically tuned for `zai-org/GLM-4.7-FP8`
- Other MoE models (Mixtral, DeepSeek, etc.) may need their own tuning
- The tuning process is documented if you want to generate configs for other models

## Contributing

If you generate configs for other models on GB10, please share them! It would be great to build a collection of optimized configs for the DGX Spark community.

Hope this helps someone! Happy to answer any questions.

---

**Tags:** DGX Spark, GB10, SGLang, GLM-4, MoE, Inference, Optimization
