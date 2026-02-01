# SGLang for DGX Spark (GB10) with GLM-4.7-FP8 Support
#
# This Dockerfile creates a ready-to-run SGLang container with:
# - MoE kernel configs optimized for GB10's 101KB shared memory limit
# - Tool call parser patch for GLM-4.7 compatibility
#
# Build:
#   docker build -t sglang-spark-glm47 .
#
# Run:
#   docker run -d --name sglang_node \
#     --network host --ipc=host --gpus all \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     sglang-spark-glm47 sleep infinity

FROM lmsysorg/sglang:spark

LABEL maintainer="BTankut"
LABEL description="SGLang for DGX Spark (GB10) with GLM-4.7-FP8 optimizations"
LABEL version="1.0"

# Create config directories
RUN mkdir -p /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0 \
    && mkdir -p /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_0

# Copy MoE kernel configs for GB10
# These configs use smaller tile sizes (64x64) to fit within GB10's 101KB shared memory limit
COPY configs/triton_3_5_0/*.json /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0/
COPY configs/triton_3_3_0/*.json /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_0/

# Apply tool call parser patch
# SGLang v0.5.4's GLM parser expects newlines between XML tags, but GLM-4.7 sometimes
# outputs without newlines. This patch makes the regex more flexible.
RUN sed -i '59s/.*/            r"<tool_call>([^<]+)(?:\\\\\\\\n|\\\\n|\\\\s*)?(.*?)<\/tool_call>", re.DOTALL/' \
    /sgl-workspace/sglang/python/sglang/srt/function_call/glm4_moe_detector.py

# Verify configs are in place
RUN ls -la /sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_0/ \
    && echo "MoE configs installed successfully"

# Set environment variables for optimal performance
ENV NCCL_DEBUG=WARN
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Default command
CMD ["sleep", "infinity"]
