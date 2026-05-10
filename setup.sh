#!/bin/bash

# 1. System update and OS dependency installation
sudo apt-get update && sudo apt-get install -y ffmpeg libavcodec-extra python3-venv python3-pip

# 2. Create and activate Python virtual environment
python3 -m venv crop_env
source crop_env/bin/activate

# For Windows activate the virtual environment using the following command
# python -m venv crop_env
# .\crop_env\Scripts\Activate.ps1

# 3. Install Python dependencies inside the virtual environment
pip install --upgrade pip
pip install -r requirements.txt

# For windows use the following command to install dependencies
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

# 4. Agents Setup

# Separate Setup for Models used for Agents
# In GPU Droplet 1
docker exec -d rocm bash -c '
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve Qwen/Qwen3-VL-30B-A3B-Thinking \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --mm-processor-kwargs "{\"max_dynamic_patch\": 16}" \
  --async-scheduling \
  --trust-remote-code > /tmp/vllm_vl.log 2>&1'

# In GPU Droplet 2
docker exec -d rocm bash -c '
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve Qwen/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --async-scheduling \
  --trust-remote-code > /tmp/vllm_reasoning.log 2>&1'

  # Embedding Model
  