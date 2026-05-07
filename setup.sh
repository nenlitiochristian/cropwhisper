#!/bin/bash

# 1. System update and OS dependency installation
sudo apt-get update && sudo apt-get install -y ffmpeg libavcodec-extra python3-venv python3-pip

# 2. Create and activate Python virtual environment
python3 -m venv crop_env
source crop_env/bin/activate

# 3. Install Python dependencies inside the virtual environment
pip install --upgrade pip
pip install \
    langgraph==0.2.35 \
    supabase==2.11.0 \
    openai \
    gradio==5.29.0 \
    transformers>=4.56.0 \
    accelerate==1.3.0 \
    soundfile==0.13.0 \
    librosa==0.11.0 \
    sentencepiece==0.2.0 \
    vllm==0.17.1

# 4. Agents Setup

# Start Agent 1: Visual Description (Qwen2.5-VL-72B)
docker exec -d rocm bash -c 'vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code > /tmp/vllm_vl.log 2>&1'

# Start Agent 2: Analyzer (Qwen3-32B-Thinking)
docker exec -d rocm bash -c 'vllm serve Qwen/Qwen3-32B-Thinking \
    --host 0.0.0.0 --port 8001 \
    --gpu-memory-utilization 0.18 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --trust-remote-code > /tmp/vllm_reasoning.log 2>&1'

# Start Agent 3: Verification (Qwen3-14B-Thinking)
docker exec -d rocm bash -c 'vllm serve Qwen/Qwen3-14B-Thinking \
    --host 0.0.0.0 --port 8002 \
    --gpu-memory-utilization 0.08 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code > /tmp/vllm_verify.log 2>&1'

# Start Agent 4: Action (Qwen3-14B)
docker exec -d rocm bash -c 'vllm serve Qwen/Qwen3-14B \
    --host 0.0.0.0 --port 8003 \
    --gpu-memory-utilization 0.08 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code > /tmp/vllm_action.log 2>&1'