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
pip install \
    langgraph \
    supabase \
    openai \
    gradio \
    transformers \
    accelerate \
    soundfile \
    librosa \
    sentencepiece \
    vllm==0.17.1

# For windows use the following command to install dependencies
# python -m pip install --upgrade pip
# python -m pip install `
#     langgraph `
#     supabase `
#     openai `
#     gradio `
#     transformers `
#     accelerate `
#     soundfile `
#     librosa `
#     sentencepiece `
#     vllm==0.17.1

# 4. Agents Setup

docker exec -d rocm bash -c 'vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.15 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --trust-remote-code > /tmp/vllm_vl.log 2>&1' 

docker exec -d rocm bash -c 'vllm serve Qwen/Qwen3-32B \
  --host 0.0.0.0 --port 8001 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --trust-remote-code > /tmp/vllm_reasoning.log 2>&1'

docker exec -d rocm bash -c 'vllm serve Qwen/Qwen3-14B \
  --host 0.0.0.0 --port 8002 \
  --gpu-memory-utilization 0.20 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --trust-remote-code > /tmp/vllm_verify.log 2>&1'
  
docker exec -d rocm bash -c 'vllm serve Qwen/Qwen3-14B \
  --host 0.0.0.0 --port 8003 \
  --gpu-memory-utilization 0.20 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --trust-remote-code > /tmp/vllm_action.log 2>&1'