#!/bin/bash

# 1. System update and dependency installation
sudo apt-get update && sudo apt-get install -y ffmpeg libavcodec-extra

# 2. Install Python dependencies
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

# 3. Initialize vLLM inference engines
# Partitioning the 192GB VRAM of the MI300X (~65% for 72B Vision, remainder for reasoning/audio)
# Note: Quantized versions (AWQ/GPTQ) for Qwen3-32B/14B are recommended to ensure VRAM headroom.

# Start Agent 1 (Vision)
docker exec -d rocm bash -c 'vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.65 \
    --max-model-len 8192 \
    --trust-remote-code > /tmp/vllm_vl.log 2>&1'

# Start Agent 2 (Reasoning)
docker exec -d rocm bash -c 'vllm serve Qwen/Qwen3-32B-Thinking \
    --host 0.0.0.0 --port 8001 \
    --gpu-memory-utilization 0.20 \
    --max-model-len 16384 \
    --trust-remote-code > /tmp/vllm_reasoning.log 2>&1'

# 4. Create project directory
mkdir -p crop-whisper-pipeline
cd crop-whisper-pipeline

# 5. Generate LangGraph orchestrator skeleton
cat << 'EOF' > main.py
import os
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from supabase import create_client
from openai import OpenAI

# Supabase Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# vLLM Client Configuration
client_vl = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
client_reasoning = OpenAI(base_url="http://localhost:8001/v1", api_key="none")

class AgentState(TypedDict):
    image_path: str
    transcript: str
    region_context: Dict
    visual_description: Dict
    diagnosis: Dict
    verified_assessment: Dict
    action_plan: str
    language: str

# --- Node Definitions ---

def visual_description_node(state: AgentState):
    # Agent 1: Qwen2.5-VL-72B logic
    return {"visual_description": {"status": "completed", "data": "..."}}

def analyzer_node(state: AgentState):
    # Agent 2: Qwen3-32B Thinking logic
    # Retrieve context from Supabase pgvector here
    return {"diagnosis": {"status": "completed", "data": "..."}}

def verification_node(state: AgentState):
    # Agent 3: Qwen3-14B (Thinking) logic
    return {"verified_assessment": {"status": "verified"}}

def action_plan_node(state: AgentState):
    # Agent 4: Qwen3-14B logic
    return {"action_plan": "Step 1: Apply Nitrogen..."}

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("agent_1_visual", visual_description_node)
workflow.add_node("agent_2_analyzer", analyzer_node)
workflow.add_node("agent_3_verify", verification_node)
workflow.add_node("agent_4_action", action_plan_node)

workflow.set_entry_point("agent_1_visual")
workflow.add_edge("agent_1_visual", "agent_2_analyzer")
workflow.add_edge("agent_2_analyzer", "agent_3_verify")
workflow.add_edge("agent_3_verify", "agent_4_action")
workflow.add_edge("agent_4_action", END)

app = workflow.compile()
EOF

# 6. Generate .env template
cat << 'EOF' > .env
SUPABASE_URL=your_project_url
SUPABASE_KEY=your_service_role_key
VLLM_VL_ENDPOINT=http://localhost:8000/v1
VLLM_REASONING_ENDPOINT=http://localhost:8001/v1
EOF