
import json
import os
from typing import TypedDict, Dict
from utils.image import encode_image_to_base64

from openai import OpenAI

class AgentState(TypedDict):
    image_path: str
    transcript: str
    region_context: Dict
    visual_description: Dict
    diagnosis: Dict
    verified_assessment: Dict
    action_plan: Dict
    language: str

client_vl = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
client_reasoning = OpenAI(base_url="http://localhost:8001/v1", api_key="none")
client_verify = OpenAI(base_url="http://localhost:8002/v1", api_key="none")
client_action = OpenAI(base_url="http://localhost:8003/v1", api_key="none")

def visual_description_node(state: AgentState) -> dict:
    """Step 1: Vision-Language model describes only what is visible — no context, no interpretation."""
    image_path = state["image_path"]
    system_prompt = get_prompt("visual")

    user_content = []
    if image_path and os.path.exists(image_path):
        b64_image = encode_image_to_base64(image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
        })

    user_content.append({
        "type": "text",
        "text": "Describe this image following your instructions exactly.",
    })

    response = client_vl.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        visual_description = json.loads(raw)
    except json.JSONDecodeError:
        visual_description = {"raw_output": raw, "parse_error": True}

    return {"visual_description": visual_description}


def analyzer_node(state: AgentState) -> dict:
    """Step 2: Receives Agent 1 output + farmer transcript + RAG context. Produces differential diagnosis."""
    visual = state["visual_description"]
    transcript = state["transcript"]
    region = state["region_context"]
    system_prompt = get_prompt("analyzer")

    user_prompt = (
        f"## Input 1 — Visual Description (from Agent 1)\n"
        f"```json\n{json.dumps(visual, indent=2)}\n```\n\n"
        f"## Input 2 — Farmer's Statement\n\"{transcript}\"\n\n"
        f"## Input 3 — Regional/Environmental Context (RAG)\n"
        f"```json\n{json.dumps(region, indent=2)}\n```\n\n"
        "Produce your differential diagnosis now."
    )

    response = client_reasoning.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        diagnosis = json.loads(raw)
    except json.JSONDecodeError:
        diagnosis = {"raw_output": raw, "parse_error": True}

    return {"diagnosis": diagnosis}


def verification_node(state: AgentState) -> dict:
    """Step 3: Receives all prior outputs. Stress-tests Agent 2's reasoning — does not generate new analysis."""
    visual = state["visual_description"]
    diagnosis = state["diagnosis"]
    transcript = state["transcript"]
    region = state["region_context"]
    system_prompt = get_prompt("verification")

    user_prompt = (
        f"## Agent 1 — Visual Description (ground truth)\n"
        f"```json\n{json.dumps(visual, indent=2)}\n```\n\n"
        f"## Agent 2 — Differential Diagnosis\n"
        f"```json\n{json.dumps(diagnosis, indent=2)}\n```\n\n"
        f"## Farmer's Original Statement\n\"{transcript}\"\n\n"
        f"## Regional RAG Context\n"
        f"```json\n{json.dumps(region, indent=2)}\n```\n\n"
        "Perform your verification now."
    )

    response = client_verify.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        assessment = json.loads(raw)
    except json.JSONDecodeError:
        assessment = {"raw_output": raw, "parse_error": True}

    return {"verified_assessment": assessment}


def action_plan_node(state: AgentState) -> dict:
    """Step 4: Receives ONLY the verified assessment. Translates into farmer-actionable advice."""
    verified = state["verified_assessment"]
    region = state["region_context"]
    system_prompt = get_prompt("action")

    user_prompt = (
        f"## Verified Diagnosis\n"
        f"```json\n{json.dumps(verified, indent=2)}\n```\n\n"
        f"## Farmer's Region\n"
        f"```json\n{json.dumps(region, indent=2)}\n```\n\n"
        "Generate the action plan for this farmer now."
    )

    response = client_action.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        action_plan = json.loads(raw)
    except json.JSONDecodeError:
        action_plan = {"raw_output": raw, "parse_error": True}

    return {"action_plan": action_plan}

def get_prompt(name: str) -> str:
    with open(f"prompts/{name}.md", "r") as f:
        return f.read()
