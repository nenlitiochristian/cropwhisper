
import json
import os
import re
from typing import TypedDict, Dict

from utils.image import encode_image_to_base64

from openai import OpenAI
from langgraph.graph import StateGraph, END
from supabase import create_client, Client as SupabaseClient


class AgentState(TypedDict):
    image_paths: list[str]
    transcript: str
    region_context: Dict
    visual_description: Dict
    diagnosis: Dict
    verified_assessment: Dict
    action_plan: Dict
    language: str


class FollowUpState(TypedDict):
    followup_image_paths: list[str]
    images_provided_labels: list[str]
    all_images_submitted: list[str]
    established_facts: list[str]
    prior_gaps: list[str]
    initial_action_plan: dict
    prior_followup_action: dict
    region_context: dict
    generated_prompt: dict
    adjusted_diagnosis: dict
    followup_verification: dict
    followup_action: dict


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost")

client_vl = OpenAI(base_url=f"{VLLM_BASE_URL}:8000/v1", api_key="none") #QwenVL-7B-Instruct
client_reasoning = OpenAI(base_url=f"{VLLM_BASE_URL}:8001/v1", api_key="none") #Qwen3-32B
client_verify = OpenAI(base_url=f"{VLLM_BASE_URL}:8002/v1", api_key="none") #Qwen3-14B
client_action = OpenAI(base_url=f"{VLLM_BASE_URL}:8003/v1", api_key="none") #Qwen3-14B

_model_name_cache: Dict[str, str] = {}

_supabase: SupabaseClient | None = None
_rag_cache: list[dict] | None = None


def _get_supabase() -> SupabaseClient | None:
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if url and key:
            _supabase = create_client(url, key)
    return _supabase


def _visual_to_text(visual: dict) -> str:
    """Flatten the visual description dict into a plain text blob for keyword matching."""
    if visual.get("parse_error"):
        raw = visual.get("raw_output", "")
        return re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    parts = []
    # Keys from disease_visual.md schema (used by RAG seed documents)
    # AND keys from visual.md schema (used by live Agent 1 output).
    # Both sets are needed because RAG documents use disease_visual keys
    # while the live pipeline produces visual.md keys.
    for key in [
        "lesions_and_anomalies", "color_and_gradients", "distribution_pattern",
        "plant_structure", "symptom_distribution", "color_gradients",
        "soil_condition", "surrounding_environment",
    ]:
        val = visual.get(key, "")
        if isinstance(val, dict):
            parts.append(" ".join(str(v) for v in val.values()))
        elif isinstance(val, str):
            parts.append(val)
    return " ".join(parts)


def _rag_lookup(visual: dict, top_k: int = 5) -> list[dict]:
    """Return top_k rag_documents rows whose visual_description best matches the query."""
    global _rag_cache
    sb = _get_supabase()
    if sb is None:
        return []

    if _rag_cache is None:
        try:
            _rag_cache = sb.table("rag_documents").select(
                "crop,condition,visual_description"
            ).execute().data
        except Exception:
            return []

    query_words = set(
        w for w in _visual_to_text(visual).lower().split() if len(w) > 3
    )
    if not query_words:
        return []

    scored = []
    for row in _rag_cache:
        doc_words = set(
            w for w in (row.get("visual_description") or "").lower().split() if len(w) > 3
        )
        overlap = len(query_words & doc_words)
        if overlap > 0:
            scored.append((overlap, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def get_model_name(client: OpenAI) -> str:
    """Query the vLLM server to get the actual loaded model name (cached after first call)."""
    key = client.base_url.host + str(client.base_url.port)
    if key not in _model_name_cache:
        models = client.models.list()
        _model_name_cache[key] = models.data[0].id
    return _model_name_cache[key]


def get_all_model_names() -> Dict[str, str]:
    """Return a mapping of agent role -> resolved model name for all four vLLM servers."""
    result = {}
    for label, client in [
        ("Agent 1 — Visual (port 8000)", client_vl),
        ("Agent 2 — Analyzer (port 8001)", client_reasoning),
        ("Agent 3 — Verifier (port 8002)", client_verify),
        ("Agent 4 — Action (port 8003)", client_action),
    ]:
        try:
            result[label] = get_model_name(client)
        except Exception as exc:
            result[label] = f"unavailable ({exc})"
    return result


# ── Initial Pipeline ─────────────────────────────────────────────────────────

def construct_graph():
    workflow = StateGraph(AgentState)
    AGENT_1 = "agent_1_visual"
    AGENT_2 = "agent_2_analyzer"
    AGENT_3 = "agent_3_verify"
    AGENT_4 = "agent_4_action"

    workflow.add_node(AGENT_1, _visual_description_node)
    workflow.add_node(AGENT_2, _analyzer_node)
    workflow.add_node(AGENT_3, _verification_node)
    workflow.add_node(AGENT_4, _action_plan_node)

    workflow.set_entry_point(AGENT_1)
    workflow.add_edge(AGENT_1, AGENT_2)
    workflow.add_edge(AGENT_2, AGENT_3)
    workflow.add_edge(AGENT_3, AGENT_4)
    workflow.add_edge(AGENT_4, END)

    return workflow.compile()


def _visual_description_node(state: AgentState) -> dict:
    """Step 1: Vision-Language model describes only what is visible."""
    image_paths = state.get("image_paths") or []
    system_prompt = _get_prompt("visual")

    user_content = []
    for img_path in image_paths:
        if img_path and os.path.exists(img_path):
            b64_image = encode_image_to_base64(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            })

    num = len([p for p in image_paths if p and os.path.exists(p)])
    text_suffix = "these images" if num > 1 else "this image"
    user_content.append({
        "type": "text",
        "text": f"Describe {text_suffix} following your instructions exactly.",
    })

    response = client_vl.chat.completions.create(
        model=get_model_name(client_vl),
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


def _analyzer_node(state: AgentState) -> dict:
    """Step 2: Receives Agent 1 output + farmer transcript + RAG context. Produces differential diagnosis."""
    visual = state["visual_description"]
    transcript = state["transcript"]
    region = state["region_context"]
    system_prompt = _get_prompt("analyzer")

    similar_cases = _rag_lookup(visual)
    rag_db_section = ""
    if similar_cases:
        lines = []
        for doc in similar_cases:
            snippet = (doc.get("visual_description") or "")[:300].replace("\n", " ")
            lines.append(f"- {doc.get('crop')} / {doc.get('condition')}: {snippet}…")
        rag_db_section = (
            "\n\n## Input 4 — Similar Cases from Disease Database\n"
            "Use these confirmed disease cases to inform your differential diagnosis:\n"
            + "\n".join(lines)
        )

    user_prompt = (
        f"## Input 1 — Visual Description (from Agent 1)\n"
        f"```json\n{json.dumps(visual, indent=2)}\n```\n\n"
        f"## Input 2 — Farmer's Statement\n\"{transcript}\"\n\n"
        f"## Input 3 — Regional/Environmental Context\n"
        f"```json\n{json.dumps(region, indent=2)}\n```"
        f"{rag_db_section}\n\n"
        "Produce your differential diagnosis now."
    )

    response = client_reasoning.chat.completions.create(
        model=get_model_name(client_reasoning),
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


def _verification_node(state: AgentState) -> dict:
    """Step 3: Receives all prior outputs. Stress-tests Agent 2's reasoning."""
    visual = state["visual_description"]
    diagnosis = state["diagnosis"]
    transcript = state["transcript"]
    region = state["region_context"]
    system_prompt = _get_prompt("verification")

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
        model=get_model_name(client_verify),
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


def _action_plan_node(state: AgentState) -> dict:
    """Step 4: Receives ONLY the verified assessment. Translates into farmer-actionable advice."""
    verified = state["verified_assessment"]
    region = state["region_context"]
    system_prompt = _get_prompt("action")

    user_prompt = (
        f"## Verified Diagnosis\n"
        f"```json\n{json.dumps(verified, indent=2)}\n```\n\n"
        f"## Farmer's Region\n"
        f"```json\n{json.dumps(region, indent=2)}\n```\n\n"
        "Generate the action plan for this farmer now."
    )

    response = client_action.chat.completions.create(
        model=get_model_name(client_action),
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


# ── Follow-Up Pipeline ──────────────────────────────────────────────────────

def construct_followup_graph():
    workflow = StateGraph(FollowUpState)
    F1 = "followup_prompt_generator"
    F2 = "followup_diagnosis_adjuster"
    F3 = "followup_verification"
    F4 = "followup_action"

    workflow.add_node(F1, _followup_prompt_generator_node)
    workflow.add_node(F2, _followup_diagnosis_adjuster_node)
    workflow.add_node(F3, _followup_verification_node)
    workflow.add_node(F4, _followup_action_node)

    workflow.set_entry_point(F1)
    workflow.add_edge(F1, F2)
    workflow.add_edge(F2, F3)
    workflow.add_edge(F3, F4)
    workflow.add_edge(F4, END)

    return workflow.compile()


def _followup_prompt_generator_node(state: FollowUpState) -> dict:
    """F1: Distill cumulative case context into a focused prompt for downstream agents."""
    system_prompt = _get_prompt("followup_prompt_generator")

    user_prompt = (
        f"## Established Facts from Prior Analysis\n"
        f"```json\n{json.dumps(state['established_facts'], indent=2)}\n```\n\n"
        f"## Prior Gaps / Uncertainties\n"
        f"```json\n{json.dumps(state['prior_gaps'], indent=2)}\n```\n\n"
        f"## Initial Action Plan\n"
        f"```json\n{json.dumps(state['initial_action_plan'], indent=2)}\n```\n\n"
        f"## Previous Follow-Up Action (if any)\n"
        f"```json\n{json.dumps(state.get('prior_followup_action') or {}, indent=2)}\n```\n\n"
        f"## New Evidence Provided by User\n"
        f"The user has submitted {len(state['followup_image_paths'])} new image(s):\n"
        f"{json.dumps(state['images_provided_labels'], indent=2)}\n\n"
        f"## All Images Submitted Across All Rounds\n"
        f"{json.dumps(state['all_images_submitted'], indent=2)}\n\n"
        "Generate the focused follow-up prompt now."
    )

    response = client_reasoning.chat.completions.create(
        model=get_model_name(client_reasoning),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        generated_prompt = json.loads(raw)
    except json.JSONDecodeError:
        generated_prompt = {"raw_output": raw, "parse_error": True}

    return {"generated_prompt": generated_prompt}


def _followup_diagnosis_adjuster_node(state: FollowUpState) -> dict:
    """F2: Analyze new evidence images and cross-reference against established facts."""
    system_prompt = _get_prompt("followup_diagnosis_adjuster")

    # First, get visual descriptions of new images via the VL model
    image_paths = state.get("followup_image_paths") or []
    user_vl_content = []
    for img_path in image_paths:
        if img_path and os.path.exists(img_path):
            b64_image = encode_image_to_base64(img_path)
            user_vl_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            })

    generated_prompt = state.get("generated_prompt") or {}
    focus = generated_prompt.get("focus_instruction", "Analyze the new images carefully.")

    user_vl_content.append({
        "type": "text",
        "text": (
            f"Describe what you see in these follow-up images. "
            f"Focus on: {focus}"
        ),
    })

    vl_response = client_vl.chat.completions.create(
        model=get_model_name(client_vl),
        messages=[
            {"role": "system", "content": _get_prompt("visual")},
            {"role": "user", "content": user_vl_content},
        ],
        max_tokens=2048,
    )
    new_visual_raw = vl_response.choices[0].message.content

    # Now feed visual output + generated prompt to reasoning model
    user_prompt = (
        f"## Generated Prompt from Follow-Up Prompt Generator\n"
        f"```json\n{json.dumps(generated_prompt, indent=2)}\n```\n\n"
        f"## Visual Description of New Evidence Images\n"
        f"{new_visual_raw}\n\n"
        "Produce your adjusted diagnosis now."
    )

    response = client_reasoning.chat.completions.create(
        model=get_model_name(client_reasoning),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        adjusted = json.loads(raw)
    except json.JSONDecodeError:
        adjusted = {"raw_output": raw, "parse_error": True}

    return {"adjusted_diagnosis": adjusted}


def _followup_verification_node(state: FollowUpState) -> dict:
    """F3: Stress-test the adjusted diagnosis against the evidence chain."""
    system_prompt = _get_prompt("followup_verification")
    adjusted = state.get("adjusted_diagnosis") or {}
    generated_prompt = state.get("generated_prompt") or {}

    user_prompt = (
        f"## Follow-Up Diagnosis Adjuster Output\n"
        f"```json\n{json.dumps(adjusted, indent=2)}\n```\n\n"
        f"## Established Facts (from prior rounds)\n"
        f"```json\n{json.dumps(state.get('established_facts', []), indent=2)}\n```\n\n"
        f"## Generated Prompt Context\n"
        f"```json\n{json.dumps(generated_prompt, indent=2)}\n```\n\n"
        "Perform your follow-up verification now."
    )

    response = client_verify.chat.completions.create(
        model=get_model_name(client_verify),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        verification = json.loads(raw)
    except json.JSONDecodeError:
        verification = {"raw_output": raw, "parse_error": True}

    return {"followup_verification": verification}


def _followup_action_node(state: FollowUpState) -> dict:
    """F4: Generate diff-style updates to the original action plan."""
    system_prompt = _get_prompt("followup_action")
    verification = state.get("followup_verification") or {}
    initial_plan = state.get("initial_action_plan") or {}
    generated_prompt = state.get("generated_prompt") or {}
    remaining_gaps = generated_prompt.get("remaining_gaps", [])

    user_prompt = (
        f"## Verified Follow-Up Assessment\n"
        f"```json\n{json.dumps(verification, indent=2)}\n```\n\n"
        f"## Original Action Plan\n"
        f"```json\n{json.dumps(initial_plan, indent=2)}\n```\n\n"
        f"## Remaining Gaps\n"
        f"```json\n{json.dumps(remaining_gaps, indent=2)}\n```\n\n"
        f"## All Images Already Submitted\n"
        f"```json\n{json.dumps(state.get('all_images_submitted', []), indent=2)}\n```\n\n"
        "Generate the follow-up action plan diff now."
    )

    response = client_action.chat.completions.create(
        model=get_model_name(client_action),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        followup_action = json.loads(raw)
    except json.JSONDecodeError:
        followup_action = {"raw_output": raw, "parse_error": True}

    return {"followup_action": followup_action}


def _get_prompt(name: str) -> str:
    with open(f"prompts/{name}.md", "r") as f:
        return f.read()
