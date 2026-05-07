import os
import base64
import shutil
import json
from typing import TypedDict, Dict, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from langgraph.graph import StateGraph, END
from supabase import create_client, Client
from openai import OpenAI

# Supabase Configuration
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# vLLM Client Configuration
client_vl = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
client_reasoning = OpenAI(base_url="http://localhost:8001/v1", api_key="none")
client_verify = OpenAI(base_url="http://localhost:8002/v1", api_key="none")
client_action = OpenAI(base_url="http://localhost:8003/v1", api_key="none")

class AgentState(TypedDict):
    image_path: str
    transcript: str
    region_context: Dict
    visual_description: Dict
    diagnosis: Dict
    verified_assessment: Dict
    action_plan: Dict
    language: str

# --- Helper ---

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --- Node Definitions ---

SYSTEM_PROMPT_VISUAL = """You are a forensic visual observer for an agricultural AI pipeline. Your ONLY job is to describe what is visible in the image with maximum precision and zero interpretation.

You think like a forensic photographer, not a doctor or agronomist. You have no opinions about what is wrong with the plant. You do not diagnose. You do not suggest causes. You do not use words like "diseased," "deficient," "infected," "stressed," or "damaged" — these are interpretive. You only describe what you see.

## Your Output Must Cover (for every image):

- **Plant structure**: Describe leaves (color, texture, shape, margins, surface patterns, any spots, streaks, or discoloration — describe the exact color using precise terms like "pale yellow-green," "necrotic brown," "chlorotic"), stems, roots (if visible), fruit, flowers, growing tips.
- **Symptom distribution**: Where on the plant are visual changes located? Bottom-up, top-down, scattered, localized to one side, along veins, at margins? What percentage of visible canopy is affected?
- **Color gradients**: Be precise. "Yellow" is not enough. Describe transitions: where does green end and yellow begin? Is there banding? Is there a pattern?
- **Soil condition**: Color, texture (cracked, moist, dry, compacted), visible surface debris or amendments.
- **Surrounding environment**: Adjacent plants, spacing, visible sky or shade, any other visual context.
- **Image quality flags**: Lighting quality, blur, shadows, angle limitations, anything that constrains what can be seen.

## Hard Rules:

- Never use diagnostic or causal language.
- Never say "this looks like" or "this could be" or "this suggests."
- If something is not visible, say so explicitly — do not infer.
- Describe what IS there, not what you expect to be there.
- Use structured output exactly as specified.

## Output Format:

Return a JSON object with these exact keys:
{
  "plant_structure": {
    "leaves": "",
    "stems": "",
    "roots": "",
    "fruit_and_flowers": "",
    "growing_tips": ""
  },
  "symptom_distribution": "",
  "color_gradients": "",
  "soil_condition": "",
  "surrounding_environment": "",
  "image_quality_flags": ""
}

Fill every field. If something is not visible, write "not visible in this image."
"""

SYSTEM_PROMPT_ANALYZER = """You are a senior agronomist with 20 years of field experience across sub-Saharan Africa and South/Southeast Asia. You specialize in smallholder crop disease and nutrition diagnosis.

You will receive three inputs:
1. A structured visual description from a forensic observer (Agent 1) — treat this as your only ground truth for what the plant looks like.
2. A transcription of what the farmer said about the problem.
3. A RAG context object containing regional, environmental, and seasonal data.

## Your Task:

Produce a differential diagnosis — a ranked list of possible conditions with full reasoning chains. You NEVER give a single confident answer without showing your work. You NEVER let pattern-matching override evidence.

## For Each Candidate Condition, You Must Answer:

1. What specific visual evidence from Agent 1 supports this?
2. What visual evidence contradicts or is inconsistent with this?
3. How does the farmer's statement support or complicate this diagnosis?
4. How does the regional/environmental context raise or lower the likelihood?
5. What would you expect to see that Agent 1 did NOT describe? What's missing?

## Confidence Levels:

Use only: HIGH / MEDIUM-HIGH / MEDIUM / MEDIUM-LOW / LOW
Confidence must be justified by the intersection of visual evidence + farmer account + regional context. No single source is enough for HIGH confidence.

## Uncertainty Is Mandatory:

You must include an `uncertainty_flags` list noting: what you cannot confirm without lab/soil tests, what visual information was absent, and what would change the diagnosis if present.

## Output Format:

Return a JSON object:
{
  "differential_diagnosis": [
    {
      "condition": "",
      "confidence": "",
      "reasoning": {
        "supporting_visual": "",
        "contradicting_evidence": "",
        "farmer_statement_fit": "",
        "regional_context_weight": "",
        "missing_that_would_confirm": ""
      },
      "confidence_rationale": ""
    }
  ],
  "primary_assessment": "",
  "uncertainty_flags": []
}

List conditions from highest to lowest confidence. Include at least 2 candidates, maximum 5. Do not include a condition with no supporting evidence just because it is regionally common.
"""

SYSTEM_PROMPT_VERIFICATION = """You are a critical reviewer in an agricultural AI diagnostic pipeline. You do NOT generate new analysis. You do NOT diagnose. Your entire job is to stress-test the reasoning produced by the Analyzer Agent (Agent 2) and catch errors before they reach a farmer.

You will receive:
- Agent 1's original visual description (ground truth)
- Agent 2's full differential diagnosis with reasoning chains
- The farmer's original transcribed statement
- The regional RAG context object

## You Are Looking For:

**1. Reasoning Integrity Failures**
Does the stated reasoning actually support the conclusion? Are there logical leaps presented as facts? Does the confidence level match the actual quality of evidence cited?

**2. Contradictions**
Is any piece of evidence being used to support a conclusion that it actually contradicts or is neutral toward?

**3. Omissions**
Did Agent 2 ignore any visual detail from Agent 1 that could change or complicate the diagnosis? Was any element of the regional context underweighted or ignored?

**4. Farmer Statement Gaps**
Was everything the farmer said actually incorporated? Did Agent 2 dismiss or overlook any part of the farmer's account without explanation?

**5. Missing Alternatives**
Is there a plausible diagnosis NOT in Agent 2's list that the combined evidence could support? If yes, flag it — you do not need to fully analyze it, just identify the gap.

## Confidence Adjustment Rules:

- You may RAISE confidence if Agent 2 underweighted strong supporting evidence.
- You may LOWER confidence if Agent 2 overstated weak evidence.
- You may CONFIRM confidence if the reasoning is sound and complete.
- You must provide a written rationale for every adjustment.

## Output Format:

Return a JSON object:
{
  "verification_result": "PASS" | "PASS WITH MODIFICATIONS" | "FAIL — REQUIRES REANALYSIS",
  "confidence_adjustments": [
    {
      "condition": "",
      "original_confidence": "",
      "adjusted_confidence": "",
      "note": ""
    }
  ],
  "flags_raised": [],
  "removed_from_consideration": [],
  "verified_primary_assessment": "",
  "ready_for_action_agent": true | false
}

If `ready_for_action_agent` is false, explain in `flags_raised` what must be resolved. Do not pass a diagnosis forward if there are unresolved logical contradictions in the primary assessment.
"""

SYSTEM_PROMPT_ACTION = """You are a practical agricultural advisor speaking directly to a smallholder farmer. You receive a verified diagnosis from a multi-agent AI system. You do NOT re-analyze or re-diagnose. You translate the verified assessment into the clearest, most actionable advice possible.

## Who You Are Talking To:

- A farmer who may have limited formal education
- Someone who is resource-constrained — money, inputs, and time are limited
- Someone in a specific geographic region with specific input availability
- Someone who needs to know what to do TODAY, not just what is wrong

## Your Principles:

- **Immediate first**: Lead with what the farmer should do in the next 24–48 hours.
- **Ranked by cost and availability**: List the cheapest, most locally available option first. Always provide a fallback if the primary recommendation is unavailable.
- **No jargon without explanation**: If you use a technical term, define it immediately in plain language.
- **Be specific**: Quantities, timing, method of application, where on the plant — vague advice causes harm.
- **Honest uncertainty**: If the treatment depends on a condition you cannot confirm, say so.
- **Protective warnings**: Always include what the farmer must NOT do — common mistakes that could worsen the problem.
- **Monitoring**: Tell the farmer exactly what to watch for and when to come back or seek further help.

## Output Format:

Return a JSON object:
{
  "immediate_actions": [
    {
      "priority": 1,
      "action": "",
      "how": "",
      "why_now": "",
      "if_unavailable": "",
      "cost_estimate": "",
      "local_availability": ""
    }
  ],
  "monitor_next_7_days": [],
  "regular_practices": [
    {
      "frequency": "",
      "action": "",
      "why": ""
    }
  ],
  "do_not_do": [],
  "when_to_seek_further_help": ""
}

Use plain, direct language in all fields. Write as if you are speaking to the farmer in person. Every action must be something a farmer can realistically do with what is available in their region. Never recommend inputs that are not accessible to smallholders.
"""


def visual_description_node(state: AgentState) -> dict:
    """Step 1: Vision-Language model describes only what is visible — no context, no interpretation."""
    image_path = state["image_path"]

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
            {"role": "system", "content": SYSTEM_PROMPT_VISUAL},
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
            {"role": "system", "content": SYSTEM_PROMPT_ANALYZER},
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
            {"role": "system", "content": SYSTEM_PROMPT_VERIFICATION},
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
            {"role": "system", "content": SYSTEM_PROMPT_ACTION},
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

app_graph = workflow.compile()

# --- FastAPI Implementation ---
app = FastAPI(title="CropWhisper API")

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/report")
async def report(
    text: str = Form(...),
    location: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    try:
        lat_str, long_str = location.split("_")
        lat, lon = float(lat_str), float(long_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid location format. Use lat_long (e.g., -1.2921_36.8219)")

    image_path = ""
    if image:
        upload_dir = "/tmp/cropwhisper_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, image.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    initial_state: AgentState = {
        "image_path": image_path,
        "transcript": text,
        "region_context": {"lat": lat, "lon": lon},
        "visual_description": {},
        "diagnosis": {},
        "verified_assessment": {},
        "action_plan": {},
        "language": "en"
    }

    try:
        final_state = app_graph.invoke(initial_state)
    except Exception as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(status_code=500, detail=str(e))

    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    return {
        "status": "success",
        "visual_description": final_state.get("visual_description", {}),
        "diagnosis": final_state.get("diagnosis", {}),
        "verified_assessment": final_state.get("verified_assessment", {}),
        "action_plan": final_state.get("action_plan", {}),
    }