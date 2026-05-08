import os
import json
import re
import time

from dotenv import load_dotenv
load_dotenv()

import shutil
from pathlib import Path

import gradio as gr

from agent import AgentState, construct_graph, get_all_model_names
from supabase import create_client, Client

# --- Graph Construction ---
app_graph = construct_graph()

CUSTOM_CSS = """
footer { display: none !important; }
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #81c784;
    border-radius: 50%;
    animation: blink 1s ease-in-out infinite;
}
"""

SCROLL_WRAP_OPEN  = '<div style="height:80vh;overflow-y:auto;border:1px solid #dde8dd;border-radius:10px;background:#fafffa;">'
SCROLL_WRAP_CLOSE = '</div>'

AGENT_ORDER = ["agent_1_visual", "agent_2_analyzer", "agent_3_verify", "agent_4_action"]
AGENT_META  = {
    "agent_1_visual":   ("🔬", "Visual Analysis"),
    "agent_2_analyzer": ("🧬", "Crop Diagnosis"),
    "agent_3_verify":   ("✅", "Verification"),
    "agent_4_action":   ("📋", "Action Plan"),
}
AGENT_STATE_KEY = {
    "agent_1_visual":   "visual_description",
    "agent_2_analyzer": "diagnosis",
    "agent_3_verify":   "verified_assessment",
    "agent_4_action":   "action_plan",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_text(agent_id: str, data: dict) -> str:
    key     = AGENT_STATE_KEY[agent_id]
    payload = data.get(key, data)
    if isinstance(payload, dict) and payload.get("parse_error"):
        raw = payload.get("raw_output", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw
    return _dict_to_lines(payload)


def _dict_to_lines(obj, indent: int = 0) -> str:
    pad   = "  " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_dict_to_lines(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                lines.append(_dict_to_lines(item, indent))
            else:
                lines.append(f"{pad}• {item}")
    else:
        lines.append(f"{pad}{obj}")
    return "\n".join(lines)


def _pipeline_html(completed: dict, running: str | None, streaming: tuple[str, str] | None = None) -> str:
    """Build the pipeline progress view.
    completed: {agent_id: full_text}
    running: agent_id currently waiting for LLM
    streaming: (agent_id, partial_text) currently being revealed
    """
    step = len(completed)
    pct  = int((step / 4) * 100)

    h = [SCROLL_WRAP_OPEN, '<div style="padding:20px 24px;font-family:inherit">']

    # Header
    h.append(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">'
        f'  <span style="font-size:22px">🌱</span>'
        f'  <div>'
        f'    <div style="font-weight:700;font-size:16px;color:#1b5e20">Analyzing your crop</div>'
        f'    <div style="font-size:12px;color:#666;margin-top:2px">Step {step} of 4 complete</div>'
        f'  </div>'
        f'</div>'
    )

    # Progress bar
    h.append(
        f'<div style="background:#c8e6c9;border-radius:6px;height:7px;margin-bottom:24px">'
        f'  <div style="background:linear-gradient(90deg,#388e3c,#81c784);width:{pct}%;height:100%;border-radius:6px;transition:width 0.6s ease"></div>'
        f'</div>'
    )

    for agent_id in AGENT_ORDER:
        icon, label = AGENT_META[agent_id]
        is_done      = agent_id in completed
        is_streaming = streaming and streaming[0] == agent_id
        is_running   = agent_id == running

        if agent_id == "agent_4_action":
            # action plan never shown in pipeline view
            status_icon  = "✅" if is_done else ("⚙️" if is_running else "○")
            status_color = "#1b5e20" if is_done else ("#4a7a4a" if is_running else "#aaa")
            h.append(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;opacity:{"1" if (is_done or is_running) else "0.4"}">'
                f'  <span style="font-size:16px">{status_icon}</span>'
                f'  <span style="font-weight:600;color:{status_color}">{label}</span>'
                f'  {"<span class=\"pulse-dot\"></span>" if is_running else ""}'
                f'</div>'
            )
            continue

        if is_done or is_streaming:
            text         = completed.get(agent_id, "") if is_done else streaming[1]
            cursor       = "" if is_done else '<span style="animation:blink 0.8s infinite">▌</span>'
            check        = "✅" if is_done else icon
            anim         = 'animation:fadeSlideIn 0.35s ease' if not is_streaming else ''
            h.append(
                f'<div style="margin-bottom:18px;{anim}">'
                f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'    <span style="font-size:15px">{check}</span>'
                f'    <span style="font-weight:600;color:#1b5e20">{label}</span>'
                f'  </div>'
                f'  <div style="background:#0d1a0d;color:#a8d5a8;font-family:\'Courier New\',monospace;'
                f'              font-size:12px;line-height:1.65;padding:14px 16px;border-radius:8px;'
                f'              white-space:pre-wrap;max-height:180px;overflow-y:auto">'
                f'{text}{cursor}'
                f'  </div>'
                f'</div>'
            )
        elif is_running:
            h.append(
                f'<div style="margin-bottom:18px">'
                f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'    <span style="font-size:15px">{icon}</span>'
                f'    <span style="font-weight:600;color:#4a7a4a">{label}</span>'
                f'    <span class="pulse-dot"></span>'
                f'  </div>'
                f'  <div style="background:#0d1a0d;color:#2d5a2d;font-family:\'Courier New\',monospace;'
                f'              font-size:12px;padding:14px 16px;border-radius:8px">'
                f'    <span style="animation:blink 0.8s infinite">▌</span>'
                f'  </div>'
                f'</div>'
            )
        else:
            h.append(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;opacity:0.35">'
                f'  <span style="font-size:15px">○</span>'
                f'  <span style="font-weight:500;color:#888">{label}</span>'
                f'</div>'
            )

    h.append('</div>')
    h.append(SCROLL_WRAP_CLOSE)
    return "".join(h)


def _format_action_plan(plan: dict) -> str:
    if plan.get("parse_error"):
        raw = plan.get("raw_output", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            return SCROLL_WRAP_OPEN + f'<div style="padding:20px;white-space:pre-wrap">{raw}</div>' + SCROLL_WRAP_CLOSE

    h = [SCROLL_WRAP_OPEN, '<div style="padding:20px;font-family:inherit;animation:fadeSlideIn 0.4s ease">']

    condition = plan.get("condition", "")
    if condition:
        h.append(
            f'<div style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:8px;padding:14px 18px;margin-bottom:20px;">'
            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#4a7a4a;margin-bottom:4px;">Crop Condition</div>'
            f'<div style="font-size:18px;font-weight:600;color:#1b5e20;">{condition}</div>'
            f'</div>'
        )

    def section(icon, title):
        h.append(f'<h2 style="color:#2d5a2d;border-bottom:2px solid #c8e6c8;padding-bottom:6px;margin-top:24px">{icon} {title}</h2>')

    immediate = plan.get("immediate_actions", [])
    if immediate:
        section("⚡", "Immediate Actions")
        for item in immediate:
            h.append(f'<h3 style="color:#3a6b3a;margin:14px 0 6px">{item.get("priority","")}. {item.get("action","")}</h3>')
            for label, key in [("How", "how"), ("Why now", "why_now"), ("If unavailable", "if_unavailable"), ("Cost", "cost_estimate"), ("Local availability", "local_availability")]:
                if item.get(key):
                    h.append(f'<p style="margin:4px 0"><strong>{label}:</strong> {item[key]}</p>')

    monitor = plan.get("monitor_next_7_days", [])
    if monitor:
        section("👁️", "Monitor Next 7 Days")
        h.append('<ul style="margin:8px 0;padding-left:20px">')
        for m in monitor:
            h.append(f'<li style="margin:4px 0">{m}</li>')
        h.append('</ul>')

    practices = plan.get("regular_practices", [])
    if practices:
        section("🔄", "Regular Practices")
        for p in practices:
            h.append(f'<p style="margin:8px 0"><strong>{p.get("frequency","")}:</strong> {p.get("action","")}</p>')
            if p.get("why"):
                h.append(f'<p style="margin:2px 0 8px;color:#666;font-style:italic">{p["why"]}</p>')

    do_not = plan.get("do_not_do", [])
    if do_not:
        section("🚫", "Do Not Do")
        h.append('<ul style="margin:8px 0;padding-left:20px">')
        for d in do_not:
            h.append(f'<li style="margin:4px 0;color:#8b0000">{d}</li>')
        h.append('</ul>')

    seek_help = plan.get("when_to_seek_further_help", "")
    if seek_help:
        section("🏥", "When to Seek Further Help")
        h.append(f'<p style="margin:8px 0;padding:12px;background:#fff8e1;border-left:4px solid #f9a825;border-radius:4px">{seek_help}</p>')

    h.append('</div>')
    h.append(SCROLL_WRAP_CLOSE)
    return "".join(h)


# ── main generator ────────────────────────────────────────────────────────────

def _run_report(text: str, latitude: float, longitude: float, image_path: str | None):
    if not text or not text.strip():
        raise gr.Error("Please provide a farmer transcript before running analysis.")
    if latitude is None or longitude is None:
        raise gr.Error("Please provide both latitude and longitude.")

    temp_image_path = ""
    if image_path:
        upload_dir = Path("/tmp/cropwhisper_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        source = Path(image_path)
        temp_image_path = str(upload_dir / source.name)
        shutil.copyfile(source, temp_image_path)

    initial_state: AgentState = {
        "image_path":        temp_image_path,
        "transcript":        text.strip(),
        "region_context":    {"lat": float(latitude), "lon": float(longitude)},
        "visual_description": {},
        "diagnosis":          {},
        "verified_assessment": {},
        "action_plan":        {},
        "language":           "en",
    }

    completed: dict[str, str] = {}
    btn_disabled = gr.update(interactive=False, value="Analyzing…")
    btn_enabled  = gr.update(interactive=True,  value="Run Analysis")

    # Initial state: all pending, first agent "running"
    yield _pipeline_html(completed, running="agent_1_visual"), btn_disabled

    try:
        for chunk in app_graph.stream(initial_state):
            node_name = list(chunk.keys())[0]
            node_data = chunk[node_name]

            if node_name == "agent_4_action":
                action_plan = node_data.get("action_plan", {})
                yield _format_action_plan(action_plan), btn_enabled
                return

            # Stream the agent's text output line by line
            full_text = _extract_text(node_name, node_data)
            lines     = [l for l in full_text.splitlines() if l.strip()]

            revealed = []
            for line in lines:
                revealed.append(line)
                partial = "\n".join(revealed)
                yield _pipeline_html(completed, running=None, streaming=(node_name, partial)), btn_disabled
                time.sleep(0.07)

            completed[node_name] = full_text

            # Determine next agent to show as "running"
            idx          = AGENT_ORDER.index(node_name)
            next_running = AGENT_ORDER[idx + 1] if idx + 1 < len(AGENT_ORDER) else None
            yield _pipeline_html(completed, running=next_running), btn_disabled

    except Exception as exc:
        yield (
            SCROLL_WRAP_OPEN
            + f'<div style="padding:20px;color:#c62828">Pipeline failed: {exc}</div>'
            + SCROLL_WRAP_CLOSE
        ), btn_enabled
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def _refresh_models():
    names = get_all_model_names()
    return "\n".join(f"- **{role}**: `{model}`" for role, model in names.items())


with gr.Blocks(title="CropWhisper", css=CUSTOM_CSS) as demo:
    gr.Markdown("# CropWhisper")
    gr.Markdown("Upload a crop image, add transcript and coordinates, then run the 4-agent pipeline.")

    with gr.Accordion("vLLM Model Info", open=False):
        model_info = gr.Markdown("Click **Refresh** to query vLLM servers for loaded model names.")
        refresh_btn = gr.Button("Refresh", size="sm")
        refresh_btn.click(fn=_refresh_models, inputs=[], outputs=[model_info])

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Crop image (optional)")
            transcript_input = gr.Textbox(
                label="Farmer transcript",
                lines=5,
                placeholder="Describe what the farmer said about the crop condition...",
            )
            with gr.Row():
                lat_input = gr.Number(label="Latitude",  value=-1.2921)
                lon_input = gr.Number(label="Longitude", value=36.8219)
            run_button = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=1):
            action_out = gr.HTML(elem_id="action-out")

    run_button.click(
        fn=_run_report,
        inputs=[transcript_input, lat_input, lon_input, image_input],
        outputs=[action_out, run_button],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
