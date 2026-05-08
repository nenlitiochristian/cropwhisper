import os
import json
import re

from dotenv import load_dotenv
load_dotenv()

import shutil
from pathlib import Path

import gradio as gr

CUSTOM_CSS = """
footer { display: none !important; }
.skel {
    background: linear-gradient(90deg, #eef2ee 25%, #dde8dd 50%, #eef2ee 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 5px;
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
"""

SCROLL_WRAP_OPEN = '<div style="height:80vh;overflow-y:auto;border:1px solid #dde8dd;border-radius:10px;background:#fafffa;">'
SCROLL_WRAP_CLOSE = '</div>'

SKELETON_HTML = SCROLL_WRAP_OPEN + """
<div style="padding:24px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
    <span style="font-size:22px">🌱</span>
    <span style="color:#4a7a4a;font-size:15px;font-weight:600;letter-spacing:0.3px;">
      Analyzing your crop — this may take a minute…
    </span>
  </div>

  <div class="skel" style="height:22px;width:52%;margin-bottom:14px;"></div>
  <div class="skel" style="height:13px;width:93%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:80%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:87%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:70%;margin:6px 0;"></div>

  <div class="skel" style="height:22px;width:42%;margin:24px 0 14px;"></div>
  <div class="skel" style="height:13px;width:88%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:74%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:82%;margin:6px 0;"></div>

  <div class="skel" style="height:22px;width:48%;margin:24px 0 14px;"></div>
  <div class="skel" style="height:13px;width:91%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:66%;margin:6px 0;"></div>

  <div class="skel" style="height:22px;width:33%;margin:24px 0 14px;"></div>
  <div class="skel" style="height:13px;width:85%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:77%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:90%;margin:6px 0;"></div>

  <div class="skel" style="height:22px;width:55%;margin:24px 0 14px;"></div>
  <div class="skel" style="height:13px;width:95%;margin:6px 0;"></div>
  <div class="skel" style="height:13px;width:78%;margin:6px 0;"></div>
</div>
""" + SCROLL_WRAP_CLOSE

from agent import AgentState, construct_graph, get_all_model_names
from supabase import create_client, Client

# Supabase Configuration
# SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
# SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
# supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# --- Graph Construction ---
app_graph = construct_graph()


def _run_report(
    text: str,
    latitude: float,
    longitude: float,
    image_path: str | None,
):
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
        "image_path": temp_image_path,
        "transcript": text.strip(),
        "region_context": {"lat": float(latitude), "lon": float(longitude)},
        "visual_description": {},
        "diagnosis": {},
        "verified_assessment": {},
        "action_plan": {},
        "language": "en",
    }

    try:
        final_state = app_graph.invoke(initial_state)
    except Exception as exc:
        raise gr.Error(f"Pipeline failed: {exc}") from exc
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    return _format_action_plan(final_state.get("action_plan", {}))


def _format_action_plan(plan: dict) -> str:
    if plan.get("parse_error"):
        raw = plan.get("raw_output", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            return SCROLL_WRAP_OPEN + f'<div style="padding:20px;white-space:pre-wrap">{raw}</div>' + SCROLL_WRAP_CLOSE

    h = [SCROLL_WRAP_OPEN, '<div style="padding:20px;font-family:inherit">']

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


def _refresh_models():
    """Fetch the resolved model name from each vLLM server."""
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
                lat_input = gr.Number(label="Latitude", value=-1.2921)
                lon_input = gr.Number(label="Longitude", value=36.8219)
            run_button = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=1):
            action_out = gr.HTML(elem_id="action-out")

    run_button.click(
        fn=lambda: (SKELETON_HTML, gr.update(interactive=False, value="Analyzing…")),
        inputs=[],
        outputs=[action_out, run_button],
        queue=False,
    ).then(
        fn=_run_report,
        inputs=[transcript_input, lat_input, lon_input, image_input],
        outputs=[action_out],
    ).then(
        fn=lambda: gr.update(interactive=True, value="Run Analysis"),
        inputs=[],
        outputs=[run_button],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
