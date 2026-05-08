import os
import shutil
from pathlib import Path

import gradio as gr

from agent import AgentState, construct_graph
from supabase import create_client, Client

# Supabase Configuration
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

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

    return (
        final_state.get("visual_description", {}),
        final_state.get("diagnosis", {}),
        final_state.get("verified_assessment", {}),
        final_state.get("action_plan", {}),
    )


with gr.Blocks(title="CropWhisper") as demo:
    gr.Markdown("# CropWhisper")
    gr.Markdown("Upload a crop image, add transcript and coordinates, then run the 4-agent pipeline.")

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
            visual_out = gr.JSON(label="Agent 1: Visual Description")
            diagnosis_out = gr.JSON(label="Agent 2: Differential Diagnosis")
            verify_out = gr.JSON(label="Agent 3: Verification")
            action_out = gr.JSON(label="Agent 4: Action Plan")

    run_button.click(
        fn=_run_report,
        inputs=[transcript_input, lat_input, lon_input, image_input],
        outputs=[visual_out, diagnosis_out, verify_out, action_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
