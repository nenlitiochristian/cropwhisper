import os
import shutil
from typing import Optional
from agent import AgentState, construct_graph
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from supabase import create_client, Client

# Supabase Configuration
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# --- Graph Construction ---
app_graph = construct_graph()

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
