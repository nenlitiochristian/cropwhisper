# CropWhisper

**AI-Powered Crop Condition Analysis for Smallholder Farmers**

---

## What is CropWhisper?

CropWhisper is a multimodal agentic AI system that diagnoses crop diseases and nutrient deficiencies from photos, then delivers a plain-language, actionable treatment plan. It is designed for the 500 million+ smallholder farmers worldwide who lack access to agronomists, soil labs, or agricultural extension services — but do have a basic smartphone.

This repository is the **demo implementation**, built as a Gradio web application. In a production release, CropWhisper would be deployed as a mobile app with real-time camera capture and voice interaction — the farmer points the camera, describes the problem in their own language, and receives a spoken action plan. No literacy or agricultural training required. The Gradio demo replicates the core analytical pipeline while providing a visual interface for testing, evaluation, and stakeholder demonstration.

---

## Application Flow

### 1. Input

The user provides three pieces of information:

| Input | Details |
|---|---|
| **Location** | Latitude/longitude via manual entry, a "Detect My Location" button, or a "Check" button for reverse geocoding. Country name and flag are displayed after lookup. |
| **Images** | One or more crop photos (total size capped at 500 KB). Supports file upload or live camera capture. |
| **Video** (optional) | A short clip of the affected crop. CropWhisper extracts 2-3 distinct keyframes using OpenCV and feeds them into the pipeline alongside any photos. |
| **Description** | Free-text description of the problem in the farmer's own words. |

### 2. Soil Data Preflight

After clicking **Run Analysis**, the app fetches the closest natural soil profile from the ISRIC SoilGrids global database. The user sees a preview showing what data was found and how far (in km) it is from their input location, with two options:

- **Use this data** — proceed with the database soil profile.
- **Answer soil questions** — a short interactive quiz (one question at a time, immediately advancing on selection) to roughly determine the local soil profile from the farmer's own knowledge.

The final soil data, whichever source was chosen, is used as regional context for the analysis.

### 3. Initial Analysis Pipeline (4 Agents)

Once soil data is determined, the initial agentic pipeline runs. Four specialized agents execute sequentially, each building on the output of the previous one. Progress is shown in a live pipeline view with animated streaming text and auto-scrolling.

```
Agent 1: Visual Analysis
  Input  ← crop images / video keyframes
  Output → structured visual description (lesions, color gradients,
           distribution patterns, plant structure, soil condition)
  Model  → Vision-Language model (VL endpoint)

        ↓

Agent 2: Crop Diagnosis
  Input  ← visual description + farmer's statement + soil/region context
           + RAG disease database matches (if enabled)
  Output → differential diagnosis with confidence levels
  Model  → Reasoning model (Reasoning endpoint)

        ↓

Agent 3: Verification
  Input  ← visual description + diagnosis + farmer's statement + region context
  Output → stress-tested, verified assessment
  Model  → Reasoning model (Reasoning endpoint)

        ↓

Agent 4: Action Plan
  Input  ← verified assessment + farmer's region
  Output → prioritized action plan with immediate actions, monitoring steps,
           regular practices, things to avoid, and when to seek further help
  Model  → Reasoning model (Reasoning endpoint)
```

The final action plan is rendered as a formatted report with clear sections: condition summary, immediate actions (with priority, how-to, cost estimates, and local availability), 7-day monitoring checklist, regular practices, things to avoid, and escalation criteria.

### 4. Follow-Up Evidence Collection

If the initial analysis has low confidence or identifies areas that need more evidence, the action plan includes an **additional evidence suggestion** — for example, "We can give you better results if you show us the stem of the plant clearly." This is displayed prominently with a **Follow Up** button.

The user uploads the requested additional images or video (same input features and validation as the initial submission), then triggers the follow-up pipeline.

### 5. Follow-Up Analysis Pipeline (4 Agents)

The follow-up pipeline is a separate agentic workflow that builds on the initial results:

```
Agent F1: Context Summary (Follow-Up Prompt Generator)
  Input  ← established facts, prior gaps, initial action plan,
           any prior follow-up results, new evidence metadata
  Output → focused prompt distilling cumulative case context
  Model  → Reasoning model

        ↓

Agent F2: Re-Analysis (Follow-Up Diagnosis Adjuster)
  Input  ← generated prompt + new evidence images
           (images are first processed through the VL model)
  Output → adjusted diagnosis cross-referenced against prior findings
  Model  → VL model (for image analysis) + Reasoning model (for adjustment)

        ↓

Agent F3: Verification (Follow-Up Verification)
  Input  ← adjusted diagnosis + established facts + generated prompt
  Output → verified follow-up assessment
  Model  → Reasoning model

        ↓

Agent F4: Action Plan Update (Follow-Up Action Plan)
  Input  ← verified follow-up assessment + original action plan + remaining gaps
  Output → diff-style updates to the initial plan, highlighting contradictions,
           modifications, and kept recommendations
  Model  → Reasoning model
```

The follow-up report is displayed as a diff against the initial plan. Changes are tagged as **CONTRADICT** (strong warning, something from the initial plan should not be done), **MODIFY** (adjusted recommendation), or **KEEP** (confirmed as correct). A summary statement highlights what changed.

### 6. Cascading Follow-Ups

If the follow-up result identifies additional gaps (e.g., "We still need to see the roots"), the follow-up button reappears with new requested inputs. Each subsequent follow-up carries forward all established facts and prior results in a cumulative context, pruning redundant information to keep token usage efficient. This continues until confidence is sufficient or no further evidence is needed.

A **Start New Case** button (with confirmation dialog) is always available to reset the form and begin a fresh analysis.

---

## Technical Specification

### Model Architecture

CropWhisper uses two model endpoints served via [vLLM](https://github.com/vllm-project/vllm):

| Endpoint | Role | Used By |
|---|---|---|
| **VL Model** (`VL_MODEL_ENDPOINT_URL`) | Vision-Language model capable of processing images and producing structured visual descriptions. | Agent 1 (Visual Analysis), Agent F2 (Re-Analysis image processing) |
| **Reasoning Model** (`REASONING_MODEL_ENDPOINT_URL`) | Text reasoning model for diagnosis, verification, and action plan generation. | Agents 2, 3, 4, and all follow-up agents (F1-F4) |

Both endpoints are accessed via the OpenAI-compatible API that vLLM exposes. Model names are resolved dynamically at startup by querying each server's `/v1/models` endpoint. The app displays a status bar indicating whether all agents are connected.

### RAG (Retrieval-Augmented Generation)

When enabled (`RAG_ENABLED=true`), Agent 2 augments its diagnosis with similar confirmed disease cases from a Supabase-hosted document store. The system performs keyword-overlap matching against a cached table of crop/condition visual descriptions, returning the top 5 most relevant cases as additional context. This can be toggled off via environment variable for testing without external dependencies.

### Frameworks and Key Dependencies

| Component | Technology |
|---|---|
| UI | Gradio (Python) with custom CSS and embedded JavaScript |
| Agentic orchestration | LangGraph (StateGraph with sequential edges) |
| Model serving | vLLM with OpenAI-compatible API |
| Model client | `openai` Python SDK |
| Soil data | ISRIC SoilGrids REST API |
| Reverse geocoding | Nominatim (OpenStreetMap) |
| RAG document store | Supabase |
| Video keyframe extraction | OpenCV (`opencv-python-headless`) |
| Image validation | PIL / custom size enforcement (500 KB total limit) |

### Environment Variables

| Variable | Description |
|---|---|
| `VL_MODEL_ENDPOINT_URL` | Full URL to the Vision-Language vLLM endpoint (e.g., `http://localhost:8000/v1`) |
| `REASONING_MODEL_ENDPOINT_URL` | Full URL to the Reasoning vLLM endpoint (e.g., `http://localhost:8001/v1`) |
| `SUPABASE_URL` | Supabase project URL (for RAG and location data) |
| `SUPABASE_KEY` | Supabase service role key |
| `RAG_ENABLED` | `true` or `false` — toggle RAG document retrieval |

---

## Vision: Production Mobile App

This Gradio demo validates the core AI pipeline. In a production deployment, CropWhisper would be a **mobile application** with the following additional capabilities:

- **Real-time camera** — the farmer points their phone at the crop; the app captures frames automatically, no manual upload required.
- **Voice input** — the farmer describes the problem by speaking in their local language. Speech-to-text converts this into the description field. No typing or literacy needed.
- **Voice output** — the action plan is read aloud in the farmer's language, using text-to-speech with a warm, conversational tone. The farmer can listen while looking at their crop.
- **Offline-first** — lightweight on-device models handle initial triage. The full agentic pipeline runs when connectivity is available, with results cached locally.
- **GPS auto-capture** — location is detected silently in the background, no manual coordinate entry.
- **Low bandwidth** — images are compressed and sent efficiently. The app is designed for 2G/3G connections common in rural areas.
- **Multilingual** — the entire interaction (input and output) happens in the farmer's own language, with no need to switch or translate.

The core analytical pipeline (4-agent initial analysis + cascading follow-up) remains the same between the demo and the production app. The demo serves as the functional proof of concept for the AI reasoning layer.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Two vLLM model servers running (one VL model, one reasoning model)
- (Optional) Supabase project for RAG and location data

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy the environment template and fill in your endpoints:

```bash
cp .env.template .env
```

### Run

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

### Hugging Face Spaces

This repository is compatible with Gradio Spaces. Create a new Space with SDK = Gradio, push this repository, and add the environment variables as Space secrets.

---

## License

This project is currently unlicensed. See [LICENSE](LICENSE) for details once added.
