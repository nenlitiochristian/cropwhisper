# 🌾 CropWhisper

> **Point. Talk. Grow.**

---

## Overview

**CropWhisper** is a multimodal AI field assistant built for the 500 million smallholder farmers worldwide who have no access to agronomists, soil labs, or extension services — but do have a cheap Android phone and generations of farming knowledge.

The entire interaction is a single tap: point your camera at your crop, talk in your language, and CropWhisper listens, looks, and thinks — then talks back with a clear, actionable plan.

- No literacy required
- No agricultural training required
- No internet subscription required beyond a basic data connection

---

## Core User Experience

```
① Farmer opens CropWhisper
   └── One screen. One large button. "Hold & Talk."
       No menus. No settings. No onboarding wall.

② Farmer holds button, points camera, speaks freely
   └── "Majani yangu yanageuka manjano na kukauka
        pembezoni, hii ilitokea wiki mbili zilizopita"
       ("My leaves are turning yellow and drying at
        the edges, this started two weeks ago")
       Camera captures the plant/field simultaneously
       GPS captured silently in background

③ Processing (3-8 seconds)
   └── Subtle animation — the app is thinking, not frozen
       Four agents run in sequence with shared context

④ CropWhisper responds in the farmer's language
   └── Voice-first response, warm and plain
       Spoken in the same language the farmer used
       Screen shows a clean visual report simultaneously

⑤ Farmer can ask follow-up questions
   └── Still voice-only, still in their language
       Conversational, not form-based
```

---

## Getting Started

### Local

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Gradio app:

```bash
python app.py
```

3. Open the URL shown in your terminal (default: `http://127.0.0.1:7860`).

### Hugging Face Spaces (Gradio)

This repository is now compatible with **Gradio Spaces**.

1. Create a new Space with SDK = **Gradio**.
2. Push this repository to the Space.
3. Ensure the Space contains:
   - `app.py` (entrypoint)
   - `requirements.txt` (dependencies)
4. Add runtime secrets if needed (for model/API endpoints).

The app listens on `0.0.0.0` and uses the `PORT` environment variable when available.

---

## License

This project is currently unlicensed. See [LICENSE](LICENSE) for details once added.
