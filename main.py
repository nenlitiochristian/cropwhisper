import os
import json
import re
import time
import math
import tempfile
import requests

from dotenv import load_dotenv
load_dotenv()

import shutil
from pathlib import Path

import gradio as gr

from agent import (
    AgentState, FollowUpState,
    construct_graph, construct_followup_graph, get_all_model_names,
)
from utils.image import validate_image_sizes
from supabase import create_client, Client

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
supabase: Client | None = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_URL and SUPABASE_KEY else None
)

app_graph = construct_graph()
followup_graph = construct_followup_graph()


# ── SVG Icons ────────────────────────────────────────────────────────────────

ICON = {
    "microscope": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>',
    "dna": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15c6.667-6 13.333 0 20-6"/><path d="M9 22c1.798-1.998 2.518-3.995 2.807-5.993"/><path d="M15 2c-1.798 1.998-2.518 3.995-2.807 5.993"/><path d="m17 6-2.5-2.5"/><path d="m14 8-1-1"/><path d="m7 18 2.5 2.5"/><path d="m3.5 14.5.5.5"/><path d="m20 9 .5.5"/><path d="m6.5 12.5 1 1"/><path d="m16.5 10.5 1 1"/><path d="m10 16 1.5 1.5"/></svg>',
    "check_circle": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "clipboard": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>',
    "leaf": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 19 2c1 2 2 4.5 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    "map_pin": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>',
    "sprout": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 20h10"/><path d="M10 20c5.5-2.5.8-6.4 3-10"/><path d="M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z"/><path d="M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z"/></svg>',
    "zap": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "eye": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>',
    "refresh": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>',
    "ban": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.9 4.9 14.2 14.2"/></svg>',
    "hospital": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6v4"/><path d="M14 14h-4"/><path d="M14 18h-4"/><path d="M14 8h-4"/><path d="M18 12h2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2h2"/><path d="M18 22V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v18"/></svg>',
    "alert": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "camera": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>',
    "plus_circle": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>',
    "arrow_right": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>',
    "x_circle": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
    "locate": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="2" x2="5" y1="12" y2="12"/><line x1="19" x2="22" y1="12" y2="12"/><line x1="12" x2="12" y1="2" y2="5"/><line x1="12" x2="12" y1="19" y2="22"/><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="3"/></svg>',
    "diff_plus": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2e7d32" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>',
    "diff_minus": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#c62828" stroke-width="2.5"><path d="M5 12h14"/></svg>',
    "diff_edit": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#e65100" stroke-width="2"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>',
}


# ── Neuromorphic CSS ─────────────────────────────────────────────────────────

NM_BG = "#e0e5ec"
NM_SHADOW_DARK = "#a3b1c6"
NM_SHADOW_LIGHT = "#ffffff"
NM_ACCENT = "#4a7a4a"
NM_ACCENT_LIGHT = "#6da86d"
NM_TEXT = "#2d3436"
NM_TEXT_MUTED = "#636e72"

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Force Gradio CSS variables to neuromorphic palette ── */
:root, .dark, body.dark, html.dark,
.gradio-container, .dark .gradio-container {
    --body-background-fill: #e0e5ec !important;
    --background-fill-primary: #e0e5ec !important;
    --background-fill-secondary: #e0e5ec !important;
    --panel-background-fill: #e0e5ec !important;
    --block-background-fill: #e0e5ec !important;
    --input-background-fill: #e0e5ec !important;
    --table-even-background-fill: #e0e5ec !important;
    --table-odd-background-fill: #d5dae1 !important;
    --neutral-50: #e0e5ec !important;
    --neutral-100: #d5dae1 !important;
    --neutral-200: #c8cdd4 !important;
    --neutral-300: #a3b1c6 !important;
    --body-text-color: #2d3436 !important;
    --block-title-text-color: #2d3436 !important;
    --block-label-text-color: #2d3436 !important;
    --input-text-color: #2d3436 !important;
    --body-text-color-subdued: #636e72 !important;
    --block-border-color: transparent !important;
    --border-color-primary: transparent !important;
    --border-color-accent: #4a7a4a !important;
    --color-accent: #4a7a4a !important;
    --color-accent-soft: #6da86d !important;
    --button-primary-background-fill: linear-gradient(135deg, #4a7a4a, #6da86d) !important;
    --button-primary-text-color: #fff !important;
    --button-secondary-background-fill: #e0e5ec !important;
    --button-secondary-text-color: #2d3436 !important;
    --shadow-drop: none !important;
    --block-shadow: none !important;
    --slider-color: #4a7a4a !important;
    --loader-color: #4a7a4a !important;
    --checkbox-background-color-selected: #4a7a4a !important;
    --checkbox-border-color-selected: #4a7a4a !important;
    --button-cancel-background-fill: #e0e5ec !important;
    --button-cancel-text-color: #c0392b !important;
}

/* ── Force neuromorphic base on EVERYTHING ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
.gradio-container, .dark,
.main, .wrap, .contain,
div.gradio-container {
    background: #e0e5ec !important;
    color: #2d3436 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

footer, .footer { display: none !important; }

/* Container: fill viewport width with comfortable padding */
.gradio-container {
    max-width: 98vw !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 20px 32px !important;
}

/* ── Kill all Gradio borders / backgrounds ── */
.block, .wrap, .contain,
.panel, .form,
div[class*="block"], div[class*="wrap"],
.gradio-group, .gradio-row, .gradio-column,
.gradio-accordion, .gradio-tabitem,
.tabitem, .tab-nav, .tabs,
div.svelte-1drgfwm,
div.border, div.rounded-lg {
    background: #e0e5ec !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Ensure side-by-side Row layout ── */
.gradio-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 20px !important;
}
.gradio-row > .gradio-column {
    flex: 1 1 0% !important;
    min-width: 0 !important;
}

/* ── Neuromorphic raised panels (columns, accordion) ── */
.gradio-column > .form,
.gradio-column > div > .form {
    background: #e0e5ec !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

/* ── Accordion ── */
.gradio-accordion {
    background: #e0e5ec !important;
    border-radius: 12px !important;
    box-shadow: 4px 4px 8px #a3b1c6, -4px -4px 8px #ffffff !important;
    overflow: hidden !important;
    margin-bottom: 12px !important;
}
.gradio-accordion > .label-wrap {
    background: #e0e5ec !important;
    border: none !important;
    padding: 10px 16px !important;
    color: #2d3436 !important;
}
.gradio-accordion > .label-wrap span,
.gradio-accordion > .label-wrap button {
    color: #2d3436 !important;
    font-weight: 600 !important;
}

/* ── Inputs: textbox, number ── */
textarea, input[type="text"], input[type="number"],
.gr-text-input textarea, .gr-input,
div[data-testid="textbox"] textarea,
div[data-testid="number-input"] input {
    background: #e0e5ec !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #ffffff !important;
    color: #2d3436 !important;
    padding: 10px 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    transition: box-shadow 0.2s ease !important;
}
textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    box-shadow: inset 4px 4px 8px #a3b1c6, inset -4px -4px 8px #ffffff !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder {
    color: #a3b1c6 !important;
}

/* Input containers */
.gr-box, .gr-panel, .gr-form,
div[data-testid="textbox"],
div[data-testid="number-input"],
div[data-testid="file"],
div[data-testid="video"] {
    background: #e0e5ec !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

/* ── Upload areas: consistent inset neuromorphic ── */
.upload-container, div.upload-container,
div[data-testid="image"] .upload-container,
div[data-testid="video"] .upload-container,
div[data-testid="gallery"] .upload-container {
    background: #e0e5ec !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #ffffff !important;
    color: #636e72 !important;
}
.upload-container span, .upload-container p {
    color: #636e72 !important;
}
.upload-container svg {
    color: #4a7a4a !important;
    opacity: 0.7;
}

/* ── Labels ── */
label, .label-wrap span, .gr-label,
span[data-testid="block-label"],
label span {
    color: #2d3436 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}
/* Label container background */
span[data-testid="block-label"] {
    background: #e0e5ec !important;
    border: none !important;
    box-shadow: 3px 3px 6px #a3b1c6, -3px -3px 6px #ffffff !important;
    border-radius: 8px !important;
    padding: 4px 10px !important;
    color: #4a7a4a !important;
}

/* ── Buttons ── */
button, .gr-button,
button[class*="primary"],
button[class*="secondary"] {
    background: #e0e5ec !important;
    color: #4a7a4a !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 4px 4px 8px #a3b1c6, -4px -4px 8px #ffffff !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    padding: 8px 16px !important;
}
button:hover, .gr-button:hover {
    box-shadow: 5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff !important;
}
button:active, .gr-button:active {
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #ffffff !important;
}

/* Primary button special */
button.primary, button.lg.primary,
button[variant="primary"],
#component-0 button.primary {
    background: linear-gradient(135deg, #4a7a4a, #6da86d) !important;
    color: #ffffff !important;
    box-shadow: 4px 4px 8px #a3b1c6, -4px -4px 8px #ffffff !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
}
button.primary:active {
    box-shadow: inset 3px 3px 6px #3a6a3a, inset -3px -3px 6px #7dba7d !important;
}

/* ── Hide soil option buttons (we use JS to click them from quiz HTML) ── */
#soil-opts-row { display: none !important; }

/* ── Output HTML container (the right panel) ── */
div#action-out {
    background: #e0e5ec !important;
    border: none !important;
    border-radius: 16px !important;
    box-shadow: 6px 6px 14px #a3b1c6, -6px -6px 14px #ffffff !important;
    color: #2d3436 !important;
    min-height: 400px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    position: relative !important;
    z-index: 1 !important;
    margin-bottom: 12px !important;
}
#action-out > div,
#action-out .prose,
#action-out > .wrap {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: #2d3436 !important;
}
#action-out * {
    max-width: 100% !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}
/* ── Kill ALL Gradio loading/pending flash ── */
.pending, .generating {
    animation: none !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
.pending *, .generating * {
    animation: none !important;
    opacity: 1 !important;
}
.eta-bar { display: none !important; }
/* Kill Gradio status-tracker overlay that blocks UI */
div[data-testid="status-tracker"] {
    display: none !important;
}
.svelte-1ed2p3z, .translucent,
div[class*="translucent"] {
    opacity: 1 !important;
    background: transparent !important;
}

/* ── Markdown text ── */
.prose, .markdown, .md,
.prose p, .prose li, .prose h1, .prose h2, .prose h3,
.markdown-text, .gr-prose,
div[data-testid="markdown"] {
    color: #2d3436 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e0e5ec; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #a3b1c6; border-radius: 3px; }

/* ── Generic SVG icon color ── */
button svg {
    color: #4a7a4a !important;
    stroke: #4a7a4a !important;
}

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
@keyframes typewriter-blink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0; }
}
@keyframes nmShimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes nmSpin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes nmFloat {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}

.pulse-indicator {
    display: inline-block;
    width: 8px; height: 8px;
    background: #6da86d;
    border-radius: 50%;
    animation: pulse 1.2s ease-in-out infinite;
    margin-left: 6px;
}
.typewriter-cursor {
    animation: typewriter-blink 0.7s infinite;
    color: #4a7a4a;
    font-weight: 300;
}

/* ── Check location button ── */
#cw-check-loc {
    align-self: flex-end !important;
    margin-bottom: 4px !important;
    min-width: 70px !important;
    flex: 0 0 auto !important;
}

/* ── Video component overrides ── */
#cw-video-input > div, #cw-image-input > div,
#cw-video-input .wrap, #cw-image-input .wrap {
    background: #e0e5ec !important;
    border: none !important;
    box-shadow: none !important;
}
#cw-video-input .upload-container,
#cw-image-input .upload-container {
    background: #e0e5ec !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #fff !important;
    min-height: 120px !important;
}
/* Restrict webcam/video frame to prevent layout shift */
#cw-video-input video,
#cw-video-input .video-container,
div[data-testid="video"] video {
    max-width: 100% !important;
    max-height: 300px !important;
    border-radius: 10px !important;
    border: none !important;
    object-fit: contain !important;
}
/* Top-right toolbar buttons (trim, download, etc) only */
#cw-video-input .icon-buttons-right button,
div[data-testid="video"] .wrap button[aria-label] {
    background: #e0e5ec !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 2px 2px 4px #a3b1c6, -2px -2px 4px #fff !important;
    padding: 4px 6px !important;
}
/* Record button: remove glow */
div[data-testid="video"] button.record,
div[data-testid="video"] button[class*="record"],
div[data-testid="video"] button[class*="Record"] {
    box-shadow: 3px 3px 6px #a3b1c6, -3px -3px 6px #fff !important;
    text-shadow: none !important;
    filter: none !important;
    animation: none !important;
}
div[data-testid="video"] button.record *,
div[data-testid="video"] button[class*="record"] * {
    text-shadow: none !important;
    filter: none !important;
    animation: none !important;
}
/* Camera switch button: remove glow */
div[data-testid="video"] button[class*="mirror"],
div[data-testid="video"] button[class*="switch"],
div[data-testid="video"] button[class*="camera"] {
    box-shadow: 3px 3px 6px #a3b1c6, -3px -3px 6px #fff !important;
    filter: none !important;
    text-shadow: none !important;
}
/* Video source switcher: z-index and spacing */
#cw-video-input .source-selection,
#cw-video-input .icon-buttons {
    position: relative !important;
    z-index: 10 !important;
    margin-top: 10px !important;
    padding: 6px 0 !important;
}
#cw-video-input .upload-container,
#cw-video-input .video-container,
#cw-video-input .player {
    margin-bottom: 8px !important;
}

/* ── Gallery component (image previews) ── */
.gallery, .gallery-item, .grid-wrap,
div[data-testid="gallery"] {
    background: #e0e5ec !important;
    border: none !important;
}
.gallery-item, .grid-wrap > div {
    background: #e0e5ec !important;
    border-radius: 10px !important;
    box-shadow: 3px 3px 6px #a3b1c6, -3px -3px 6px #fff !important;
    border: none !important;
    overflow: hidden !important;
}
.gallery-item:hover {
    box-shadow: 5px 5px 10px #a3b1c6, -5px -5px 10px #fff !important;
}
.gallery-item.selected {
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #fff !important;
    outline: 2px solid #4a7a4a !important;
    outline-offset: -2px !important;
}
.gallery-item img, .grid-wrap img {
    border-radius: 8px !important;
}
.gallery-item .caption {
    color: #636e72 !important;
    font-size: 11px !important;
    background: #e0e5ec !important;
}
.preview .image-container,
.preview img {
    border-radius: 12px !important;
    background: #e0e5ec !important;
}

/* ── Gallery upload area: consistent inset style ── */
div[data-testid="gallery"] .upload-container,
div[data-testid="gallery"] .wrap,
div[data-testid="gallery"] > div > div {
    background: #e0e5ec !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #ffffff !important;
}
div[data-testid="gallery"] .grid-wrap {
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #ffffff !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

/* ── Soil choice row ── */
#soil-choice-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    margin: 16px 0 20px !important;
    padding: 16px !important;
    gap: 20px !important;
    position: relative !important;
    z-index: 50 !important;
    background: #e0e5ec !important;
    border-radius: 14px !important;
    box-shadow: 4px 4px 8px #a3b1c6, -4px -4px 8px #ffffff !important;
}
/* Text on the left */
#soil-choice-row > div:first-child {
    flex: 0 0 auto !important;
    min-width: 120px !important;
}
/* Buttons column on the right */
#soil-choice-row button,
#soil-choice-row button.secondary,
#soil-choice-row button.lg {
    flex: 1 1 0% !important;
    padding: 14px 20px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #2d3436 !important;
    background: #e0e5ec !important;
    box-shadow: 4px 4px 8px #a3b1c6, -4px -4px 8px #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    transition: box-shadow 0.15s ease !important;
    text-align: center !important;
    white-space: nowrap !important;
    animation: none !important;
    opacity: 1 !important;
}
#soil-choice-row button span,
#soil-choice-row button div,
#soil-choice-row button * {
    animation: none !important;
    color: #2d3436 !important;
    opacity: 1 !important;
}
#soil-choice-row button:hover {
    box-shadow: 5px 5px 10px #a3b1c6, -5px -5px 10px #ffffff !important;
}
#soil-choice-row button:active {
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #ffffff !important;
}

/* ── Column sizing: fill viewport, stable proportions ── */
#cw-left-col {
    flex: 0 0 42% !important;
    min-width: 400px !important;
    max-width: 560px !important;
}
#cw-right-col {
    flex: 1 1 0% !important;
    min-width: 300px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

/* ── Consistent inset outline for ALL upload areas ── */
#cw-image-input, #cw-video-input {
    background: #e0e5ec !important;
    border-radius: 14px !important;
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #ffffff !important;
    border: none !important;
    padding: 4px !important;
}
#cw-image-input > div, #cw-video-input > div,
#cw-image-input .wrap, #cw-video-input .wrap,
#cw-image-input > div > div, #cw-video-input > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#cw-image-input .grid-container,
#cw-image-input .grid-wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
#cw-image-input .upload-container,
#cw-video-input .upload-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Source switcher: universal fix for ALL components ── */
.source-selection, .icon-buttons,
#cw-video-input .source-selection,
#cw-image-input .source-selection {
    display: flex !important;
    gap: 8px !important;
    justify-content: center !important;
    padding: 8px 0 !important;
}
.source-selection button, .icon-buttons button,
#cw-video-input .source-selection button,
#cw-image-input .source-selection button {
    width: auto !important;
    min-width: 80px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 6px 14px !important;
    border-radius: 10px !important;
    background: #e0e5ec !important;
    box-shadow: 3px 3px 6px #a3b1c6, -3px -3px 6px #fff !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #4a7a4a !important;
}
.source-selection button.selected,
.source-selection button[aria-pressed="true"],
.icon-buttons button.selected {
    box-shadow: inset 3px 3px 6px #a3b1c6, inset -3px -3px 6px #fff !important;
    color: #4a7a4a !important;
}
.source-selection button svg, .icon-buttons button svg {
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    min-height: 18px !important;
    color: #4a7a4a !important;
    stroke: #4a7a4a !important;
    flex-shrink: 0 !important;
}
"""


# ── Constants ────────────────────────────────────────────────────────────────

AGENT_ORDER = [
    "agent_1_visual", "agent_2_analyzer", "agent_3_verify", "agent_4_action"
]
AGENT_META = {
    "agent_1_visual":   (ICON["microscope"],    "Visual Analysis"),
    "agent_2_analyzer": (ICON["dna"],           "Crop Diagnosis"),
    "agent_3_verify":   (ICON["check_circle"],  "Verification"),
    "agent_4_action":   (ICON["clipboard"],     "Action Plan"),
}
AGENT_STATE_KEY = {
    "agent_1_visual":   "visual_description",
    "agent_2_analyzer": "diagnosis",
    "agent_3_verify":   "verified_assessment",
    "agent_4_action":   "action_plan",
}

FOLLOWUP_ORDER = [
    "followup_prompt_generator", "followup_diagnosis_adjuster",
    "followup_verification", "followup_action",
]
FOLLOWUP_META = {
    "followup_prompt_generator":    (ICON["clipboard"],    "Context Summary"),
    "followup_diagnosis_adjuster":  (ICON["dna"],          "Re-Analysis"),
    "followup_verification":        (ICON["check_circle"], "Verification"),
    "followup_action":              (ICON["clipboard"],    "Updated Plan"),
}
FOLLOWUP_STATE_KEY = {
    "followup_prompt_generator":    "generated_prompt",
    "followup_diagnosis_adjuster":  "adjusted_diagnosis",
    "followup_verification":        "followup_verification",
    "followup_action":              "followup_action",
}

SOIL_PROPS = {
    "phh2o":    ("pH (water)",            ""),
    "clay":     ("Clay",                  "%"),
    "sand":     ("Sand",                  "%"),
    "silt":     ("Silt",                  "%"),
    "soc":      ("Organic Carbon",        "g/kg"),
    "nitrogen": ("Nitrogen",              "g/kg"),
    "bdod":     ("Bulk Density",          "kg/dm\u00b3"),
    "cec":      ("Cation Exchange Cap.",  "mmol/kg"),
}

SOIL_QUESTIONS = [
    {
        "question": "When you squeeze a handful of wet soil, what happens?",
        "options": [
            ("Falls apart easily", "sandy"),
            ("Holds shape but crumbles when poked", "loamy"),
            ("Stays in a tight, sticky ball", "clayey"),
        ],
        "key": "texture",
    },
    {
        "question": "What color is your soil when dry?",
        "options": [
            ("Light or pale", "light"),
            ("Reddish-brown", "reddish"),
            ("Dark brown or black", "dark"),
            ("Gray", "gray"),
        ],
        "key": "color",
    },
    {
        "question": "Does water pool on the surface after rain?",
        "options": [
            ("Yes, often", "poor_drainage"),
            ("Sometimes", "moderate_drainage"),
            ("Drains quickly", "good_drainage"),
        ],
        "key": "drainage",
    },
    {
        "question": "Have you added any fertilizer or compost recently?",
        "options": [
            ("Yes, within a month", "recent_fertilizer"),
            ("Yes, but months ago", "old_fertilizer"),
            ("No, never", "no_fertilizer"),
        ],
        "key": "fertility",
    },
    {
        "question": "Does the soil smell sour or rotten when wet?",
        "options": [
            ("Yes", "acidic_smell"),
            ("No", "neutral_smell"),
            ("Not sure", "unsure_smell"),
        ],
        "key": "smell",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _find_nearest_location(lat, lon):
    if not supabase:
        return {"lat": lat, "lon": lon, "country": "Unknown",
                "region": "Unknown", "continent": "Unknown"}
    rows = supabase.table("location").select(
        "lat,lon,country,region,continent"
    ).execute().data
    if not rows:
        return {"lat": lat, "lon": lon}
    nearest = min(rows, key=lambda r: _haversine(lat, lon, r["lat"], r["lon"]))
    nearest["distance_km"] = round(
        _haversine(lat, lon, nearest["lat"], nearest["lon"]), 1
    )
    return nearest


def _get_soil_data(lat, lon):
    try:
        resp = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": lon, "lat": lat,
                "property": list(SOIL_PROPS.keys()),
                "depth": "0-5cm", "value": "mean",
            },
            timeout=20,
        )
        resp.raise_for_status()
        layers = resp.json().get("properties", {}).get("layers", [])
        result = {}
        for layer in layers:
            name = layer["name"]
            d_factor = layer.get("unit_measure", {}).get("d_factor", 1) or 1
            depths = layer.get("depths", [])
            if depths:
                raw = depths[0].get("values", {}).get("mean")
                result[name] = round(raw / d_factor, 2) if raw is not None else None
        return result
    except Exception as exc:
        return {"error": str(exc)}


def _derive_soil_from_answers(answers):
    """Convert questionnaire answers to approximate soil property values."""
    soil = {}
    texture = answers.get("texture", "loamy")
    if texture == "sandy":
        soil.update({"clay": 10, "sand": 75, "silt": 15, "bdod": 1.55})
    elif texture == "clayey":
        soil.update({"clay": 50, "sand": 15, "silt": 35, "bdod": 1.25})
    else:
        soil.update({"clay": 25, "sand": 40, "silt": 35, "bdod": 1.35})

    color = answers.get("color", "dark")
    if color == "dark":
        soil.update({"soc": 25.0, "nitrogen": 2.5})
    elif color == "reddish":
        soil.update({"soc": 15.0, "nitrogen": 1.5})
    elif color == "light":
        soil.update({"soc": 8.0, "nitrogen": 0.8})
    else:
        soil.update({"soc": 12.0, "nitrogen": 1.2})

    smell = answers.get("smell", "neutral_smell")
    if smell == "acidic_smell":
        soil["phh2o"] = 4.8
    elif smell == "neutral_smell":
        soil["phh2o"] = 6.5
    else:
        soil["phh2o"] = 5.8

    drainage = answers.get("drainage", "moderate_drainage")
    if drainage == "poor_drainage":
        soil["cec"] = 30.0
    elif drainage == "good_drainage":
        soil["cec"] = 10.0
    else:
        soil["cec"] = 20.0

    return soil


def _nm_wrap(inner_html):
    return (
        '<div style="height:82vh;overflow-y:auto;background:#e0e5ec;'
        'border-radius:16px;box-shadow:inset 4px 4px 8px #a3b1c6,inset -4px -4px 8px #fff;'
        'padding:6px;">'
        f'{inner_html}'
        '</div>'
    )


# ── Soil Card ────────────────────────────────────────────────────────────────

def _format_soil_card(location, soil, source="database"):
    country = location.get("country", "Unknown")
    region = location.get("region", "Unknown")
    continent = location.get("continent", "Unknown")
    dist = location.get("distance_km", "")

    if source == "database":
        dist_line = (
            f'<div style="font-size:12px;color:#e67e22;margin:4px 0 6px;'
            f'display:flex;align-items:center;gap:4px">'
            f'{ICON["alert"]}'
            f'Data from nearest grid point, <strong>{dist} km</strong> from your location'
            f'</div>'
        ) if dist else ""
        disclaimer = (
            '<div style="font-size:11px;color:#888;background:#d4d9e0;padding:8px 10px;'
            'border-radius:8px;margin-bottom:10px;line-height:1.5;'
            'box-shadow:inset 2px 2px 4px #a3b1c6,inset -2px -2px 4px #fff">'
            'This soil data is from the ISRIC SoilGrids global database and represents '
            'natural soil conditions. It may not reflect the actual soil in your farm or garden.'
            '</div>'
        )
    else:
        dist_line = ""
        disclaimer = (
            '<div style="font-size:11px;color:#4a7a4a;background:#d4d9e0;'
            'padding:8px 10px;border-radius:8px;margin-bottom:10px;line-height:1.5;'
            'box-shadow:inset 2px 2px 4px #a3b1c6,inset -2px -2px 4px #fff">'
            'Soil profile estimated from your answers.'
            '</div>'
        )

    rows = ""
    for key, (label, unit) in SOIL_PROPS.items():
        val = soil.get(key)
        display = (
            f"{val} {unit}".strip() if val is not None
            else '<span style="color:#bbb">N/A</span>'
        )
        rows += (
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:1px solid rgba(74,122,74,0.12)">'
            f'<span style="color:#636e72;font-size:13px">{label}</span>'
            f'<span style="font-weight:600;color:#4a7a4a;font-size:13px">'
            f'{display}</span>'
            f'</div>'
        )

    return (
        f'<div style="background:#e0e5ec;border-radius:14px;'
        f'box-shadow:6px 6px 12px #a3b1c6,-6px -6px 12px #fff;'
        f'padding:18px 20px;margin-bottom:18px">'
        f'  <div style="display:flex;align-items:center;gap:6px;font-size:11px;'
        f'font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        f'color:#636e72;margin-bottom:6px">'
        f'    {ICON["map_pin"]} Location &amp; Soil Context</div>'
        f'  <div style="font-size:15px;font-weight:600;color:#4a7a4a;'
        f'margin-bottom:2px">{country}</div>'
        f'  <div style="font-size:12px;color:#636e72;margin-bottom:4px">'
        f'    {region} &middot; {continent}</div>'
        f'  {dist_line}'
        f'  {disclaimer}'
        f'  <div style="display:flex;align-items:center;gap:6px;font-size:11px;'
        f'font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        f'color:#636e72;margin:10px 0 6px">'
        f'    {ICON["sprout"]} Soil (0-5 cm depth)</div>'
        f'  {rows}'
        f'</div>'
    )


# ── Soil Questionnaire HTML ──────────────────────────────────────────────────

def _soil_question_html(q_index, answers_so_far=None):
    if q_index >= len(SOIL_QUESTIONS):
        return ""
    q = SOIL_QUESTIONS[q_index]
    total = len(SOIL_QUESTIONS)
    progress_pct = int((q_index / total) * 100)

    options_html = ""
    for i, (label, _value) in enumerate(q["options"]):
        options_html += (
            f'<div style="background:#e0e5ec;border-radius:12px;'
            f'box-shadow:4px 4px 8px #a3b1c6,-4px -4px 8px #fff;'
            f'padding:14px 18px;margin-bottom:10px;cursor:pointer;'
            f'transition:all 0.15s ease;font-size:15px;font-weight:500;'
            f'color:#2d3436;display:flex;align-items:center;gap:10px"'
            f' onclick="(function(el){{'
            f'el.style.boxShadow=\'inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff\';'
            f'el.style.color=\'#4a7a4a\';'
            f'el.querySelector(\'.soil-opt-circle\').style.background=\'linear-gradient(135deg,#4a7a4a,#6da86d)\';'
            f'el.querySelector(\'.soil-opt-circle\').style.color=\'#fff\';'
            f'el.querySelector(\'.soil-opt-circle\').style.boxShadow=\'none\';'
            f'setTimeout(function(){{var r=document.querySelector(\'#soil-opts-row\');'
            f'if(r){{var btns=r.querySelectorAll(\'button\');if(btns[{i}])btns[{i}].click();}}}},250);'
            f'}})(this)"'
            f' onmouseover="this.style.boxShadow=\'6px 6px 14px #a3b1c6,-6px -6px 14px #fff\'"'
            f' onmouseout="this.style.boxShadow=\'4px 4px 8px #a3b1c6,-4px -4px 8px #fff\'">'
            f'<span class="soil-opt-circle" style="width:32px;height:32px;border-radius:50%;'
            f'background:#e0e5ec;box-shadow:inset 2px 2px 4px #a3b1c6,'
            f'inset -2px -2px 4px #fff;display:flex;align-items:center;'
            f'justify-content:center;font-size:13px;color:#4a7a4a;font-weight:700;'
            f'flex-shrink:0;transition:all 0.2s ease">'
            f'{chr(65 + i)}</span>'
            f'{label}'
            f'</div>'
        )

    return _nm_wrap(
        f'<div style="padding:24px;animation:fadeSlideIn 0.3s ease">'
        f'  <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">'
        f'    <span style="color:#4a7a4a">{ICON["sprout"]}</span>'
        f'    <div>'
        f'      <div style="font-weight:700;font-size:16px;color:#2d3436">'
        f'        Tell us about your soil</div>'
        f'      <div style="font-size:12px;color:#636e72">'
        f'        Question {q_index + 1} of {total}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div style="background:#d4d9e0;border-radius:6px;height:6px;margin-bottom:24px;'
        f'box-shadow:inset 2px 2px 4px #a3b1c6,inset -2px -2px 4px #fff">'
        f'    <div style="background:linear-gradient(90deg,#4a7a4a,#6da86d);'
        f'width:{progress_pct}%;height:100%;border-radius:6px;'
        f'transition:width 0.4s ease"></div>'
        f'  </div>'
        f'  <div style="font-size:18px;font-weight:600;color:#2d3436;margin-bottom:20px;'
        f'line-height:1.4">{q["question"]}</div>'
        f'  {options_html}'
        f'</div>'
    )


# ── Pipeline HTML Builder ────────────────────────────────────────────────────

def _extract_text(agent_id, data):
    key = AGENT_STATE_KEY[agent_id]
    payload = data.get(key, data)
    if isinstance(payload, dict) and payload.get("parse_error"):
        raw = payload.get("raw_output", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw
    return _format_agent_output(agent_id, payload)


def _format_agent_output(agent_id, data):
    """Format agent output as readable structured text instead of raw JSON."""
    if not isinstance(data, dict):
        return str(data)

    if agent_id == "agent_1_visual":
        return _format_visual_output(data)
    elif agent_id == "agent_2_analyzer":
        return _format_diagnosis_output(data)
    elif agent_id == "agent_3_verify":
        return _format_verification_output(data)
    return _dict_to_lines(data)


def _format_visual_output(d):
    lines = []
    ps = d.get("plant_structure", {})
    if isinstance(ps, dict):
        lines.append("PLANT STRUCTURE")
        for k in ["leaves", "stems", "roots", "fruit_and_flowers", "growing_tips"]:
            v = ps.get(k, "")
            if v:
                lines.append(f"  {k.replace('_', ' ').title()}: {v}")
    for key in ["symptom_distribution", "color_gradients", "soil_condition",
                 "surrounding_environment", "image_quality_flags"]:
        v = d.get(key, "")
        if v:
            lines.append(f"\n{key.replace('_', ' ').upper()}")
            lines.append(f"  {v}")
    return "\n".join(lines) if lines else _dict_to_lines(d)


def _format_diagnosis_output(d):
    lines = []
    dd = d.get("differential_diagnosis", [])
    if dd:
        lines.append("DIFFERENTIAL DIAGNOSIS")
        for i, dx in enumerate(dd, 1):
            cond = dx.get("condition", "Unknown")
            conf = dx.get("confidence", "?")
            lines.append(f"\n  {i}. {cond}  [{conf}]")
            r = dx.get("reasoning", {})
            if isinstance(r, dict):
                for rk, rv in r.items():
                    if rv:
                        lines.append(f"     {rk.replace('_', ' ').title()}: {rv}")
    pa = d.get("primary_assessment", "")
    if pa:
        lines.append(f"\nPRIMARY ASSESSMENT: {pa}")
    uf = d.get("uncertainty_flags", [])
    if uf:
        lines.append("\nUNCERTAINTY FLAGS")
        for f in uf:
            lines.append(f"  - {f}")
    return "\n".join(lines) if lines else _dict_to_lines(d)


def _format_verification_output(d):
    lines = []
    vr = d.get("verification_result", "")
    if vr:
        lines.append(f"RESULT: {vr}")
    ca = d.get("confidence_adjustments", [])
    if ca:
        lines.append("\nCONFIDENCE ADJUSTMENTS")
        for adj in ca:
            lines.append(
                f"  {adj.get('condition', '?')}: "
                f"{adj.get('original_confidence', '?')} -> "
                f"{adj.get('adjusted_confidence', '?')}"
            )
            if adj.get("note"):
                lines.append(f"    Note: {adj['note']}")
    fr = d.get("flags_raised", [])
    if fr:
        lines.append("\nFLAGS RAISED")
        for f in fr:
            lines.append(f"  - {f}")
    vpa = d.get("verified_primary_assessment", "")
    if vpa:
        lines.append(f"\nVERIFIED ASSESSMENT: {vpa}")
    return "\n".join(lines) if lines else _dict_to_lines(d)


def _dict_to_lines(obj, indent=0):
    pad = "  " * indent
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
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{obj}")
    return "\n".join(lines)


def _pipeline_html(completed, running=None, streaming=None, pre_step=None):
    step = len(completed)
    pct = int((step / 4) * 100)

    h = ['<div style="padding:20px 24px;font-family:inherit">']

    subtitle = pre_step if pre_step else f"Step {step} of 4 complete"
    h.append(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">'
        f'  <span style="color:#4a7a4a">{ICON["leaf"]}</span>'
        f'  <div>'
        f'    <div style="font-weight:700;font-size:16px;color:#2d3436">'
        f'      Analyzing your crop</div>'
        f'    <div style="font-size:12px;color:#636e72;margin-top:2px">'
        f'      {subtitle}</div>'
        f'  </div>'
        f'</div>'
    )

    # Neuromorphic progress bar
    h.append(
        f'<div style="background:#e0e5ec;border-radius:8px;height:8px;'
        f'margin-bottom:24px;box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff">'
        f'  <div style="background:linear-gradient(90deg,#4a7a4a,#6da86d);'
        f'width:{pct}%;height:100%;border-radius:8px;transition:width 0.6s ease"></div>'
        f'</div>'
    )

    # Step indicators
    h.append('<div style="display:flex;gap:8px;margin-bottom:20px">')
    for i, agent_id in enumerate(AGENT_ORDER):
        icon, label = AGENT_META[agent_id]
        done = agent_id in completed
        active = agent_id == running or (streaming and streaming[0] == agent_id)
        style_bg = (
            "background:linear-gradient(135deg,#4a7a4a,#6da86d);color:#fff;"
            "box-shadow:3px 3px 6px #a3b1c6,-3px -3px 6px #fff;"
            if done else
            ("background:#e0e5ec;color:#4a7a4a;"
             "box-shadow:3px 3px 6px #a3b1c6,-3px -3px 6px #fff;"
             if active else
             "background:#e0e5ec;color:#bbb;"
             "box-shadow:inset 2px 2px 4px #a3b1c6,inset -2px -2px 4px #fff;")
        )
        pulse = '<span class="pulse-indicator"></span>' if active else ""
        h.append(
            f'<div style="flex:1;text-align:center;padding:8px 4px;border-radius:10px;'
            f'font-size:11px;font-weight:600;transition:all 0.3s ease;{style_bg}">'
            f'{label}{pulse}'
            f'</div>'
        )
    h.append('</div>')

    # Agent output cards
    for agent_id in AGENT_ORDER:
        icon, label = AGENT_META[agent_id]
        is_done = agent_id in completed
        is_streaming = streaming and streaming[0] == agent_id
        is_running = agent_id == running

        if agent_id == "agent_4_action":
            if is_running:
                h.append(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:12px 16px;'
                    f'background:#e0e5ec;border-radius:12px;'
                    f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
                    f'color:#4a7a4a;font-weight:600">'
                    f'{icon} {label}<span class="pulse-indicator"></span>'
                    f'</div>'
                )
            continue

        if is_done or is_streaming:
            text = completed.get(agent_id, "") if is_done else streaming[1]
            cursor = (
                "" if is_done
                else '<span class="typewriter-cursor">|</span>'
            )
            check = (
                f'<span style="color:#27ae60">'
                f'{ICON["check_circle"]}</span>'
                if is_done else
                f'<span style="color:#4a7a4a">{icon}</span>'
            )
            anim = 'animation:fadeSlideIn 0.35s ease;' if is_done else ''
            h.append(
                f'<div style="margin-bottom:14px;{anim}">'
                f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'    {check}'
                f'    <span style="font-weight:600;color:#2d3436;'
                f'font-size:14px">{label}</span>'
                f'  </div>'
                f'  <div style="background:#e0e5ec;border-radius:10px;'
                f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
                f'padding:14px 16px;font-family:\'Inter\',monospace;font-size:12px;'
                f'line-height:1.7;color:#2d3436;'
                f'white-space:pre-wrap;max-height:180px;overflow-y:auto">'
                f'{text}{cursor}'
                f'  </div>'
                f'</div>'
            )
        elif is_running:
            h.append(
                f'<div style="margin-bottom:14px">'
                f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'    <span style="color:#4a7a4a">{icon}</span>'
                f'    <span style="font-weight:600;color:#2d3436;'
                f'font-size:14px">{label}</span>'
                f'    <span class="pulse-indicator"></span>'
                f'  </div>'
                f'  <div style="background:#e0e5ec;border-radius:10px;'
                f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
                f'padding:14px 16px;color:#bbb;font-size:12px">'
                f'    <span class="typewriter-cursor">|</span>'
                f'  </div>'
                f'</div>'
            )
        else:
            h.append(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'margin-bottom:14px;opacity:0.35;padding:8px 0">'
                f'  <span>{icon}</span>'
                f'  <span style="font-weight:500;color:#999;font-size:13px">{label}</span>'
                f'</div>'
            )

    h.append('</div>')
    return _nm_wrap("".join(h))


# ── Action Plan Formatter ────────────────────────────────────────────────────

def _format_action_plan(plan, soil_card=""):
    if plan.get("parse_error"):
        raw = plan.get("raw_output", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            return _nm_wrap(
                soil_card
                + f'<div style="padding:20px;white-space:pre-wrap;'
                f'color:#2d3436">{raw}</div>'
            )

    h = ['<div style="padding:20px;font-family:inherit;animation:fadeSlideIn 0.4s ease">']

    if soil_card:
        h.append(soil_card)

    condition = plan.get("condition", "")
    if condition:
        h.append(
            f'<div style="background:#e0e5ec;border-radius:14px;'
            f'box-shadow:6px 6px 12px #a3b1c6,-6px -6px 12px #fff;'
            f'padding:16px 20px;margin-bottom:18px">'
            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:1px;color:#636e72;margin-bottom:6px">'
            f'Crop Condition</div>'
            f'<div style="font-size:18px;font-weight:700;'
            f'color:#4a7a4a">{condition}</div>'
            f'</div>'
        )

    def section(icon_html, title):
        h.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:24px;'
            f'margin-bottom:12px;padding-bottom:8px;'
            f'border-bottom:2px solid rgba(74,122,74,0.15)">'
            f'<span style="color:#4a7a4a">{icon_html}</span>'
            f'<span style="font-size:16px;font-weight:700;'
            f'color:#2d3436">{title}</span>'
            f'</div>'
        )

    immediate = plan.get("immediate_actions", [])
    if immediate:
        section(ICON["zap"], "Immediate Actions")
        for item in immediate:
            h.append(
                f'<div style="background:#e0e5ec;border-radius:12px;'
                f'box-shadow:4px 4px 8px #a3b1c6,-4px -4px 8px #fff;'
                f'padding:14px 16px;margin-bottom:10px">'
                f'<div style="font-weight:600;color:#4a7a4a;'
                f'margin-bottom:6px">'
                f'{item.get("priority", "")}. {item.get("action", "")}</div>'
            )
            for lbl, key in [("How", "how"), ("Why now", "why_now"),
                             ("If unavailable", "if_unavailable"),
                             ("Cost", "cost_estimate"),
                             ("Local availability", "local_availability")]:
                if item.get(key):
                    h.append(
                        f'<div style="margin:3px 0;font-size:13px">'
                        f'<span style="font-weight:600;color:#636e72">'
                        f'{lbl}:</span> {item[key]}</div>'
                    )
            h.append('</div>')

    monitor = plan.get("monitor_next_7_days", [])
    if monitor:
        section(ICON["eye"], "Monitor Next 7 Days")
        for m in monitor:
            h.append(
                f'<div style="display:flex;align-items:flex-start;gap:8px;'
                f'margin:6px 0;font-size:13px;color:#2d3436">'
                f'<span style="color:#4a7a4a;margin-top:2px">'
                f'{ICON["arrow_right"]}</span>{m}</div>'
            )

    practices = plan.get("regular_practices", [])
    if practices:
        section(ICON["refresh"], "Regular Practices")
        for p in practices:
            h.append(
                f'<div style="background:#e0e5ec;border-radius:10px;'
                f'box-shadow:3px 3px 6px #a3b1c6,-3px -3px 6px #fff;'
                f'padding:12px 14px;margin-bottom:8px">'
                f'<div style="font-weight:600;font-size:13px;'
                f'color:#2d3436">'
                f'{p.get("frequency", "")}: {p.get("action", "")}</div>'
            )
            if p.get("why"):
                h.append(
                    f'<div style="font-size:12px;color:#636e72;'
                    f'margin-top:4px;font-style:italic">{p["why"]}</div>'
                )
            h.append('</div>')

    do_not = plan.get("do_not_do", [])
    if do_not:
        section(ICON["ban"], "Do Not Do")
        for d in do_not:
            h.append(
                f'<div style="display:flex;align-items:flex-start;gap:8px;'
                f'margin:6px 0;font-size:13px;color:#c0392b">'
                f'{ICON["x_circle"]} {d}</div>'
            )

    seek_help = plan.get("when_to_seek_further_help", "")
    if seek_help:
        section(ICON["hospital"], "When to Seek Further Help")
        h.append(
            f'<div style="background:#e0e5ec;border-radius:10px;padding:12px 14px;'
            f'border-left:4px solid #e67e22;'
            f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
            f'font-size:13px;color:#2d3436;line-height:1.5">{seek_help}</div>'
        )

    # Additional evidence suggestion
    suggestion = plan.get("additional_evidence_suggestion", {})
    if suggestion.get("can_improve_with_more_evidence"):
        sug_text = suggestion.get("suggestion_text", "")
        requested = suggestion.get("requested_images", [])
        h.append(
            f'<div style="margin-top:24px;background:#e0e5ec;'
            f'border-radius:14px;box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
            f'padding:18px 20px;border-left:3px solid #4a7a4a">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">'
            f'  <span style="color:#4a7a4a">{ICON["camera"]}</span>'
            f'  <span style="font-weight:700;font-size:14px;color:#4a7a4a">'
            f'    Want better results?</span>'
            f'</div>'
            f'<div style="font-size:14px;color:#2d3436;margin-bottom:12px;'
            f'line-height:1.5">{sug_text}</div>'
        )
        if requested:
            h.append('<div style="display:flex;flex-wrap:wrap;gap:8px">')
            for img_req in requested:
                h.append(
                    f'<div style="background:#e0e5ec;border-radius:8px;'
                    f'box-shadow:3px 3px 6px #a3b1c6,-3px -3px 6px #fff;'
                    f'padding:8px 12px;font-size:12px">'
                    f'<div style="font-weight:600;color:#4a7a4a">'
                    f'{img_req.get("description", "")}</div>'
                    f'<div style="color:#636e72;margin-top:2px">'
                    f'{img_req.get("reason", "")}</div>'
                    f'</div>'
                )
            h.append('</div>')
        h.append('</div>')

    h.append('</div>')
    return _nm_wrap("".join(h))


# ── Follow-Up Report Formatter ───────────────────────────────────────────────

def _format_followup_report(followup_action, round_num=1):
    if followup_action.get("parse_error"):
        raw = followup_action.get("raw_output", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        try:
            followup_action = json.loads(raw)
        except json.JSONDecodeError:
            return _nm_wrap(
                f'<div style="padding:20px;white-space:pre-wrap">{raw}</div>'
            )

    h = ['<div style="padding:20px;font-family:inherit;animation:fadeSlideIn 0.4s ease">']

    h.append(
        f'<div style="background:linear-gradient(135deg,#e0e5ec,#d5dae1);'
        f'border-radius:14px;box-shadow:6px 6px 12px #a3b1c6,-6px -6px 12px #fff;'
        f'padding:16px 20px;margin-bottom:18px">'
        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1px;color:#636e72;margin-bottom:4px">'
        f'Follow-Up Report (Round {round_num})</div>'
        f'<div style="font-size:15px;font-weight:600;color:#2d3436">'
        f'{followup_action.get("updated_condition", followup_action.get("summary_statement", ""))}'
        f'</div></div>'
    )

    summary = followup_action.get("summary_statement", "")
    if summary:
        h.append(
            f'<div style="background:#e0e5ec;border-radius:10px;padding:14px 16px;'
            f'border-left:4px solid #e67e22;margin-bottom:18px;'
            f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
            f'font-size:14px;font-weight:500;line-height:1.5;'
            f'color:#2d3436">{summary}</div>'
        )

    changes = followup_action.get("changes_to_initial_plan", [])
    change_type_style = {
        "CONTRADICT": (ICON["x_circle"], "#c0392b", "CHANGED"),
        "MODIFY": (ICON["diff_edit"], "#e65100", "MODIFIED"),
        "ADD": (ICON["diff_plus"], "#2e7d32", "NEW"),
        "REMOVE": (ICON["diff_minus"], "#c62828", "REMOVED"),
        "KEEP": (ICON["check_circle"], "#4a7a4a", "UNCHANGED"),
    }

    if changes:
        h.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;'
            f'padding-bottom:8px;border-bottom:2px solid rgba(74,122,74,0.15)">'
            f'<span style="font-size:16px;font-weight:700;'
            f'color:#2d3436">Changes to Action Plan</span></div>'
        )
        for change in changes:
            ct = change.get("change_type", "KEEP")
            icon_html, color, badge_text = change_type_style.get(
                ct, (ICON["check_circle"], "#888", ct)
            )
            is_contradict = ct == "CONTRADICT"
            border_style = f"border-left:4px solid {color};" if is_contradict else ""
            bg = "#e0e5ec"

            h.append(
                f'<div style="background:{bg};border-radius:12px;{border_style}'
                f'box-shadow:4px 4px 8px #a3b1c6,-4px -4px 8px #fff;'
                f'padding:14px 16px;margin-bottom:10px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'  {icon_html}'
                f'  <span style="background:{color};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-size:10px;font-weight:700;letter-spacing:0.5px">'
                f'{badge_text}</span>'
                f'  <span style="font-size:12px;color:#636e72">'
                f'{change.get("section", "")}</span>'
                f'</div>'
            )
            if change.get("initial_recommendation") and ct != "ADD":
                h.append(
                    f'<div style="font-size:12px;color:#999;text-decoration:line-through;'
                    f'margin-bottom:4px">{change["initial_recommendation"]}</div>'
                )
            if change.get("updated_recommendation"):
                h.append(
                    f'<div style="font-size:14px;font-weight:{"700" if is_contradict else "500"};'
                    f'color:{color if is_contradict else "#2d3436"};'
                    f'line-height:1.5">{change["updated_recommendation"]}</div>'
                )
            if change.get("reason"):
                h.append(
                    f'<div style="font-size:12px;color:#636e72;'
                    f'margin-top:4px;font-style:italic">{change["reason"]}</div>'
                )
            h.append('</div>')

    # Follow-up suggestion
    suggestion = followup_action.get("additional_evidence_suggestion", {})
    if suggestion.get("can_improve_with_more_evidence"):
        sug_text = suggestion.get("suggestion_text", "")
        requested = suggestion.get("requested_images", [])
        h.append(
            f'<div style="margin-top:20px;background:#e0e5ec;'
            f'border-radius:14px;box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
            f'padding:18px 20px;border-left:3px solid #4a7a4a">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            f'  <span style="color:#4a7a4a">{ICON["camera"]}</span>'
            f'  <span style="font-weight:700;font-size:14px;'
            f'color:#4a7a4a">Still need more info?</span>'
            f'</div>'
            f'<div style="font-size:13px;color:#2d3436;'
            f'margin-bottom:10px">{sug_text}</div>'
        )
        if requested:
            h.append('<div style="display:flex;flex-wrap:wrap;gap:8px">')
            for img_req in requested:
                h.append(
                    f'<div style="background:#e0e5ec;border-radius:8px;'
                    f'box-shadow:3px 3px 6px #a3b1c6,-3px -3px 6px #fff;'
                    f'padding:8px 12px;font-size:12px">'
                    f'<strong style="color:#4a7a4a">'
                    f'{img_req.get("description", "")}</strong>'
                    f'</div>'
                )
            h.append('</div>')
        h.append('</div>')

    h.append('</div>')
    return _nm_wrap("".join(h))




# ── Soil Preview (shown after Run Analysis fetches soil) ──────────────────────

def _soil_preview_html(location, soil):
    """Render a preview of fetched soil data with instructions to choose."""
    country = location.get("country", "Unknown")
    region = location.get("region", "Unknown")
    continent = location.get("continent", "Unknown")
    dist = location.get("distance_km", "")

    dist_line = ""
    if dist:
        dist_line = (
            f'<div style="display:flex;align-items:center;gap:6px;font-size:13px;'
            f'color:#e67e22;margin:6px 0 10px">'
            f'{ICON["alert"]} Data from nearest grid point, '
            f'<strong>{dist} km</strong> from your location</div>'
        )

    rows = ""
    for key, (label, unit) in SOIL_PROPS.items():
        val = soil.get(key)
        display = (
            f"{val} {unit}".strip() if val is not None
            else '<span style="color:#a3b1c6">N/A</span>'
        )
        rows += (
            f'<div style="display:flex;justify-content:space-between;padding:8px 0;'
            f'border-bottom:1px solid rgba(74,122,74,0.1)">'
            f'<span style="color:#636e72;font-size:13px">{label}</span>'
            f'<span style="font-weight:600;color:#4a7a4a;font-size:13px">'
            f'{display}</span></div>'
        )

    return _nm_wrap(
        f'<div style="padding:24px;animation:fadeSlideIn 0.3s ease">'
        f'  <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">'
        f'    <span style="color:#4a7a4a">{ICON["sprout"]}</span>'
        f'    <div>'
        f'      <div style="font-weight:700;font-size:17px;color:#2d3436">'
        f'        Soil Data Retrieved</div>'
        f'      <div style="font-size:12px;color:#636e72">'
        f'        Review and choose how to proceed</div>'
        f'    </div>'
        f'  </div>'
        f'  <div style="background:#e0e5ec;border-radius:14px;'
        f'box-shadow:4px 4px 8px #a3b1c6,-4px -4px 8px #fff;'
        f'padding:18px 20px;margin-bottom:16px">'
        f'    <div style="display:flex;align-items:center;gap:6px;font-size:11px;'
        f'font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        f'color:#636e72;margin-bottom:6px">'
        f'      {ICON["map_pin"]} Source Location</div>'
        f'    <div style="font-size:16px;font-weight:600;color:#4a7a4a;'
        f'margin-bottom:2px">{country}</div>'
        f'    <div style="font-size:12px;color:#636e72;margin-bottom:4px">'
        f'      {region} &middot; {continent}</div>'
        f'    {dist_line}'
        f'    <div style="background:#d4d9e0;border-radius:8px;padding:8px 12px;'
        f'margin:8px 0 12px;box-shadow:inset 2px 2px 4px #a3b1c6,inset -2px -2px 4px #fff;'
        f'font-size:11px;color:#636e72;line-height:1.5">'
        f'      This data is from the ISRIC SoilGrids global database and represents '
        f'natural soil conditions. It may not reflect your actual farm soil.</div>'
        f'    <div style="display:flex;align-items:center;gap:6px;font-size:11px;'
        f'font-weight:700;text-transform:uppercase;letter-spacing:1px;'
        f'color:#636e72;margin:12px 0 8px">'
        f'      {ICON["sprout"]} Soil Properties (0-5 cm)</div>'
        f'    {rows}'
        f'  </div>'
        f'</div>'
    )


# ── Main Report Generator ────────────────────────────────────────────────────

def _collect_image_paths(images, video):
    """Collect image paths from Gallery or File upload and optional video keyframes."""
    paths = []
    if images:
        for img in images:
            if isinstance(img, str):
                paths.append(img)
            elif isinstance(img, (tuple, list)):
                # Gallery returns (filepath, caption) tuples
                paths.append(str(img[0]))
            elif hasattr(img, "name"):
                paths.append(img.name)

    if video:
        video_path = video if isinstance(video, str) else video.name
        try:
            from utils.video import extract_keyframes
            keyframe_paths = extract_keyframes(video_path, num_keyframes=3)
            paths.extend(keyframe_paths)
        except Exception:
            pass

    return paths


def _preflight_loading_html(lat, lon):
    """Neuromorphic loading state shown while fetching soil data."""
    return _nm_wrap(
        '<div style="padding:30px;animation:fadeSlideIn 0.3s ease">'
        '  <div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;min-height:350px;gap:24px">'
        # Floating loader card
        '    <div style="background:#e0e5ec;border-radius:20px;'
        'box-shadow:6px 6px 14px #a3b1c6,-6px -6px 14px #fff;'
        'padding:32px 40px;text-align:center;animation:nmFloat 2s ease-in-out infinite">'
        # Spinner ring
        '      <div style="width:56px;height:56px;border-radius:50%;'
        'border:4px solid #d5dae1;border-top:4px solid #4a7a4a;'
        'animation:nmSpin 1s linear infinite;margin:0 auto 16px"></div>'
        f'      <div style="font-size:16px;font-weight:700;color:#2d3436;'
        f'margin-bottom:6px">{ICON["sprout"]} Checking Soil Data</div>'
        f'      <div style="font-size:13px;color:#636e72">'
        f'Querying SoilGrids for coordinates ({lat:.4f}, {lon:.4f})</div>'
        '    </div>'
        # Skeleton preview
        '    <div style="width:100%;max-width:440px">'
        '      <div style="font-size:11px;font-weight:600;color:#a3b1c6;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'
        'text-align:center">Loading soil properties...</div>'
        + ''.join(
            '<div style="display:flex;justify-content:space-between;'
            'padding:10px 0;border-bottom:1px solid rgba(163,177,198,0.15)">'
            '<div style="height:14px;border-radius:6px;'
            f'width:{w}%;background:linear-gradient(90deg,#d5dae1 25%,#e0e5ec 50%,#d5dae1 75%);'
            'background-size:800px 100%;animation:nmShimmer 1.5s infinite linear"></div>'
            '<div style="height:14px;width:50px;border-radius:6px;'
            'background:linear-gradient(90deg,#d5dae1 25%,#e0e5ec 50%,#d5dae1 75%);'
            'background-size:800px 100%;animation:nmShimmer 1.5s infinite linear"></div>'
            '</div>'
            for w in [30, 22, 25, 18, 35, 28, 32, 40]
        )
        + '    </div>'
        '  </div>'
        '</div>'
    )


def _run_preflight(text, latitude, longitude, images, video):
    """Phase 1: validate inputs, copy images, fetch soil data, show preview."""
    if not text or not text.strip():
        gr.Warning("Please provide a farmer transcript before running analysis.")
        yield gr.update(), gr.update(), None, gr.update(visible=False)
        return

    try:
        lat_f = float(latitude)
        lon_f = float(longitude)
    except (TypeError, ValueError):
        gr.Warning("Please provide valid latitude and longitude values.")
        yield gr.update(), gr.update(), None, gr.update(visible=False)
        return

    image_paths = _collect_image_paths(images, video)
    if not image_paths:
        gr.Warning("Please upload at least one image or video.")
        yield gr.update(), gr.update(), None, gr.update(visible=False)
        return

    valid, err_msg, _ = validate_image_sizes(image_paths)
    if not valid:
        gr.Warning(err_msg)
        yield gr.update(), gr.update(), None, gr.update(visible=False)
        return

    loading_html = _preflight_loading_html(lat_f, lon_f)
    yield (
        loading_html,
        gr.update(interactive=False, value="Fetching soil data..."),
        None,
        gr.update(visible=False),
    )

    upload_dir = Path(tempfile.mkdtemp(prefix="cropwhisper_"))
    temp_paths = []
    for p in image_paths:
        if p and os.path.exists(p):
            dest = str(upload_dir / Path(p).name)
            shutil.copyfile(p, dest)
            temp_paths.append(dest)

    location = _find_nearest_location(lat_f, lon_f)
    soil = _get_soil_data(location["lat"], location["lon"])

    preview_html = _soil_preview_html(location, soil)

    preflight_data = {
        "image_paths": temp_paths,
        "upload_dir": str(upload_dir),
        "transcript": text.strip(),
        "lat": lat_f,
        "lon": lon_f,
        "location": location,
        "soil_db": soil,
    }

    yield (
        preview_html,
        gr.update(interactive=False, value="Waiting for soil choice..."),
        preflight_data,
        gr.update(visible=True),
    )


def _start_soil_quiz_from_preflight(preflight_state, soil_state):
    """User chose to answer soil questions. Show quiz in the right panel."""
    if not preflight_state:
        raise gr.Error("Please click Run Analysis first.")

    soil_state = {"q_index": 0, "answers": {}, "mode": "quiz"}
    html = _soil_question_html(0)
    q = SOIL_QUESTIONS[0]
    btn_updates = []
    for i in range(5):
        if i < len(q["options"]):
            btn_updates.append(gr.update(visible=True, value=q["options"][i][0]))
        else:
            btn_updates.append(gr.update(visible=False))
    return [
        html,
        soil_state,
        gr.update(visible=False),
        gr.update(visible=True),
    ] + btn_updates


def _maybe_start_pipeline_after_quiz(soil_state, preflight_state):
    """Called via .then() after each quiz answer. Starts pipeline if quiz is done."""
    if not soil_state or soil_state.get("mode") != "done":
        yield gr.update(), gr.update(), gr.update(), gr.update()
        return

    if not preflight_state:
        yield gr.update(), gr.update(), gr.update(), gr.update()
        return

    soil = _derive_soil_from_answers(soil_state["answers"])
    yield from _run_pipeline(preflight_state, soil, "answers", soil_state)


def _run_pipeline(preflight_state, soil, soil_source, soil_state):
    """Phase 2: run the actual agent pipeline with finalized soil data."""
    location = preflight_state["location"]
    soil_card = _format_soil_card(location, soil, source=soil_source)

    initial_state: AgentState = {
        "image_paths":        preflight_state["image_paths"],
        "transcript":         preflight_state["transcript"],
        "region_context":     {"lat": preflight_state["lat"], "lon": preflight_state["lon"]},
        "visual_description": {},
        "diagnosis":          {},
        "verified_assessment": {},
        "action_plan":        {},
        "language":           "en",
    }
    initial_state["region_context"].update({
        "country":   location.get("country"),
        "region":    location.get("region"),
        "continent": location.get("continent"),
        "soil":      soil,
    })

    completed: dict[str, str] = {}
    btn_disabled = gr.update(interactive=False, value="Analyzing...")
    btn_enabled = gr.update(interactive=True, value="Run Analysis")

    yield _pipeline_html(completed, running="agent_1_visual"), btn_disabled, gr.update(), gr.update(visible=False)

    try:
        for chunk in app_graph.stream(initial_state):
            node_name = list(chunk.keys())[0]
            node_data = chunk[node_name]

            if node_name == "agent_4_action":
                action_plan = node_data.get("action_plan", {})
                report_html = _format_action_plan(action_plan, soil_card=soil_card)

                case_ctx = {
                    "initial_results": {
                        "visual_description": initial_state.get("visual_description", {}),
                        "diagnosis": initial_state.get("diagnosis", {}),
                        "verified_assessment": initial_state.get("verified_assessment", {}),
                        "action_plan": action_plan,
                    },
                    "followups": [],
                    "all_images_submitted": ["initial crop photo"],
                    "region_context": initial_state["region_context"],
                }

                can_followup = action_plan.get(
                    "additional_evidence_suggestion", {}
                ).get("can_improve_with_more_evidence", False)

                yield report_html, btn_enabled, case_ctx, gr.update(visible=can_followup)
                return

            full_text = _extract_text(node_name, node_data)
            lines = [l for l in full_text.splitlines() if l.strip()]
            revealed = []
            for line in lines:
                revealed.append(line)
                partial = "\n".join(revealed)
                yield _pipeline_html(completed, running=None, streaming=(node_name, partial)), btn_disabled, gr.update(), gr.update()
                time.sleep(0.05)

            completed[node_name] = full_text
            idx = AGENT_ORDER.index(node_name)
            next_running = AGENT_ORDER[idx + 1] if idx + 1 < len(AGENT_ORDER) else None
            yield _pipeline_html(completed, running=next_running), btn_disabled, gr.update(), gr.update()

    except Exception as exc:
        yield (
            _nm_wrap(
                f'<div style="padding:20px;color:#c0392b">'
                f'Pipeline failed: {exc}</div>'
            ),
            btn_enabled, gr.update(), gr.update()
        )
    finally:
        upload_dir = preflight_state.get("upload_dir")
        if upload_dir:
            shutil.rmtree(upload_dir, ignore_errors=True)


# ── Follow-Up Report Generator ───────────────────────────────────────────────

def _run_followup(followup_images, followup_video, case_context_state):
    if not case_context_state:
        raise gr.Error("No initial report found. Please run an analysis first.")

    image_paths = _collect_image_paths(followup_images, followup_video)
    if not image_paths:
        raise gr.Error("Please provide at least one follow-up image or video.")

    if image_paths:
        valid, err_msg, _ = validate_image_sizes(image_paths)
        if not valid:
            raise gr.Error(err_msg)

    temp_paths = []
    upload_dir = Path(tempfile.mkdtemp(prefix="cropwhisper_fu_"))
    for p in image_paths:
        if p and os.path.exists(p):
            dest = str(upload_dir / Path(p).name)
            shutil.copyfile(p, dest)
            temp_paths.append(dest)

    initial_results = case_context_state.get("initial_results", {})
    action_plan = initial_results.get("action_plan", {})
    diagnosis = initial_results.get("diagnosis", {})
    verified = initial_results.get("verified_assessment", {})

    followups = case_context_state.get("followups", [])
    round_num = len(followups) + 1

    suggested = action_plan.get("additional_evidence_suggestion", {})
    requested_imgs = suggested.get("requested_images", [])
    image_labels = [r.get("description", f"Image {i+1}") for i, r in enumerate(requested_imgs)]
    if len(image_labels) < len(temp_paths):
        image_labels.extend(
            [f"Additional image {i+1}" for i in range(len(temp_paths) - len(image_labels))]
        )

    all_submitted = case_context_state.get("all_images_submitted", [])
    all_submitted = all_submitted + image_labels

    if followups:
        last_fu = followups[-1]
        established = last_fu.get("prompt_summary", {}).get("established_facts", [])
        prior_gaps = last_fu.get("followup_action", {}).get("remaining_gaps", [])
        prior_fu_action = last_fu.get("followup_action", {})
    else:
        established = []
        pa = diagnosis.get("primary_assessment", "")
        if pa:
            established.append(pa)
        prior_gaps = diagnosis.get("uncertainty_flags", [])
        missing = []
        for dd in diagnosis.get("differential_diagnosis", []):
            m = dd.get("reasoning", {}).get("missing_that_would_confirm", "")
            if m:
                missing.append(m)
        prior_gaps.extend(missing)
        prior_gaps.extend(verified.get("flags_raised", []))
        prior_fu_action = None

    fu_state: FollowUpState = {
        "followup_image_paths": temp_paths,
        "images_provided_labels": image_labels,
        "all_images_submitted": all_submitted,
        "established_facts": established,
        "prior_gaps": prior_gaps,
        "initial_action_plan": action_plan,
        "prior_followup_action": prior_fu_action or {},
        "region_context": case_context_state.get("region_context", {}),
        "generated_prompt": {},
        "adjusted_diagnosis": {},
        "followup_verification": {},
        "followup_action": {},
    }

    btn_disabled = gr.update(interactive=False, value="Analyzing...")
    btn_enabled = gr.update(interactive=True, value="Submit Follow-Up")

    completed: dict[str, str] = {}

    yield _pipeline_html_followup(completed, running=None, pre_step="Preparing follow-up analysis..."), btn_disabled, gr.update(), gr.update()

    try:
        for chunk in followup_graph.stream(fu_state):
            node_name = list(chunk.keys())[0]
            node_data = chunk[node_name]

            if node_name == "followup_action":
                followup_action = node_data.get("followup_action", {})
                report_html = _format_followup_report(followup_action, round_num=round_num)

                new_followup = {
                    "round": round_num,
                    "images_provided": image_labels,
                    "prompt_summary": fu_state.get("generated_prompt", {}),
                    "adjusted_diagnosis": fu_state.get("adjusted_diagnosis", {}),
                    "verification": fu_state.get("followup_verification", {}),
                    "followup_action": followup_action,
                }
                updated_ctx = dict(case_context_state)
                updated_ctx["followups"] = followups + [new_followup]
                updated_ctx["all_images_submitted"] = all_submitted

                can_followup = followup_action.get(
                    "additional_evidence_suggestion", {}
                ).get("can_improve_with_more_evidence", False)

                yield report_html, btn_enabled, updated_ctx, gr.update(visible=can_followup)
                return

            state_key = FOLLOWUP_STATE_KEY.get(node_name, node_name)
            payload = node_data.get(state_key, node_data)
            if isinstance(payload, dict) and payload.get("parse_error"):
                raw = payload.get("raw_output", "")
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                full_text = raw
            else:
                full_text = _dict_to_lines(payload)

            lines = [l for l in full_text.splitlines() if l.strip()]
            revealed = []
            for line in lines:
                revealed.append(line)
                partial = "\n".join(revealed)
                yield _pipeline_html_followup(completed, running=None, streaming=(node_name, partial)), btn_disabled, gr.update(), gr.update()
                time.sleep(0.05)

            completed[node_name] = full_text
            idx = FOLLOWUP_ORDER.index(node_name) if node_name in FOLLOWUP_ORDER else -1
            next_running = FOLLOWUP_ORDER[idx + 1] if 0 <= idx < len(FOLLOWUP_ORDER) - 1 else None
            yield _pipeline_html_followup(completed, running=next_running), btn_disabled, gr.update(), gr.update()

    except Exception as exc:
        yield (
            _nm_wrap(
                f'<div style="padding:20px;color:#c0392b">'
                f'Follow-up pipeline failed: {exc}</div>'
            ),
            btn_enabled, gr.update(), gr.update()
        )
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


def _pipeline_html_followup(completed, running=None, streaming=None, pre_step=None):
    step = len(completed)
    pct = int((step / 4) * 100)

    h = ['<div style="padding:20px 24px;font-family:inherit">']
    subtitle = pre_step if pre_step else f"Step {step} of 4 complete"
    h.append(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">'
        f'  <span style="color:#4a7a4a">{ICON["refresh"]}</span>'
        f'  <div>'
        f'    <div style="font-weight:700;font-size:16px;color:#2d3436">'
        f'      Follow-Up Analysis</div>'
        f'    <div style="font-size:12px;color:#636e72;margin-top:2px">'
        f'      {subtitle}</div>'
        f'  </div></div>'
    )
    h.append(
        f'<div style="background:#e0e5ec;border-radius:8px;height:8px;'
        f'margin-bottom:24px;box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff">'
        f'  <div style="background:linear-gradient(90deg,#4a7a4a,#6da86d);'
        f'width:{pct}%;height:100%;border-radius:8px;transition:width 0.6s ease"></div>'
        f'</div>'
    )
    for agent_id in FOLLOWUP_ORDER:
        icon, label = FOLLOWUP_META[agent_id]
        is_done = agent_id in completed
        is_streaming = streaming and streaming[0] == agent_id
        is_running = agent_id == running

        if agent_id == "followup_action" and not is_done and not is_streaming:
            if is_running:
                h.append(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:12px 16px;'
                    f'background:#e0e5ec;border-radius:12px;'
                    f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
                    f'color:#4a7a4a;font-weight:600">'
                    f'{icon} {label}<span class="pulse-indicator"></span>'
                    f'</div>'
                )
            continue

        if is_done or is_streaming:
            text = completed.get(agent_id, "") if is_done else streaming[1]
            cursor = "" if is_done else '<span class="typewriter-cursor">|</span>'
            check = f'<span style="color:#27ae60">{ICON["check_circle"]}</span>' if is_done else f'<span style="color:#4a7a4a">{icon}</span>'
            h.append(
                f'<div style="margin-bottom:14px;{"animation:fadeSlideIn 0.35s ease;" if is_done else ""}">'
                f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'    {check}<span style="font-weight:600;color:#2d3436;font-size:14px">{label}</span></div>'
                f'  <div style="background:#e0e5ec;border-radius:10px;'
                f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
                f'padding:14px 16px;font-family:\'Inter\',monospace;font-size:12px;'
                f'line-height:1.7;color:#2d3436;white-space:pre-wrap;max-height:180px;overflow-y:auto">'
                f'{text}{cursor}</div></div>'
            )
        elif is_running:
            h.append(
                f'<div style="margin-bottom:14px">'
                f'  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'    <span style="color:#4a7a4a">{icon}</span>'
                f'    <span style="font-weight:600;color:#2d3436;font-size:14px">{label}</span>'
                f'    <span class="pulse-indicator"></span></div>'
                f'  <div style="background:#e0e5ec;border-radius:10px;'
                f'box-shadow:inset 3px 3px 6px #a3b1c6,inset -3px -3px 6px #fff;'
                f'padding:14px 16px;color:#bbb;font-size:12px">'
                f'    <span class="typewriter-cursor">|</span></div></div>'
            )
        else:
            h.append(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;opacity:0.35;padding:8px 0">'
                f'  <span>{icon}</span><span style="font-weight:500;color:#999;font-size:13px">{label}</span></div>'
            )

    h.append('</div>')
    return _nm_wrap("".join(h))


# ── New Case Reset ───────────────────────────────────────────────────────────

def _reset_case():
    placeholder = _nm_wrap(
        '<div style="display:flex;align-items:center;justify-content:center;'
        'height:100%;min-height:400px;color:#a3b1c6;font-size:15px;text-align:center;'
        'padding:40px"><div>'
        f'<div style="margin-bottom:16px;color:#4a7a4a;opacity:0.4;'
        f'transform:scale(1.5)">{ICON["leaf"]}</div>'
        '<div style="color:#636e72;font-weight:500;line-height:1.6">'
        'Upload an image and run analysis<br>to see results here</div>'
        '</div></div>'
    )
    return (
        gr.update(value=[]),         # images (Gallery)
        gr.update(value=None),      # video
        gr.update(value=""),        # transcript
        gr.update(value="-1.2921"), # lat
        gr.update(value="36.8219"), # lon
        placeholder,                # output html
        None,                       # case_context state
        gr.update(visible=False),   # followup section
        {"q_index": -1, "answers": {}, "mode": "database"},  # soil state
        None,                       # preflight state
        gr.update(visible=False),   # soil choice row
    )


# ── Model Info ───────────────────────────────────────────────────────────────

def _refresh_models():
    names = get_all_model_names()
    return "\n".join(f"- **{role}**: `{model}`" for role, model in names.items())


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="CropWhisper",
    css=CUSTOM_CSS,
) as demo:

    case_context = gr.State(value=None)
    soil_state = gr.State(value={"q_index": -1, "answers": {}, "mode": "database"})
    preflight_state = gr.State(value=None)

    # Header
    gr.HTML(
        f'<div style="display:flex;align-items:center;gap:16px;padding:16px 8px 20px;margin-bottom:4px">'
        f'  <div style="width:48px;height:48px;border-radius:14px;'
        f'background:linear-gradient(135deg,#4a7a4a,#6da86d);'
        f'box-shadow:4px 4px 8px #a3b1c6,-4px -4px 8px #fff;'
        f'display:flex;align-items:center;justify-content:center;color:#fff">'
        f'    {ICON["leaf"]}</div>'
        f'  <div>'
        f'    <div style="font-size:26px;font-weight:700;color:#2d3436;'
        f'letter-spacing:-0.5px">Crop<span style="color:#4a7a4a">Whisper</span></div>'
        f'    <div style="font-size:12px;color:#636e72;font-weight:500;'
        f'letter-spacing:0.5px">Point. Analyze. Grow.</div>'
        f'  </div>'
        f'</div>'
    )

    # Fix broken Gradio tooltips + map message listener
    gr.HTML("""<script>
    (function fixTooltips() {
        function fix() {
            document.querySelectorAll('button[title*="common."]').forEach(function(b){
                var t = b.title;
                if (t.includes('upload')) b.title = 'Upload files';
                else if (t.includes('clear')) b.title = 'Clear';
                else if (t.includes('download')) b.title = 'Download';
                else b.title = t.replace('common.', '');
            });
            document.querySelectorAll('button[aria-label*="common."]').forEach(function(b){
                var t = b.getAttribute('aria-label');
                if (t.includes('upload')) b.setAttribute('aria-label', 'Upload files');
                else if (t.includes('clear')) b.setAttribute('aria-label', 'Clear');
                else b.setAttribute('aria-label', t.replace('common.', ''));
            });
        }
        /* Label source-switcher buttons with text */
        document.querySelectorAll('.source-selection button, .icon-buttons button').forEach(function(btn){
            if (btn.dataset.cwLabeled) return;
            btn.dataset.cwLabeled = '1';
            var label = btn.getAttribute('aria-label') || btn.title || '';
            if (!label) {
                var svg = btn.querySelector('svg');
                if (svg) {
                    var path = svg.innerHTML || '';
                    if (path.includes('upload') || path.includes('M21 15v4')) label = 'Upload';
                    else if (path.includes('camera') || path.includes('M23 19')) label = 'Webcam';
                }
            }
            if (label && !btn.querySelector('.cw-src-label')) {
                var txt = label.toLowerCase();
                var display = 'Upload';
                if (txt.includes('webcam') || txt.includes('camera') || txt.includes('record')) display = 'Webcam';
                else if (txt.includes('upload') || txt.includes('file')) display = 'Upload';
                else if (txt.includes('clipboard') || txt.includes('paste')) display = 'Paste';
                var sp = document.createElement('span');
                sp.className = 'cw-src-label';
                sp.style.cssText = 'font-size:12px;font-weight:600;color:#4a7a4a;white-space:nowrap';
                sp.textContent = display;
                btn.appendChild(sp);
            }
        });

        fix();
        setInterval(fix, 2000);
    })();

    </script>""")

    with gr.Accordion("Model Info", open=False):
        model_info = gr.Markdown("Click Refresh to query vLLM servers.")
        refresh_btn = gr.Button("Refresh", size="sm")
        refresh_btn.click(fn=_refresh_models, inputs=[], outputs=[model_info])

    with gr.Row():
        # ── Left Column: Inputs ──
        with gr.Column(scale=1, elem_id="cw-left-col"):
            # Image input
            gr.HTML(
                f'<div style="display:flex;align-items:center;gap:8px;font-size:14px;'
                f'font-weight:700;color:#4a7a4a;margin-bottom:6px;padding:4px 0">'
                f'{ICON["camera"]} <span>Crop Images</span></div>'
            )
            image_input = gr.Gallery(
                label="Upload images (max 500KB total)",
                type="filepath",
                interactive=True,
                file_types=["image"],
                columns=4,
                height="auto",
                elem_id="cw-image-input",
            )
            gr.HTML(
                f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;'
                f'color:#636e72;margin:-4px 0 6px;padding:0 4px">'
                f'{ICON["camera"]} <span>Or capture/upload video (keyframes auto-extracted)</span></div>'
            )
            video_input = gr.Video(
                show_label=False,
                sources=["upload", "webcam"],
                elem_id="cw-video-input",
            )

            # Transcript
            gr.HTML(
                f'<div style="display:flex;align-items:center;gap:8px;font-size:14px;'
                f'font-weight:700;color:#4a7a4a;margin:12px 0 6px;padding:4px 0">'
                f'{ICON["clipboard"]} <span>Farmer Description</span></div>'
            )
            transcript_input = gr.Textbox(
                label="",
                show_label=False,
                lines=4,
                placeholder="Describe what the farmer said about the crop condition...",
            )

            # Location
            gr.HTML(
                f'<div style="display:flex;align-items:center;gap:8px;font-size:14px;'
                f'font-weight:700;color:#4a7a4a;margin:12px 0 6px;padding:4px 0">'
                f'{ICON["map_pin"]} <span>Location</span></div>'
            )
            with gr.Row():
                lat_input = gr.Textbox(
                    label="Latitude", value="-1.2921",
                    elem_id="cw-lat-hidden",
                )
                lon_input = gr.Textbox(
                    label="Longitude", value="36.8219",
                    elem_id="cw-lng-hidden",
                )
                check_loc_btn = gr.Button(
                    "Check", size="sm", variant="secondary",
                    elem_id="cw-check-loc",
                )
            location_display = gr.HTML(
                value='<div id="cw-loc-display" style="padding:6px 12px;font-size:13px;'
                      'color:#636e72;min-height:20px"></div>',
                elem_id="cw-loc-display-wrap",
            )
            detect_loc_btn = gr.Button(
                "Detect My Location", size="sm", variant="secondary",
            )
            detect_loc_btn.click(
                fn=None,
                js="""() => {
                    var display = document.getElementById('cw-loc-display');
                    if (!navigator.geolocation) {
                        if (display) display.innerHTML = '<span style=\"color:#c0392b\">Geolocation not supported by your browser</span>';
                        return;
                    }
                    if (display) display.innerHTML = '<span style=\"color:#636e72\">Requesting location access...</span>';
                    navigator.geolocation.getCurrentPosition(
                        function(pos) {
                            var lat = pos.coords.latitude, lng = pos.coords.longitude;
                            function setVal(sel, v) {
                                var el = document.querySelector(sel);
                                if (!el) return;
                                var proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement : HTMLInputElement;
                                var ns = Object.getOwnPropertyDescriptor(proto.prototype, 'value');
                                if (ns && ns.set) ns.set.call(el, v); else el.value = v;
                                el.dispatchEvent(new Event('input', {bubbles: true}));
                            }
                            setVal('#cw-lat-hidden textarea, #cw-lat-hidden input', lat.toFixed(6));
                            setVal('#cw-lng-hidden textarea, #cw-lng-hidden input', lng.toFixed(6));
                            if (display) display.innerHTML = '<span style=\"color:#636e72\">Resolving location...</span>';
                            fetch('https://nominatim.openstreetmap.org/reverse?format=json&lat='+lat+'&lon='+lng+'&zoom=5')
                            .then(function(r){return r.json()})
                            .then(function(d){
                                var country = (d.address && d.address.country) || '';
                                var code = (d.address && d.address.country_code) || '';
                                var flag = code ? String.fromCodePoint(...[...code.toUpperCase()].map(c=>0x1F1E6+c.charCodeAt(0)-65)) : '';
                                if (display) display.innerHTML = '<span style=\"font-size:20px\">' + flag + '</span> '
                                    + '<strong style=\"color:#4a7a4a\">' + country + '</strong>'
                                    + ' <span style=\"color:#636e72\">(' + lat.toFixed(4) + ', ' + lng.toFixed(4) + ')</span>';
                            })
                            .catch(function(){
                                if (display) display.innerHTML = '<span style=\"color:#636e72\">' + lat.toFixed(4) + ', ' + lng.toFixed(4) + '</span>';
                            });
                        },
                        function(err) {
                            if (display) display.innerHTML = '<span style=\"color:#c0392b\">Location access denied. Please allow location in browser settings.</span>';
                        },
                        {enableHighAccuracy: true, timeout: 10000}
                    );
                }""",
            )
            check_loc_btn.click(
                fn=None,
                js="""(lat, lon) => {
                    var display = document.getElementById('cw-loc-display');
                    if (!lat || !lon) {
                        if (display) display.innerHTML = '<span style=\"color:#c0392b\">Please enter both latitude and longitude</span>';
                        return;
                    }
                    var la = parseFloat(lat), lo = parseFloat(lon);
                    if (isNaN(la) || isNaN(lo)) {
                        if (display) display.innerHTML = '<span style=\"color:#c0392b\">Invalid coordinates</span>';
                        return;
                    }
                    if (display) display.innerHTML = '<span style=\"color:#636e72\">Looking up location...</span>';
                    fetch('https://nominatim.openstreetmap.org/reverse?format=json&lat='+la+'&lon='+lo+'&zoom=5')
                    .then(function(r){return r.json()})
                    .then(function(d){
                        var country = (d.address && d.address.country) || 'Unknown';
                        var code = (d.address && d.address.country_code) || '';
                        var flag = code ? String.fromCodePoint(...[...code.toUpperCase()].map(c=>0x1F1E6+c.charCodeAt(0)-65)) : '';
                        if (display) display.innerHTML = '<span style=\"font-size:20px\">' + flag + '</span> '
                            + '<strong style=\"color:#4a7a4a\">' + country + '</strong>'
                            + ' <span style=\"color:#636e72\">(' + la.toFixed(4) + ', ' + lo.toFixed(4) + ')</span>';
                    })
                    .catch(function(){
                        if (display) display.innerHTML = '<span style=\"color:#636e72\">' + la.toFixed(4) + ', ' + lo.toFixed(4) + '</span>';
                    });
                }""",
                inputs=[lat_input, lon_input],
            )

            run_button = gr.Button("Run Analysis", variant="primary", size="lg")

        # ── Right Column: Output ──
        with gr.Column(scale=1, elem_id="cw-right-col"):
            action_out = gr.HTML(
                elem_id="action-out",
                value=_nm_wrap(
                    '<div style="display:flex;align-items:center;justify-content:center;'
                    'height:100%;min-height:400px;color:#a3b1c6;font-size:15px;text-align:center;'
                    'padding:40px">'
                    '<div>'
                    f'<div style="margin-bottom:16px;color:#4a7a4a;opacity:0.4;'
                    f'transform:scale(1.5)">{ICON["leaf"]}</div>'
                    '<div style="color:#636e72;font-weight:500;line-height:1.6">'
                    'Upload an image and run analysis<br>to see results here</div>'
                    '</div></div>'
                ),
            )

            # Soil choice buttons (shown after preflight fetches soil data)
            with gr.Row(visible=False, elem_id="soil-choice-row") as soil_choice_row:
                gr.HTML(
                    '<div class="soil-proceed-label">'
                    '<div style="font-size:15px;color:#2d3436;font-weight:600;margin-bottom:2px">'
                    'How would you like to proceed?</div>'
                    '<div style="font-size:12px;color:#636e72">Choose an option</div>'
                    '</div>'
                )
                soil_use_db_btn = gr.Button("Use This Soil Data", variant="secondary", size="lg")
                soil_quiz_btn = gr.Button("Answer Soil Questions Instead", variant="secondary", size="lg")

            # Hidden quiz option buttons (triggered by JS from quiz HTML)
            with gr.Row(visible=False, elem_id="soil-opts-row") as soil_answer_row:
                soil_opt_btns = []
                for i in range(5):
                    b = gr.Button(f"Option {chr(65+i)}", size="sm")
                    soil_opt_btns.append(b)

            # Follow-up section
            with gr.Column(visible=False) as followup_section:
                gr.HTML(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-top:16px;'
                    f'padding:12px 16px;border-radius:12px;'
                    f'background:linear-gradient(135deg,#e8f0e8,#e0e5ec);'
                    f'box-shadow:inset 2px 2px 4px #a3b1c6,inset -2px -2px 4px #fff">'
                    f'<span style="color:#4a7a4a">{ICON["plus_circle"]}</span>'
                    f'<span style="font-weight:700;font-size:15px;'
                    f'color:#4a7a4a">Follow Up</span></div>'
                )
                followup_images = gr.Gallery(
                    label="Follow-up images (max 500KB total)",
                    type="filepath",
                    interactive=True,
                    file_types=["image"],
                    columns=4,
                    height="auto",
                )
                followup_video = gr.Video(
                    label="Or follow-up video",
                    sources=["upload", "webcam"],
                )
                with gr.Row():
                    followup_btn = gr.Button("Submit Follow-Up", variant="primary")
                    new_case_btn = gr.Button("Start New Case", variant="secondary", size="sm")

    # ── Step 1: Run Analysis → preflight (validate + fetch soil + show preview) ──
    run_button.click(
        fn=_run_preflight,
        inputs=[transcript_input, lat_input, lon_input, image_input, video_input],
        outputs=[action_out, run_button, preflight_state, soil_choice_row],
    )

    # ── Step 2a: User clicks "Use This Soil Data" → hide choice, run pipeline ──
    def _use_db_soil_wrapper(pf_state, soil_st):
        if not pf_state:
            raise gr.Error("Please click Run Analysis first.")
        soil_st = {"q_index": -1, "answers": {}, "mode": "database"}
        for out in _run_pipeline(pf_state, pf_state["soil_db"], "database", soil_st):
            yield out

    def _hide_soil_choice():
        return gr.update(visible=False)

    soil_use_db_btn.click(
        fn=_hide_soil_choice,
        inputs=[],
        outputs=[soil_choice_row],
    ).then(
        fn=_use_db_soil_wrapper,
        inputs=[preflight_state, soil_state],
        outputs=[action_out, run_button, case_context, followup_section],
    )

    # ── Step 2b: User clicks "Answer Soil Questions" → show quiz ──
    soil_quiz_btn.click(
        fn=_start_soil_quiz_from_preflight,
        inputs=[preflight_state, soil_state],
        outputs=[action_out, soil_state, soil_choice_row, soil_answer_row] + soil_opt_btns,
    )

    # ── Step 3: Soil quiz answer handling ──
    def _make_soil_answer_fn(choice_idx):
        def fn(soil_st, pf_state):
            q_idx = soil_st.get("q_index", 0)
            if q_idx >= len(SOIL_QUESTIONS):
                return [gr.update(), soil_st] + [gr.update() for _ in range(5)]

            q = SOIL_QUESTIONS[q_idx]
            if 0 <= choice_idx < len(q["options"]):
                _, value = q["options"][choice_idx]
                soil_st["answers"][q["key"]] = value

            next_idx = q_idx + 1
            soil_st["q_index"] = next_idx

            if next_idx >= len(SOIL_QUESTIONS):
                soil_st["mode"] = "done"
                soil = _derive_soil_from_answers(soil_st["answers"])
                summary_card = _format_soil_card(
                    {"country": "Your location", "region": "", "continent": ""},
                    soil, source="answers"
                )
                done_html = _nm_wrap(
                    f'<div style="padding:20px;animation:fadeSlideIn 0.3s ease">'
                    f'<div style="text-align:center;margin-bottom:16px">'
                    f'<span style="color:#27ae60">{ICON["check_circle"]}</span>'
                    f'<div style="font-weight:700;font-size:16px;margin-top:8px;'
                    f'color:#2d3436">Soil profile ready!</div>'
                    f'<div style="font-size:13px;color:#636e72;margin-top:4px">'
                    f'Starting analysis with your soil profile...</div>'
                    f'</div>{summary_card}</div>'
                )
                btn_updates = [gr.update(visible=False) for _ in range(5)]
                return [done_html, soil_st] + btn_updates

            html = _soil_question_html(next_idx, soil_st.get("answers"))
            next_q = SOIL_QUESTIONS[next_idx]
            btn_updates = []
            for i in range(5):
                if i < len(next_q["options"]):
                    btn_updates.append(gr.update(visible=True, value=next_q["options"][i][0]))
                else:
                    btn_updates.append(gr.update(visible=False))
            return [html, soil_st] + btn_updates
        return fn

    for i, btn in enumerate(soil_opt_btns):
        btn.click(
            fn=_make_soil_answer_fn(i),
            inputs=[soil_state, preflight_state],
            outputs=[action_out, soil_state] + soil_opt_btns,
        ).then(
            fn=_maybe_start_pipeline_after_quiz,
            inputs=[soil_state, preflight_state],
            outputs=[action_out, run_button, case_context, followup_section],
        )

    # Follow-up
    followup_btn.click(
        fn=_run_followup,
        inputs=[followup_images, followup_video, case_context],
        outputs=[action_out, followup_btn, case_context, followup_section],
    )

    # New case
    new_case_btn.click(
        fn=_reset_case,
        inputs=[],
        outputs=[
            image_input, video_input, transcript_input, lat_input, lon_input,
            action_out, case_context, followup_section, soil_state,
            preflight_state, soil_choice_row,
        ],
        js="() => { if(!confirm('This will clear all case data. You will not be able to follow up on this case. Continue?')) throw new Error('cancelled'); }",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
