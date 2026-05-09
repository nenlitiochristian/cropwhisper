from .style import ICON
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
