"""Registry of inference endpoints exposed in the UI.

Adding a new model is one entry. Detector entries pick a `backend` — either
`vlm_proxy` (a vision LLM acting as a detector via structured output, useful
when no native detector is deployed yet) or `databricks_endpoint` (a real
Databricks Model Serving endpoint that returns boxes natively). The UI and
inference layer treat both backends uniformly, so swapping later is a
config-only change.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


Family = Literal["vlm", "detector"]
Backend = Literal["vlm_proxy", "databricks_endpoint"]


@dataclass(frozen=True)
class ModelEntry:
    id: str
    label: str
    family: Family
    backend: Backend = "vlm_proxy"
    # backend_config:
    #   for vlm_proxy: {"vlm_endpoint": str, "classes": list[str], "instructions": str}
    #   for databricks_endpoint: {"endpoint_name": str, "parser": str}
    backend_config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# Foundation Model API endpoints used directly as VLMs.
_VLMS: list[ModelEntry] = [
    ModelEntry(id="databricks-claude-sonnet-4-6", label="Claude Sonnet 4.6 (vision)", family="vlm"),
    ModelEntry(id="databricks-claude-opus-4-7", label="Claude Opus 4.7 (vision)", family="vlm"),
    ModelEntry(id="databricks-gpt-5", label="GPT-5 (vision)", family="vlm"),
    ModelEntry(id="databricks-gemini-2-5-pro", label="Gemini 2.5 Pro (vision)", family="vlm"),
    ModelEntry(id="databricks-gemini-2-5-flash", label="Gemini 2.5 Flash (fast vision)", family="vlm"),
    ModelEntry(id="databricks-llama-4-maverick", label="Llama 4 Maverick (vision)", family="vlm"),
]


# Detectors. Today these are VLM-backed proxies — strong VLMs do reasonable
# bounding boxes when prompted with strict JSON output. Swap any entry's
# `backend` to "databricks_endpoint" and update `backend_config` once a real
# detector serving endpoint exists.
_DETECTORS: list[ModelEntry] = [
    ModelEntry(
        id="ppe-helmet",
        label="PPE / Helmet detector",
        family="detector",
        backend="vlm_proxy",
        backend_config={
            "vlm_endpoint": "databricks-gemini-2-5-pro",
            "classes": ["helmet", "person without helmet"],
            "instructions": (
                "Detect all people in the image and classify each as wearing a hard hat "
                "(label='helmet') or not (label='person without helmet')."
            ),
        },
        notes="Backed by Gemini 2.5 Pro (object grounding). Drop in a finetuned SHWD endpoint later.",
    ),
    ModelEntry(
        id="pcb-defects",
        label="PCB defect detector",
        family="detector",
        backend="vlm_proxy",
        backend_config={
            "vlm_endpoint": "databricks-gemini-2-5-pro",
            "classes": ["open", "short", "mousebite", "spur", "copper", "pin-hole"],
            "instructions": (
                "Inspect this printed circuit board for manufacturing defects. "
                "Categories: open (broken trace), short (unintended trace bridge), "
                "mousebite (notch in trace), spur (extra trace branch), copper (excess copper), "
                "pin-hole (small hole in trace). Mark each defect location with a tight bounding box."
            ),
        },
        notes="Backed by Gemini 2.5 Pro. Drop in a finetuned DeepPCB endpoint later.",
    ),
    ModelEntry(
        id="corrosion",
        label="Corrosion / asset wear detector",
        family="detector",
        backend="vlm_proxy",
        backend_config={
            "vlm_endpoint": "databricks-gemini-2-5-pro",
            "classes": ["corrosion", "rust", "structural damage"],
            "instructions": (
                "Identify visible corrosion, rust, or structural damage on infrastructure / "
                "industrial assets. Mark each affected region with a tight bounding box."
            ),
        },
        notes="Backed by Gemini 2.5 Pro. Drop in a finetuned corrosion endpoint later.",
    ),
]


MODELS: list[ModelEntry] = _VLMS + _DETECTORS


def by_id(model_id: str) -> ModelEntry:
    for m in MODELS:
        if m.id == model_id:
            return m
    raise KeyError(f"Unknown model id: {model_id}")


def for_family(family: Family) -> list[ModelEntry]:
    return [m for m in MODELS if m.family == family]
