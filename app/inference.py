"""Inference helpers for VLM and detector serving endpoints.

Two model families flow through here:
- vlm: single chat-completions call returning text
- detector: returns Detection[]; today via vlm_proxy backend (structured-output
  prompt against a VLM), tomorrow swappable to a real Databricks Model Serving
  endpoint by changing config.ModelEntry.backend without touching this code's
  call sites.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np
import requests
from databricks.sdk import WorkspaceClient

from config import ModelEntry


# ===== Workspace + auth =====

_workspace_client = None


def _get_workspace_client() -> WorkspaceClient:
    global _workspace_client
    if _workspace_client is None:
        _workspace_client = WorkspaceClient()
    return _workspace_client


def _auth_header() -> dict[str, str]:
    token = os.environ.get("DATABRICKS_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return dict(_get_workspace_client().config.authenticate())


def _endpoint_url(name: str) -> str:
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if not host:
        host = _get_workspace_client().config.host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"
    return f"{host}/serving-endpoints/{name}/invocations"


# ===== Image helpers =====

def _b64_jpeg(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _b64_jpeg_from_bytes(image_bytes: bytes) -> str:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return base64.b64encode(image_bytes).decode("ascii")
    return _b64_jpeg(img)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


# ===== VLM (chat completions) =====

def _vlm_chat(endpoint: str, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(
        _endpoint_url(endpoint),
        json=body,
        headers={"Content-Type": "application/json", **_auth_header()},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(data)


def query_vlm(model: ModelEntry, prompt: str, image_b64: str, max_tokens: int = 512) -> str:
    return _vlm_chat(
        endpoint=model.id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        max_tokens=max_tokens,
    )


# ===== Detector =====

@dataclass
class Detection:
    label: str
    bbox_norm: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized 0-1
    confidence: float


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    annotated_jpeg_b64: str = ""
    image_size: tuple[int, int] = (0, 0)  # (w, h)
    raw_response: str = ""  # debug: original backend response text


_DETECTOR_PROMPT = """You are an expert object detection model. Carefully examine the image and locate every instance of these classes only: {classes}.

{instructions}

For EACH detection, draw a tight bounding box around the actual pixel location of the object. The bbox must visually enclose the object — do not guess, do not produce placeholder coordinates, and do not cluster boxes in one area unless the objects are actually clustered there.

bbox format: [x1, y1, x2, y2] where each value is the normalized image coordinate in [0.0, 1.0]:
- x1, x2 are horizontal positions (0 = left edge, 1 = right edge)
- y1, y2 are vertical positions (0 = TOP edge, 1 = BOTTOM edge)
- x1 < x2 and y1 < y2

Return ONLY a valid JSON array, no prose, no code fences. Each item:
{{"label": "<one of the classes above>", "bbox": [x1, y1, x2, y2], "confidence": <float 0-1>}}

If nothing is detected, return []. Do not invent detections that are not visibly present.
"""


_BOX_COLORS_BGR = [
    (108, 220, 0),    # teal-green
    (92, 107, 255),   # coral
    (191, 95, 139),   # purple
    (0, 165, 255),    # orange
    (210, 210, 80),   # cyan
    (60, 180, 250),   # gold
]


def _color_for_label(label: str, classes: list[str]) -> tuple[int, int, int]:
    if label in classes:
        return _BOX_COLORS_BGR[classes.index(label) % len(_BOX_COLORS_BGR)]
    return _BOX_COLORS_BGR[hash(label) % len(_BOX_COLORS_BGR)]


def _parse_json_loose(text: str) -> list[dict]:
    """Parse a JSON array from VLM output, tolerating code fences and truncation."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start < 0:
        return []
    # First try the easy case: well-formed array.
    end = text.rfind("]")
    if end > start:
        try:
            v = json.loads(text[start : end + 1])
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    # Truncated case: collect complete top-level objects via brace counting.
    items: list[dict] = []
    i = start + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i >= n or text[i] != "{":
            break
        depth = 0
        j = i
        in_str = False
        esc = False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth != 0 or j >= n:
            break
        try:
            items.append(json.loads(text[i : j + 1]))
        except json.JSONDecodeError:
            pass
        i = j + 1
    return items


def _detector_via_vlm_proxy(model: ModelEntry, image_bgr: np.ndarray) -> tuple[list[Detection], str]:
    cfg = model.backend_config
    classes: list[str] = cfg["classes"]
    prompt = _DETECTOR_PROMPT.format(
        classes=", ".join(f'"{c}"' for c in classes),
        instructions=cfg.get("instructions", ""),
    )
    img_b64 = _b64_jpeg(image_bgr)
    print(f"[detector] vlm_proxy → {cfg['vlm_endpoint']} for {model.id} | classes={classes}", flush=True)
    raw = _vlm_chat(
        endpoint=cfg["vlm_endpoint"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    print(f"[detector] raw response ({len(raw)} chars): {raw[:500]}", flush=True)
    items = _parse_json_loose(raw)
    print(f"[detector] parsed {len(items)} items", flush=True)
    detections: list[Detection] = []
    for it in items:
        try:
            x1, y1, x2, y2 = _normalize_bbox(it)
            detections.append(
                Detection(
                    label=str(it["label"]),
                    bbox_norm=(x1, y1, x2, y2),
                    confidence=float(it.get("confidence", 0.5)),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return detections, raw


def _normalize_bbox(item: dict) -> tuple[float, float, float, float]:
    """Convert any of the bbox formats VLMs emit into normalized [x1,y1,x2,y2] in 0..1.

    Supported:
    - "bbox": [x1, y1, x2, y2] normalized 0..1 (our requested format, used by Llama/Claude)
    - "box_2d" / "box_d": [ymin, xmin, ymax, xmax] in 0..1000 (Gemini native format)
    """
    raw = item.get("bbox") or item.get("box_2d") or item.get("box_d") or item.get("box")
    if raw is None or len(raw) != 4:
        raise ValueError("missing bbox")
    a, b, c, d = (float(v) for v in raw)
    is_gemini = ("box_2d" in item or "box_d" in item) or max(a, b, c, d) > 1.5
    if is_gemini:
        # Gemini: [ymin, xmin, ymax, xmax] in 0..1000
        ymin, xmin, ymax, xmax = a, b, c, d
        scale = 1000.0 if max(a, b, c, d) > 1.5 else 1.0
        x1, y1, x2, y2 = xmin / scale, ymin / scale, xmax / scale, ymax / scale
    else:
        x1, y1, x2, y2 = a, b, c, d
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return x1, y1, x2, y2


def _detector_via_databricks_endpoint(model: ModelEntry, image_bgr: np.ndarray) -> tuple[list[Detection], str]:
    """Placeholder for a real Databricks Model Serving detector.

    To wire up: add backend_config={'endpoint_name': '<name>', 'parser': '<key>'}
    to config.ModelEntry, then implement the parser here. The output contract is
    identical to the vlm_proxy path so the UI doesn't change.
    """
    cfg = model.backend_config
    endpoint = cfg["endpoint_name"]
    img_b64 = _b64_jpeg(image_bgr)
    body = {"dataframe_records": [{"image_b64": img_b64}]}
    r = requests.post(
        _endpoint_url(endpoint),
        json=body,
        headers={"Content-Type": "application/json", **_auth_header()},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    raw = data.get("predictions") or data.get("outputs") or data
    rows = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    detections: list[Detection] = []
    if isinstance(rows, list):
        for it in rows:
            try:
                x1, y1, x2, y2 = (float(v) for v in it["bbox"])
                detections.append(
                    Detection(
                        label=str(it["label"]),
                        bbox_norm=(x1, y1, x2, y2),
                        confidence=float(it.get("confidence", 0.5)),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
    return detections, json.dumps(data)[:2000]


def _draw_boxes(image_bgr: np.ndarray, detections: list[Detection], classes: list[str]) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d.bbox_norm
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        color = _color_for_label(d.label, classes)
        cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)
        caption = f"{d.label} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bg_y2 = p1[1]
        bg_y1 = max(0, p1[1] - th - 8)
        cv2.rectangle(img, (p1[0], bg_y1), (p1[0] + tw + 8, bg_y2), color, -1)
        cv2.putText(img, caption, (p1[0] + 4, bg_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_detector(model: ModelEntry, image_bgr: np.ndarray, threshold: float = 0.0) -> DetectionResult:
    if model.backend == "vlm_proxy":
        dets, raw = _detector_via_vlm_proxy(model, image_bgr)
    elif model.backend == "databricks_endpoint":
        dets, raw = _detector_via_databricks_endpoint(model, image_bgr)
    else:
        raise NotImplementedError(f"Unknown backend: {model.backend}")
    print(f"[detector] {len(dets)} pre-threshold; threshold={threshold}", flush=True)
    dets = [d for d in dets if d.confidence >= threshold]
    classes = list(model.backend_config.get("classes", []))
    annotated = _draw_boxes(image_bgr, dets, classes)
    h, w = image_bgr.shape[:2]
    return DetectionResult(
        detections=dets,
        annotated_jpeg_b64=_b64_jpeg(annotated),
        image_size=(w, h),
        raw_response=raw,
    )


# ===== Public dispatch =====

@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    text: str = ""
    detection: DetectionResult | None = None


def run_image(model: ModelEntry, image_bytes: bytes, prompt: str, threshold: float = 0.0):
    """Returns str (vlm) or DetectionResult (detector)."""
    if model.family == "vlm":
        return query_vlm(model, prompt, _b64_jpeg_from_bytes(image_bytes))
    if model.family == "detector":
        return run_detector(model, _decode_image(image_bytes), threshold=threshold)
    raise NotImplementedError(f"Unknown family: {model.family}")


def iter_video_frames(video_bytes: bytes, frame_stride: int) -> Iterable[tuple[int, float, np.ndarray]]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % max(1, frame_stride) == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_video(
    model: ModelEntry,
    video_bytes: bytes,
    prompt: str,
    frame_stride: int = 30,
    max_frames: int = 20,
    threshold: float = 0.0,
) -> list[FrameResult]:
    results: list[FrameResult] = []
    for fi, ts, frame in iter_video_frames(video_bytes, frame_stride):
        if len(results) >= max_frames:
            break
        if model.family == "vlm":
            b64 = _b64_jpeg(frame)
            text = query_vlm(model, prompt, b64)
            results.append(FrameResult(frame_index=fi, timestamp_s=ts, text=text))
        elif model.family == "detector":
            det = run_detector(model, frame, threshold=threshold)
            results.append(FrameResult(frame_index=fi, timestamp_s=ts, detection=det))
        else:
            raise NotImplementedError(f"Unknown family: {model.family}")
    return results
