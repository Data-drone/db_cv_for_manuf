# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Detection Job
# MAGIC
# MAGIC Iterates image files in `input_path`, runs the VLM-proxy detector with the
# MAGIC supplied class list + instructions + endpoint, writes annotated JPEGs to
# MAGIC `<output_path>/annotated/`, a detections JSONL, and a `_summary.json`.
# MAGIC
# MAGIC Logs run progress to the Delta table `<catalog>.<schema>.batch_run_log`.
# MAGIC The CV Inspect app reads that table to render the run-history panel —
# MAGIC no polling.

# COMMAND ----------

# MAGIC %pip install --quiet --no-deps opencv-python-headless

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "cv_manufacturing")
dbutils.widgets.text("input_path", "")
dbutils.widgets.text("output_path", "")
dbutils.widgets.text("vlm_endpoint", "databricks-gemini-2-5-pro")
dbutils.widgets.text("classes", "[]")
dbutils.widgets.text("instructions", "")
dbutils.widgets.text("model_label", "")
dbutils.widgets.text("model_id", "")
dbutils.widgets.text("threshold", "0.0")
dbutils.widgets.text("max_files", "200")

# COMMAND ----------

import base64
import json
import os
import re
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
input_path = dbutils.widgets.get("input_path").rstrip("/")
output_path = dbutils.widgets.get("output_path").rstrip("/")
vlm_endpoint = dbutils.widgets.get("vlm_endpoint")
classes = json.loads(dbutils.widgets.get("classes") or "[]")
instructions = dbutils.widgets.get("instructions")
model_label = dbutils.widgets.get("model_label")
model_id = dbutils.widgets.get("model_id")
threshold = float(dbutils.widgets.get("threshold") or "0.0")
max_files = int(dbutils.widgets.get("max_files") or "200")

assert catalog, "catalog widget required"
assert input_path.startswith("/Volumes/"), f"input_path must be a UC Volume path: {input_path}"
assert output_path.startswith("/Volumes/"), f"output_path must be a UC Volume path: {output_path}"

LOG_TABLE = f"{catalog}.{schema}.batch_run_log"
run_id = str(uuid.uuid4())
started_at = datetime.now(timezone.utc)
triggered_by = spark.sql("SELECT current_user() AS u").collect()[0]["u"]

print(f"run_id={run_id}")
print(f"input_path={input_path}")
print(f"output_path={output_path}")
print(f"model={model_label} ({model_id}) via {vlm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure log table exists

# COMMAND ----------

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
        run_id STRING,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        status STRING,
        input_path STRING,
        output_path STRING,
        model_label STRING,
        model_id STRING,
        vlm_endpoint STRING,
        classes ARRAY<STRING>,
        threshold DOUBLE,
        image_count INT,
        detection_count INT,
        triggered_by STRING,
        error_message STRING
    ) USING DELTA
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## List inputs and write the initial RUNNING row

# COMMAND ----------

w = WorkspaceClient()
exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
entries = list(w.files.list_directory_contents(input_path))
files = sorted(
    [
        e for e in entries
        if e.name and e.name.lower().endswith(exts) and not getattr(e, "is_directory", False)
    ],
    key=lambda e: e.name,
)
files = files[:max_files]
image_count = len(files)
print(f"found {image_count} image(s) (capped at {max_files})")

log_schema = StructType([
    StructField("run_id", StringType(), False),
    StructField("started_at", TimestampType(), False),
    StructField("completed_at", TimestampType(), True),
    StructField("status", StringType(), False),
    StructField("input_path", StringType(), False),
    StructField("output_path", StringType(), False),
    StructField("model_label", StringType(), True),
    StructField("model_id", StringType(), True),
    StructField("vlm_endpoint", StringType(), True),
    StructField("classes", ArrayType(StringType()), True),
    StructField("threshold", DoubleType(), True),
    StructField("image_count", IntegerType(), True),
    StructField("detection_count", IntegerType(), True),
    StructField("triggered_by", StringType(), True),
    StructField("error_message", StringType(), True),
])

spark.createDataFrame(
    [Row(
        run_id=run_id,
        started_at=started_at,
        completed_at=None,
        status="RUNNING",
        input_path=input_path,
        output_path=output_path,
        model_label=model_label,
        model_id=model_id,
        vlm_endpoint=vlm_endpoint,
        classes=classes,
        threshold=threshold,
        image_count=image_count,
        detection_count=0,
        triggered_by=triggered_by,
        error_message=None,
    )],
    schema=log_schema,
).write.mode("append").saveAsTable(LOG_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference helpers
# MAGIC
# MAGIC Mirrors `app/inference.py` — duplicated so this notebook is self-contained.

# COMMAND ----------

DETECTOR_PROMPT = """You are an expert object detection model. Carefully examine the image and locate every instance of these classes only: {classes}.

{instructions}

For EACH detection, draw a tight bounding box around the actual pixel location. The bbox must visually enclose the object.

bbox format: [x1, y1, x2, y2] normalized [0.0, 1.0]:
- x1, x2 horizontal (0 = left edge, 1 = right edge)
- y1, y2 vertical (0 = TOP edge, 1 = BOTTOM edge)
- x1 < x2 and y1 < y2

Return ONLY a valid JSON array, no prose, no code fences. Each item:
{{"label": "<one of the classes above>", "bbox": [x1, y1, x2, y2], "confidence": <float 0-1>}}

If nothing is detected, return [].
"""


def parse_json_loose(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start < 0:
        return []
    end = text.rfind("]")
    if end > start:
        try:
            v = json.loads(text[start: end + 1])
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    items = []
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
            items.append(json.loads(text[i: j + 1]))
        except json.JSONDecodeError:
            pass
        i = j + 1
    return items


def normalize_bbox(item):
    raw = item.get("bbox") or item.get("box_2d") or item.get("box_d") or item.get("box")
    if raw is None or len(raw) != 4:
        raise ValueError("missing bbox")
    a, b, c, d = (float(v) for v in raw)
    is_gemini = ("box_2d" in item or "box_d" in item) or max(a, b, c, d) > 1.5
    if is_gemini:
        scale = 1000.0 if max(a, b, c, d) > 1.5 else 1.0
        x1, y1, x2, y2 = b / scale, a / scale, d / scale, c / scale
    else:
        x1, y1, x2, y2 = a, b, c, d
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return x1, y1, x2, y2


def call_vlm(prompt: str, img_b64: str, endpoint: str, max_tokens: int = 4096) -> str:
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if not host:
        host = w.config.host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"
    url = f"{host}/serving-endpoints/{endpoint}/invocations"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    headers.update(w.config.authenticate())
    r = requests.post(url, json=body, headers=headers, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_BOX_COLORS = [(108, 220, 0), (92, 107, 255), (191, 95, 139), (0, 165, 255), (210, 210, 80), (60, 180, 250)]


def draw_boxes(img_bgr, dets, classes_list):
    out = img_bgr.copy()
    h, w_ = out.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        p1 = (int(x1 * w_), int(y1 * h))
        p2 = (int(x2 * w_), int(y2 * h))
        idx = classes_list.index(d["label"]) if d["label"] in classes_list else abs(hash(d["label"])) % len(_BOX_COLORS)
        color = _BOX_COLORS[idx % len(_BOX_COLORS)]
        cv2.rectangle(out, p1, p2, color, 2, cv2.LINE_AA)
        cap = f"{d['label']} {d['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bg_y2 = p1[1]
        bg_y1 = max(0, p1[1] - th - 8)
        cv2.rectangle(out, (p1[0], bg_y1), (p1[0] + tw + 8, bg_y2), color, -1)
        cv2.putText(out, cap, (p1[0] + 4, bg_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process files

# COMMAND ----------

status = "SUCCESS"
error_message = None
total_detections = 0
detections_log = []

annotated_dir = f"{output_path}/annotated"
detections_path = f"{output_path}/detections.jsonl"
summary_path = f"{output_path}/_summary.json"

prompt = DETECTOR_PROMPT.format(
    classes=", ".join(f'"{c}"' for c in classes),
    instructions=instructions or "",
)

try:
    for entry in files:
        file_path = f"{input_path}/{entry.name}"
        try:
            dl = w.files.download(file_path)
            try:
                data = dl.contents.read()
            finally:
                try:
                    dl.contents.close()
                except Exception:
                    pass

            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                detections_log.append({"file": entry.name, "error": "could not decode image", "detections": []})
                continue

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            img_b64 = base64.b64encode(buf.tobytes()).decode()

            raw = call_vlm(prompt, img_b64, vlm_endpoint)
            items = parse_json_loose(raw)

            dets = []
            for it in items:
                try:
                    x1, y1, x2, y2 = normalize_bbox(it)
                    conf = float(it.get("confidence", 0.5))
                    if conf < threshold:
                        continue
                    dets.append({
                        "label": str(it["label"]),
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    })
                except (KeyError, ValueError, TypeError):
                    continue

            annotated = draw_boxes(img, dets, classes)
            ok2, buf2 = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            w.files.upload(f"{annotated_dir}/{entry.name}", contents=buf2.tobytes(), overwrite=True)

            detections_log.append({"file": entry.name, "detections": dets})
            total_detections += len(dets)
        except Exception as exc:
            detections_log.append({"file": entry.name, "error": str(exc)[:300], "detections": []})

    summary = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_label": model_label,
        "model_id": model_id,
        "vlm_endpoint": vlm_endpoint,
        "classes": classes,
        "image_count": image_count,
        "detection_count": total_detections,
        "input_path": input_path,
        "output_path": output_path,
    }
    w.files.upload(summary_path, contents=json.dumps(summary, indent=2).encode(), overwrite=True)

    detections_jsonl = "\n".join(json.dumps(d) for d in detections_log)
    w.files.upload(detections_path, contents=detections_jsonl.encode(), overwrite=True)

except Exception as exc:
    status = "FAILED"
    error_message = str(exc)[:500]
    print(f"FAILED: {error_message}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update the log row

# COMMAND ----------

completed_at = datetime.now(timezone.utc)


def _sql_str(v):
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


spark.sql(
    f"""
    UPDATE {LOG_TABLE}
    SET completed_at = TIMESTAMP '{completed_at.strftime('%Y-%m-%d %H:%M:%S')}',
        status = '{status}',
        detection_count = {total_detections},
        error_message = {_sql_str(error_message)}
    WHERE run_id = '{run_id}'
    """
)

print(f"run {run_id}: {status}, {image_count} files, {total_detections} detections")
dbutils.notebook.exit(json.dumps({
    "run_id": run_id,
    "status": status,
    "image_count": image_count,
    "detection_count": total_detections,
}))
