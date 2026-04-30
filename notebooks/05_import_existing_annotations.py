# Databricks notebook source
# MAGIC %md
# MAGIC # Import Existing Annotations into CV Explorer
# MAGIC
# MAGIC Converts raw dataset annotations (VOC XML, DeepPCB TXT, Corrosion Parquet)
# MAGIC into COCO JSON, stages them into the `imports` UC volume, and POSTs to the
# MAGIC CV Explorer app's `/api/projects/{id}/import` endpoint.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - CV Explorer app deployed and running
# MAGIC - Raw datasets downloaded (notebook 00a)
# MAGIC - Images extracted to labeling volume (notebook 00)
# MAGIC
# MAGIC **Idempotent:** uses `on_existing_annotations=replace` so re-runs converge.

# COMMAND ----------

# MAGIC %pip install requests pillow
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "brian_gen_ai")
dbutils.widgets.text("schema", "cv_manufacturing")
dbutils.widgets.text("dataset", "shwd", "Dataset (shwd / deeppcb / corrosion)")
dbutils.widgets.text("app_url", "", "App URL (auto-discovered if blank)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
DATASET = dbutils.widgets.get("dataset").strip().lower()
APP_URL = dbutils.widgets.get("app_url").strip().rstrip("/")

RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
LABELING_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/labeling"
IMPORTS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/imports"

print(f"Dataset:  {DATASET}")
print(f"Raw:      {RAW_VOLUME}")
print(f"Labeling: {LABELING_VOLUME}")
print(f"Imports:  {IMPORTS_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Discover the app URL and get an auth token

# COMMAND ----------

import os
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
APP_NAME = "cv-explorer-dev"

app_info = w.api_client.do("GET", f"/api/2.0/apps/{APP_NAME}")
APP_CLIENT_ID = app_info.get("oauth2_app_client_id", "")

if not APP_URL:
    APP_URL = app_info.get("url", "").rstrip("/")
    if not APP_URL:
        raise RuntimeError(f"Could not get URL for app '{APP_NAME}'")
print(f"App URL: {APP_URL}")
print(f"App OAuth client ID: {APP_CLIENT_ID}")

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
notebook_token = ctx.apiToken().get()
workspace_url = w.config.host.rstrip("/")

token_resp = requests.post(
    f"{workspace_url}/oidc/v1/token",
    data={
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": notebook_token,
        "subject_token_type": "urn:databricks:params:oauth:token-type:personal-access-token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "scope": "all-apis",
        "audience": APP_CLIENT_ID,
    },
    timeout=10,
)
token_resp.raise_for_status()
APP_TOKEN = token_resp.json()["access_token"]
HEADERS = {"Authorization": f"Bearer {APP_TOKEN}"}
print("Token exchange succeeded")

health = requests.get(f"{APP_URL}/api/health", headers=HEADERS, timeout=10)
health.raise_for_status()
print(f"App health: {health.json()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Find or create the project

# COMMAND ----------

import json

DATASET_CONFIG = {
    "shwd": {
        "name": "SHWD Safety Helmets",
        "task_type": "detection",
        "class_list": ["hat", "person"],
        "source_volume": f"{LABELING_VOLUME}/shwd",
    },
    "deeppcb": {
        "name": "DeepPCB Defects",
        "task_type": "detection",
        "class_list": ["open", "short", "mousebite", "spur", "copper", "pin-hole"],
        "source_volume": f"{LABELING_VOLUME}/deeppcb",
    },
    "corrosion": {
        "name": "Corrosion Detection",
        "task_type": "detection",
        "class_list": ["corrosion"],
        "source_volume": f"{LABELING_VOLUME}/corrosion",
    },
}

if DATASET not in DATASET_CONFIG:
    raise ValueError(f"Unknown dataset '{DATASET}'. Choose from: {list(DATASET_CONFIG.keys())}")

cfg = DATASET_CONFIG[DATASET]

projects = requests.get(f"{APP_URL}/api/projects", headers=HEADERS, timeout=30).json()
existing = [p for p in projects if p["name"] == cfg["name"]]

if existing:
    project = existing[0]
    print(f"Found existing project: id={project['id']}, samples={project['sample_count']}")
else:
    print(f"Creating project: {cfg['name']}")
    resp = requests.post(
        f"{APP_URL}/api/projects",
        headers=HEADERS,
        json={
            "name": cfg["name"],
            "description": f"Imported from {DATASET} raw dataset",
            "task_type": cfg["task_type"],
            "class_list": cfg["class_list"],
            "source_volume": cfg["source_volume"],
        },
        timeout=120,
    )
    resp.raise_for_status()
    project = resp.json()
    print(f"Created project: id={project['id']}, samples={project['sample_count']}")

PROJECT_ID = project["id"]

if project["sample_count"] == 0:
    raise RuntimeError(
        f"Project {PROJECT_ID} has 0 samples — the app's volume scan found no images. "
        f"This usually means the app service principal's UC permissions (USE CATALOG, "
        f"READ_VOLUME) haven't propagated yet. Wait ~60s, delete the project, and re-run. "
        f"Source volume: {cfg['source_volume']}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Convert annotations to COCO JSON

# COMMAND ----------

import xml.etree.ElementTree as ET
from PIL import Image
import io

def convert_shwd_to_coco():
    """Convert SHWD Pascal VOC annotations to COCO JSON."""
    import zipfile

    src_zip = f"{RAW_VOLUME}/shwd_safety_helmet/shwd_voc2028.zip"
    tmp_dir = "/tmp/shwd_extract"

    if not os.path.exists(os.path.join(tmp_dir, "VOC2028", "Annotations")):
        with zipfile.ZipFile(src_zip, 'r') as zf:
            zf.extractall(tmp_dir)

    ann_dir = os.path.join(tmp_dir, "VOC2028", "Annotations")
    labeling_dir = f"{LABELING_VOLUME}/shwd"
    labeling_files = set(os.listdir(labeling_dir))

    cat_to_id = {name: i for i, name in enumerate(cfg["class_list"])}
    categories = [{"id": i, "name": name} for i, name in enumerate(cfg["class_list"])]

    images = []
    annotations = []
    ann_id = 1

    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])
    print(f"SHWD: processing {len(xml_files)} annotation files")

    for img_idx, xml_file in enumerate(xml_files):
        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()

        fname_elem = root.find("filename")
        filename = fname_elem.text.strip() if fname_elem is not None else xml_file.replace(".xml", ".jpg")

        if filename not in labeling_files:
            continue

        size = root.find("size")
        width = int(size.find("width").text) if size is not None else 0
        height = int(size.find("height").text) if size is not None else 0

        if width == 0 or height == 0:
            try:
                img = Image.open(os.path.join(labeling_dir, filename))
                width, height = img.size
            except Exception:
                continue

        image_id = img_idx + 1
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        for obj in root.findall(".//object"):
            name = obj.find("name").text.strip()
            if name not in cat_to_id:
                continue
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_to_id[name],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0,
            })
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def convert_deeppcb_to_coco():
    """Convert DeepPCB TXT annotations to COCO JSON."""
    src_dir = f"{RAW_VOLUME}/deep_pcb_defects"
    labeling_dir = f"{LABELING_VOLUME}/deeppcb"
    labeling_files = set(os.listdir(labeling_dir))

    defect_names = cfg["class_list"]
    cat_to_id = {name: i for i, name in enumerate(defect_names)}
    categories = [{"id": i, "name": name} for i, name in enumerate(defect_names)]

    images = []
    annotations = []
    ann_id = 1
    img_idx = 0

    for group_name in sorted(os.listdir(src_dir)):
        group_path = os.path.join(src_dir, group_name)
        if not os.path.isdir(group_path) or not group_name.startswith("group"):
            continue

        for fname in sorted(os.listdir(group_path)):
            if not fname.endswith("_test.jpg"):
                continue

            prefixed_img = f"{group_name}_{fname}"
            if prefixed_img not in labeling_files:
                continue

            txt_file = fname.replace("_test.jpg", ".txt")
            txt_path = os.path.join(group_path, txt_file)
            if not os.path.exists(txt_path):
                continue

            img_idx += 1
            image_id = img_idx
            images.append({
                "id": image_id,
                "file_name": prefixed_img,
                "width": 640,
                "height": 640,
            })

            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    x1, y1, x2, y2, dtype = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
                    defect_idx = dtype - 1
                    if defect_idx < 0 or defect_idx >= len(defect_names):
                        continue

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": defect_idx,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    })
                    ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def convert_corrosion_to_coco():
    """Convert corrosion parquet annotations to COCO JSON."""
    import pyarrow.parquet as pq

    src_parquet = f"{RAW_VOLUME}/corrosion_detection/train.parquet"
    labeling_dir = f"{LABELING_VOLUME}/corrosion"
    labeling_files = set(os.listdir(labeling_dir))

    categories = [{"id": 0, "name": "corrosion"}]
    images = []
    annotations = []
    ann_id = 1

    table = pq.read_table(src_parquet)

    objects_col = table.column("objects") if "objects" in table.column_names else None
    image_col = table.column("image") if "image" in table.column_names else None
    width_col = table.column("width") if "width" in table.column_names else None
    height_col = table.column("height") if "height" in table.column_names else None

    for idx in range(len(table)):
        fname = f"{idx:06d}.jpg"
        if fname not in labeling_files:
            continue

        w = int(width_col[idx].as_py()) if width_col is not None else 640
        h = int(height_col[idx].as_py()) if height_col is not None else 640

        if (w == 0 or h == 0) and image_col is not None:
            try:
                img_struct = image_col[idx].as_py()
                img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else img_struct
                img = Image.open(io.BytesIO(img_bytes))
                w, h = img.size
            except Exception:
                w, h = 640, 640

        image_id = idx + 1
        images.append({
            "id": image_id,
            "file_name": fname,
            "width": w,
            "height": h,
        })

        if objects_col is not None:
            obj = objects_col[idx].as_py()
            if obj and "bbox" in obj:
                bboxes = obj["bbox"]
                for bbox in bboxes:
                    if len(bbox) >= 4:
                        bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                        if bw <= 0 or bh <= 0:
                            continue
                        annotations.append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 0,
                            "bbox": [bx, by, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0,
                        })
                        ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


converters = {
    "shwd": convert_shwd_to_coco,
    "deeppcb": convert_deeppcb_to_coco,
    "corrosion": convert_corrosion_to_coco,
}

print(f"Converting {DATASET} annotations to COCO JSON...")
coco_data = converters[DATASET]()
print(f"  Images:      {len(coco_data['images'])}")
print(f"  Annotations: {len(coco_data['annotations'])}")
print(f"  Categories:  {[c['name'] for c in coco_data['categories']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Stage COCO JSON to imports volume

# COMMAND ----------

import_dir = f"{IMPORTS_VOLUME}/{DATASET}"
dbutils.fs.mkdirs(import_dir)

import_path = f"{import_dir}/labels.json"
with open(import_path, "w") as f:
    json.dump(coco_data, f)

size_mb = os.path.getsize(import_path) / (1024 * 1024)
print(f"Staged: {import_path} ({size_mb:.1f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. POST to the app's import endpoint

# COMMAND ----------

print(f"Importing into project {PROJECT_ID}...")
resp = requests.post(
    f"{APP_URL}/api/projects/{PROJECT_ID}/import",
    headers=HEADERS,
    json={
        "volume_path": import_path,
        "format": "coco",
        "on_missing_sample": "skip",
        "on_existing_annotations": "replace",
    },
    timeout=300,
)

if resp.status_code == 200:
    result = resp.json()
    print(f"Import succeeded:")
    print(f"  samples_touched:      {result.get('samples_touched', 0)}")
    print(f"  annotations_created:  {result.get('annotations_created', 0)}")
    print(f"  annotations_replaced: {result.get('annotations_replaced', 0)}")
    print(f"  samples_skipped:      {result.get('samples_skipped', 0)}")
    print(f"  samples_created:      {result.get('samples_created', 0)}")
    if result.get("samples_touched", 0) == 0 and result.get("samples_skipped", 0) > 0:
        raise RuntimeError(
            f"Import returned 200 but touched 0 samples ({result['samples_skipped']} skipped). "
            f"This means all COCO image filenames were skipped because the project has no "
            f"matching samples. Delete the project, wait for SP permissions to propagate, "
            f"then re-create and re-import."
        )
elif resp.status_code == 422:
    body = resp.json()
    print(f"Validation failed: {body.get('error_count', '?')} errors")
    for err in body.get("errors", [])[:10]:
        print(f"  row {err.get('row')}: {err.get('filename')} — {err.get('reason')}")
    raise RuntimeError(f"Import validation failed with {body.get('error_count')} errors")
else:
    body = resp.text[:3000]
    raise RuntimeError(f"Import failed: HTTP {resp.status_code}\n{body}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify in the app

# COMMAND ----------

proj = requests.get(f"{APP_URL}/api/projects/{PROJECT_ID}", headers=HEADERS, timeout=30).json()
print(f"Project: {proj['name']}")
print(f"  Total samples: {proj['sample_count']}")
print(f"  Labeled:       {proj['labeled_count']}")
print(f"  Completion:    {proj['labeled_count'] / max(proj['sample_count'], 1) * 100:.1f}%")
