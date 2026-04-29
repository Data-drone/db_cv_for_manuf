# Databricks notebook source
# MAGIC %md
# MAGIC # App Export → COCO Train/Test Splits
# MAGIC
# MAGIC Bridges the CV Explorer app's single-file COCO export into the
# MAGIC `train/_annotations.coco.json` + `test/_annotations.coco.json` layout
# MAGIC expected by `03_finetune_sam31_detection.py`.
# MAGIC
# MAGIC 1. Triggers an export via the app API
# MAGIC 2. Reads the exported COCO JSON
# MAGIC 3. Splits images deterministically (seed + ratio)
# MAGIC 4. Writes train/test COCO JSONs + copies images to the `coco_datasets` volume

# COMMAND ----------

# MAGIC %pip install requests
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "brian_gen_ai")
dbutils.widgets.text("schema", "cv_manufacturing")
dbutils.widgets.text("project_name", "SHWD Safety Helmets")
dbutils.widgets.text("app_url", "", "App URL (auto-discovered if blank)")
dbutils.widgets.text("split_ratio", "0.8", "Train fraction")
dbutils.widgets.text("seed", "42", "Random seed for split")
dbutils.widgets.text("output_name", "", "Output folder name (defaults to project name)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
PROJECT_NAME = dbutils.widgets.get("project_name").strip()
APP_URL = dbutils.widgets.get("app_url").strip().rstrip("/")
SPLIT_RATIO = float(dbutils.widgets.get("split_ratio"))
SEED = int(dbutils.widgets.get("seed"))
OUTPUT_NAME = dbutils.widgets.get("output_name").strip()

EXPORTS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/exports"
COCO_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/coco_datasets"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Connect to the app

# COMMAND ----------

import os
import json
import random
import shutil
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
APP_NAME = "cv-explorer-dev"

app_info = w.api_client.do("GET", f"/api/2.0/apps/{APP_NAME}")
APP_CLIENT_ID = app_info.get("oauth2_app_client_id", "")

if not APP_URL:
    APP_URL = app_info.get("url", "").rstrip("/")
print(f"App URL: {APP_URL}")

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
HEADERS = {"Authorization": f"Bearer {token_resp.json()['access_token']}"}

projects = requests.get(f"{APP_URL}/api/projects", headers=HEADERS, timeout=30).json()
project = next((p for p in projects if p["name"] == PROJECT_NAME), None)
if not project:
    available = [p["name"] for p in projects]
    raise ValueError(f"Project '{PROJECT_NAME}' not found. Available: {available}")

PROJECT_ID = project["id"]
print(f"Project: {project['name']} (id={PROJECT_ID}, samples={project['sample_count']}, labeled={project['labeled_count']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Trigger export

# COMMAND ----------

print(f"Exporting project {PROJECT_ID} to {EXPORTS_VOLUME}...")
resp = requests.post(
    f"{APP_URL}/api/projects/{PROJECT_ID}/export",
    headers=HEADERS,
    json={"export_volume": EXPORTS_VOLUME},
    timeout=600,
)
resp.raise_for_status()
export_result = resp.json()

export_path = export_result["export_path"]
print(f"Export complete:")
print(f"  Path:        {export_path}")
print(f"  Format:      {export_result['format']}")
print(f"  Images:      {export_result['images']}")
print(f"  Annotations: {export_result['annotations']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Read exported COCO JSON and split

# COMMAND ----------

ann_file = f"{export_path}/annotations.json"
with open(ann_file) as f:
    coco = json.load(f)

all_images = coco["images"]
all_anns = coco["annotations"]
categories = coco["categories"]

print(f"Loaded: {len(all_images)} images, {len(all_anns)} annotations, {len(categories)} categories")

random.seed(SEED)
image_ids = [img["id"] for img in all_images]
random.shuffle(image_ids)

split_idx = int(len(image_ids) * SPLIT_RATIO)
train_ids = set(image_ids[:split_idx])
test_ids = set(image_ids[split_idx:])

print(f"Split ({SPLIT_RATIO}/{1 - SPLIT_RATIO:.1f}, seed={SEED}): train={len(train_ids)}, test={len(test_ids)}")

train_images = [img for img in all_images if img["id"] in train_ids]
test_images = [img for img in all_images if img["id"] in test_ids]
train_anns = [ann for ann in all_anns if ann["image_id"] in train_ids]
test_anns = [ann for ann in all_anns if ann["image_id"] in test_ids]

print(f"  Train: {len(train_images)} images, {len(train_anns)} annotations")
print(f"  Test:  {len(test_images)} images, {len(test_anns)} annotations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write COCO splits + copy images

# COMMAND ----------

safe_name = OUTPUT_NAME or PROJECT_NAME.lower().replace(" ", "_")
output_dir = f"{COCO_VOLUME}/{safe_name}"

for split_name, split_images, split_anns in [("train", train_images, train_anns), ("test", test_images, test_anns)]:
    split_dir = f"{output_dir}/{split_name}"
    dbutils.fs.mkdirs(split_dir)

    split_coco = {
        "images": split_images,
        "annotations": split_anns,
        "categories": categories,
    }
    ann_path = f"{split_dir}/_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(split_coco, f)

    copied = 0
    export_images_dir = f"{export_path}/images"
    for img in split_images:
        src = f"{export_images_dir}/{img['file_name']}"
        dst = f"{split_dir}/{img['file_name']}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    print(f"{split_name}: {len(split_images)} images ({copied} copied), {len(split_anns)} annotations → {split_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify

# COMMAND ----------

for split in ["train", "test"]:
    ann_path = f"{output_dir}/{split}/_annotations.coco.json"
    with open(ann_path) as f:
        c = json.load(f)
    img_count = len(os.listdir(f"{output_dir}/{split}")) - 1
    print(f"{split}: {len(c['images'])} images in JSON, {img_count} files on disk, {len(c['annotations'])} annotations")
    print(f"  Categories: {[cat['name'] for cat in c['categories']]}")

print(f"\nReady for finetune notebook 03 with:")
print(f"  UC_COCO_VOLUME = {COCO_VOLUME}")
print(f"  DATASET_NAME = {safe_name}")
