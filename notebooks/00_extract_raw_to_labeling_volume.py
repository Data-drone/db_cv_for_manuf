# Databricks notebook source
# MAGIC %md
# MAGIC # Extract Raw Datasets to Labeling Volume
# MAGIC
# MAGIC Reads archives/parquets from the `raw` UC volume and writes flat JPEGs
# MAGIC to the `labeling` volume, one subfolder per dataset. The labeling volume
# MAGIC is what the CV Explorer app scans when creating projects.
# MAGIC
# MAGIC | Dataset | Raw format | Output |
# MAGIC |---------|-----------|--------|
# MAGIC | SHWD | VOC ZIP → JPEGImages/ | `labeling/shwd/*.jpg` |
# MAGIC | DeepPCB | Group folders with `_test.jpg` + `_temp.jpg` | `labeling/deeppcb/{group}_{file}_test.jpg` |
# MAGIC | Corrosion | Parquet with embedded PIL images | `labeling/corrosion/{index:06d}.jpg` |
# MAGIC
# MAGIC **Idempotent** — skips files that already exist at destination.

# COMMAND ----------

# MAGIC %pip install pillow
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "brian_gen_ai")
dbutils.widgets.text("schema", "cv_manufacturing")
dbutils.widgets.text("dataset", "all", "Dataset (shwd / deeppcb / corrosion / all)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
DATASET = dbutils.widgets.get("dataset").strip().lower()

RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
LABELING_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/labeling"

print(f"Raw:      {RAW_VOLUME}")
print(f"Labeling: {LABELING_VOLUME}")
print(f"Dataset:  {DATASET}")

def volume_mkdirs(path):
    """Create directories inside a UC Volume via dbutils (avoids FUSE permission errors)."""
    dbutils.fs.mkdirs(path)  # noqa: F821

# COMMAND ----------

import os
import shutil
import zipfile
from pathlib import Path

def extract_shwd():
    """Extract SHWD VOC ZIP → flat JPEGs in labeling/shwd/."""
    src_zip = f"{RAW_VOLUME}/shwd_safety_helmet/shwd_voc2028.zip"
    dst_dir = f"{LABELING_VOLUME}/shwd"
    volume_mkdirs(dst_dir)

    existing = set(os.listdir(dst_dir)) if os.path.exists(dst_dir) else set()
    if existing:
        print(f"SHWD: {len(existing)} files already in {dst_dir}")

    if not os.path.exists(src_zip):
        print(f"SHWD: source not found at {src_zip} — run 00a_download_raw_datasets first")
        return 0

    tmp_dir = "/tmp/shwd_extract"
    if not os.path.exists(os.path.join(tmp_dir, "VOC2028", "JPEGImages")):
        print("SHWD: extracting ZIP...")
        with zipfile.ZipFile(src_zip, 'r') as zf:
            zf.extractall(tmp_dir)

    jpeg_dir = os.path.join(tmp_dir, "VOC2028", "JPEGImages")
    if not os.path.exists(jpeg_dir):
        for root, dirs, files in os.walk(tmp_dir):
            if "JPEGImages" in dirs:
                jpeg_dir = os.path.join(root, "JPEGImages")
                break

    if not os.path.exists(jpeg_dir):
        print(f"SHWD: JPEGImages directory not found in ZIP")
        return 0

    files = [f for f in os.listdir(jpeg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    copied = 0
    for fname in files:
        if fname in existing:
            continue
        shutil.copy2(os.path.join(jpeg_dir, fname), os.path.join(dst_dir, fname))
        copied += 1

    total = len(existing) + copied
    print(f"SHWD: {total} images in {dst_dir} ({copied} new)")
    return total

# COMMAND ----------

def extract_deeppcb():
    """Extract DeepPCB test images → flat JPEGs in labeling/deeppcb/.

    Only copies *_test.jpg files (defect images, not templates).
    Prefixes filenames with the group folder to guarantee uniqueness.
    """
    src_dir = f"{RAW_VOLUME}/deep_pcb_defects"
    dst_dir = f"{LABELING_VOLUME}/deeppcb"
    volume_mkdirs(dst_dir)

    existing = set(os.listdir(dst_dir)) if os.path.exists(dst_dir) else set()
    if existing:
        print(f"DeepPCB: {len(existing)} files already in {dst_dir}")

    if not os.path.exists(src_dir):
        print(f"DeepPCB: source not found at {src_dir} — run 00a_download_raw_datasets first")
        return 0

    copied = 0
    for group_name in sorted(os.listdir(src_dir)):
        group_path = os.path.join(src_dir, group_name)
        if not os.path.isdir(group_path) or not group_name.startswith("group"):
            continue

        for fname in os.listdir(group_path):
            if not fname.endswith("_test.jpg"):
                continue
            prefixed = f"{group_name}_{fname}"
            if prefixed in existing:
                continue
            shutil.copy2(os.path.join(group_path, fname), os.path.join(dst_dir, prefixed))
            copied += 1

    total = len(existing) + copied
    print(f"DeepPCB: {total} images in {dst_dir} ({copied} new)")
    return total

# COMMAND ----------

def extract_corrosion():
    """Decode corrosion parquet embedded images → flat JPEGs in labeling/corrosion/."""
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    src_parquet = f"{RAW_VOLUME}/corrosion_detection/train.parquet"
    dst_dir = f"{LABELING_VOLUME}/corrosion"
    volume_mkdirs(dst_dir)

    existing = set(os.listdir(dst_dir)) if os.path.exists(dst_dir) else set()
    if existing:
        print(f"Corrosion: {len(existing)} files already in {dst_dir}")

    if not os.path.exists(src_parquet):
        print(f"Corrosion: source not found at {src_parquet} — run 00a_download_raw_datasets first")
        return 0

    table = pq.read_table(src_parquet)
    image_col = table.column("image")

    copied = 0
    for idx in range(len(table)):
        fname = f"{idx:06d}.jpg"
        if fname in existing:
            continue

        img_struct = image_col[idx].as_py()
        img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else img_struct
        if img_bytes is None:
            continue

        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(os.path.join(dst_dir, fname), "JPEG", quality=95)
        copied += 1

    total = len(existing) + copied
    print(f"Corrosion: {total} images in {dst_dir} ({copied} new)")
    return total

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run extraction

# COMMAND ----------

results = {}

if DATASET in ("shwd", "all"):
    results["shwd"] = extract_shwd()

if DATASET in ("deeppcb", "all"):
    results["deeppcb"] = extract_deeppcb()

if DATASET in ("corrosion", "all"):
    results["corrosion"] = extract_corrosion()

if not results:
    print(f"Unknown dataset: '{DATASET}'. Use shwd, deeppcb, corrosion, or all.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("Labeling volume contents:")
for name in ["shwd", "deeppcb", "corrosion"]:
    path = f"{LABELING_VOLUME}/{name}"
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if not f.startswith(".")])
        print(f"  {name}: {count} files")
    else:
        print(f"  {name}: (not extracted)")
