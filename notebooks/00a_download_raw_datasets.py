# Databricks notebook source
# MAGIC %md
# MAGIC # Download Raw Datasets to UC Volume
# MAGIC
# MAGIC Downloads the three CV manufacturing datasets from their public sources
# MAGIC and stages them into the `raw` UC volume.
# MAGIC
# MAGIC | Dataset | Source | Format |
# MAGIC |---------|--------|--------|
# MAGIC | SHWD | Google Drive | VOC ZIP (XML + JPEG) |
# MAGIC | DeepPCB | GitHub | Image pairs + TXT annotations |
# MAGIC | Corrosion | HuggingFace | Parquet with embedded images |
# MAGIC
# MAGIC **Run once per workspace** — subsequent runs skip already-downloaded files.

# COMMAND ----------

# MAGIC %pip install gdown datasets
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "brian_gen_ai")
dbutils.widgets.text("schema", "cv_manufacturing")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

print(f"Target: {RAW_VOLUME}")

def volume_mkdirs(path):
    """Create directories inside a UC Volume.

    os.makedirs fails on the FUSE mount when parent paths at the
    catalog/schema level don't exist yet. dbutils.fs.mkdirs works
    because it goes through the UC API.
    """
    dbutils.fs.mkdirs(path)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. SHWD — Safety Helmet Wearing Dataset

# COMMAND ----------

import os
import gdown

shwd_dir = f"{RAW_VOLUME}/shwd_safety_helmet"
shwd_zip = f"{shwd_dir}/shwd_voc2028.zip"

volume_mkdirs(shwd_dir)

if os.path.exists(shwd_zip):
    size_mb = os.path.getsize(shwd_zip) / (1024 * 1024)
    print(f"SHWD already downloaded: {shwd_zip} ({size_mb:.0f} MB)")
else:
    print("Downloading SHWD from Google Drive...")
    gdrive_id = "1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX"
    gdown.download(id=gdrive_id, output=shwd_zip, quiet=False)
    size_mb = os.path.getsize(shwd_zip) / (1024 * 1024)
    print(f"Downloaded: {shwd_zip} ({size_mb:.0f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. DeepPCB — PCB Defect Detection

# COMMAND ----------

import subprocess
import shutil

deeppcb_dir = f"{RAW_VOLUME}/deep_pcb_defects"

marker = f"{deeppcb_dir}/.download_complete"
if os.path.exists(marker):
    group_count = len([d for d in os.listdir(deeppcb_dir) if d.startswith("group")])
    print(f"DeepPCB already downloaded: {group_count} groups in {deeppcb_dir}")
else:
    print("Cloning DeepPCB from GitHub...")
    tmp_clone = "/tmp/DeepPCB_clone"
    if os.path.exists(tmp_clone):
        shutil.rmtree(tmp_clone)

    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/tangsanli5201/DeepPCB.git", tmp_clone],
        check=True,
    )

    pcb_data = os.path.join(tmp_clone, "PCBData")
    volume_mkdirs(deeppcb_dir)

    copied = 0
    for item in os.listdir(pcb_data):
        src = os.path.join(pcb_data, item)
        dst = os.path.join(deeppcb_dir, item)
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
            copied += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

    with open(marker, "w") as f:
        f.write("ok")

    shutil.rmtree(tmp_clone, ignore_errors=True)
    print(f"DeepPCB downloaded: {copied} items to {deeppcb_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Corrosion Detection

# COMMAND ----------

from datasets import load_dataset
import json

corrosion_dir = f"{RAW_VOLUME}/corrosion_detection"
marker = f"{corrosion_dir}/.download_complete"

if os.path.exists(marker):
    parquet_files = [f for f in os.listdir(corrosion_dir) if f.endswith(".parquet")]
    print(f"Corrosion already downloaded: {len(parquet_files)} parquet files in {corrosion_dir}")
else:
    print("Downloading corrosion-detection from HuggingFace...")
    volume_mkdirs(corrosion_dir)

    dataset_id = None
    for candidate in ["Francesco/corrosion-detection", "Francesco/corrosion-bi3q3"]:
        try:
            ds = load_dataset(candidate, split="train")
            dataset_id = candidate
            break
        except Exception as e:
            print(f"  {candidate}: {e}")
            continue

    if ds is None:
        raise RuntimeError("Could not load corrosion dataset from any known HuggingFace ID")

    parquet_path = f"{corrosion_dir}/train.parquet"
    ds.to_parquet(parquet_path)

    with open(f"{corrosion_dir}/metadata.json", "w") as f:
        json.dump({
            "source": dataset_id,
            "split": "train",
            "num_rows": len(ds),
            "features": str(ds.features),
        }, f, indent=2)

    with open(marker, "w") as f:
        f.write("ok")

    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"Corrosion downloaded: {len(ds)} rows, {size_mb:.0f} MB to {corrosion_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

for dataset_name in ["shwd_safety_helmet", "deep_pcb_defects", "corrosion_detection"]:
    path = f"{RAW_VOLUME}/{dataset_name}"
    if os.path.exists(path):
        items = os.listdir(path)
        total_size = sum(
            os.path.getsize(os.path.join(path, f))
            for f in items
            if os.path.isfile(os.path.join(path, f))
        )
        print(f"  {dataset_name}: {len(items)} items, {total_size / (1024*1024):.0f} MB top-level files")
    else:
        print(f"  {dataset_name}: NOT FOUND")
