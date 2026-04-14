# Databricks notebook source
# MAGIC %md
# MAGIC # Log SAM 3.1 to MLflow (Unity Catalog)
# MAGIC
# MAGIC This notebook downloads the SAM 3.1 (Segment Anything with Concepts) model from
# MAGIC HuggingFace and registers it in MLflow under Unity Catalog.
# MAGIC
# MAGIC **Model:** `facebook/sam3.1` — 848M params, 3.3GB checkpoint
# MAGIC **Architecture:** Sam3VideoModel (DETR detector + SAM 2 tracker)
# MAGIC **License:** SAM License (see HuggingFace for details)
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks ML Runtime 16.4+ with GPU
# MAGIC - HuggingFace token with access to the gated `facebook/sam3.1` repo

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies

# COMMAND ----------

# MAGIC %pip install huggingface_hub mlflow
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

import os

# Unity Catalog model path
UC_CATALOG = "brian_gen_ai"
UC_SCHEMA = "cv_manufacturing"
UC_MODEL_NAME = "sam3_1"
FULL_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

# HuggingFace model
HF_MODEL_ID = "facebook/sam3.1"

# HuggingFace token — set via Databricks secret or environment variable
# Option 1: Use Databricks secrets (recommended for production)
# HF_TOKEN = dbutils.secrets.get(scope="hf", key="token")
# Option 2: Set directly (for initial setup only)
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set via Databricks secrets or env var

print(f"Model will be registered as: {FULL_MODEL_NAME}")
print(f"HuggingFace model: {HF_MODEL_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download SAM 3.1 checkpoint from HuggingFace

# COMMAND ----------

from huggingface_hub import snapshot_download
import tempfile

# Download to a local temp directory
download_dir = tempfile.mkdtemp(prefix="sam31_")
print(f"Downloading SAM 3.1 to {download_dir}...")

local_path = snapshot_download(
    repo_id=HF_MODEL_ID,
    token=HF_TOKEN,
    local_dir=download_dir,
    ignore_patterns=["*.md", "*.png", ".gitattributes"],
)

print(f"Download complete: {local_path}")

# List downloaded files
for root, dirs, files in os.walk(local_path):
    for f in files:
        fpath = os.path.join(root, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {os.path.relpath(fpath, local_path):40s}  {size_mb:.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Set MLflow to use Unity Catalog

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
print(f"MLflow registry URI: {mlflow.get_registry_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log the model to MLflow

# COMMAND ----------

import json

# Load the model config for metadata
config_path = os.path.join(local_path, "config.json")
with open(config_path) as f:
    model_config = json.load(f)

# Define model metadata
tags = {
    "source": "huggingface",
    "hf_model_id": HF_MODEL_ID,
    "architecture": "Sam3VideoModel",
    "model_family": "SAM",
    "version": "3.1",
    "task": "segmentation,detection,tracking",
    "params": "848M",
    "checkpoint_file": "sam3.1_multiplex.pt",
    "license": "SAM License",
    "framework": "pytorch",
}

# Log the model artifacts to MLflow
with mlflow.start_run(run_name="sam3.1_registration") as run:
    # Log model config and metadata
    mlflow.log_dict(model_config, "model_config.json")
    mlflow.log_params({
        "hf_model_id": HF_MODEL_ID,
        "architecture": "Sam3VideoModel",
        "model_version": "3.1",
        "num_params": "848M",
    })

    # Log the full model directory as artifacts
    mlflow.log_artifacts(local_path, artifact_path="model")

    # Get the run URI for registration
    run_id = run.info.run_id
    artifact_uri = f"runs:/{run_id}/model"

    print(f"Run ID: {run_id}")
    print(f"Artifact URI: {artifact_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register the model in Unity Catalog

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Create the registered model if it doesn't exist
try:
    client.create_registered_model(
        name=FULL_MODEL_NAME,
        description=(
            "SAM 3.1 (Segment Anything with Concepts) by Meta AI. "
            "Unified foundation model for promptable segmentation in images and videos. "
            "Supports text prompts, visual prompts (points, boxes, masks), and "
            "open-vocabulary concept segmentation. "
            "848M parameters, DETR-based detector + SAM 2 tracker architecture. "
            "SAM 3.1 adds Object Multiplex for ~7x faster multi-object tracking."
        ),
        tags=tags,
    )
    print(f"Created registered model: {FULL_MODEL_NAME}")
except Exception as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"Registered model already exists: {FULL_MODEL_NAME}")
    else:
        raise

# COMMAND ----------

# Create a model version from the logged artifacts
mv = client.create_model_version(
    name=FULL_MODEL_NAME,
    source=artifact_uri,
    run_id=run_id,
    description="SAM 3.1 base checkpoint (sam3.1_multiplex.pt) from HuggingFace facebook/sam3.1",
    tags={
        "stage": "base",
        "finetuned": "false",
        "source_checkpoint": "sam3.1_multiplex.pt",
    },
)

print(f"Model version created: {FULL_MODEL_NAME} v{mv.version}")
print(f"Status: {mv.status}")

# COMMAND ----------

# Set alias for easy reference
client.set_registered_model_alias(
    name=FULL_MODEL_NAME,
    alias="base",
    version=mv.version,
)
print(f"Alias 'base' set to version {mv.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify registration

# COMMAND ----------

# Verify the model is registered
model_info = client.get_registered_model(FULL_MODEL_NAME)
print(f"Model: {model_info.name}")
print(f"Description: {model_info.description[:100]}...")
print(f"Tags: {model_info.tags}")

# List versions
versions = client.search_model_versions(f"name='{FULL_MODEL_NAME}'")
for v in versions:
    print(f"  Version {v.version}: status={v.status}, aliases={v.aliases}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Cleanup temp files

# COMMAND ----------

import shutil
shutil.rmtree(download_dir, ignore_errors=True)
print("Temp files cleaned up.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC SAM 3.1 has been registered in Unity Catalog as:
# MAGIC ```
# MAGIC brian_gen_ai.cv_manufacturing.sam3_1
# MAGIC ```
# MAGIC
# MAGIC The base checkpoint is available as version 1 with alias `base`.
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Finetune on manufacturing/safety datasets (SHWD, DeepPCB, Corrosion)
# MAGIC - Register finetuned versions as new model versions
# MAGIC - Deploy for inference via Databricks Model Serving
