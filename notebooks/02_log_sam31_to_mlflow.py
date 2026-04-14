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
# MAGIC ## 5. Log the model to MLflow using pyfunc wrapper
# MAGIC
# MAGIC Unity Catalog requires a proper MLflow model format with a model signature
# MAGIC (input/output type specs). We wrap the SAM 3.1 checkpoint in a pyfunc model.

# COMMAND ----------

import json
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec

# Load the model config for metadata
config_path = os.path.join(local_path, "config.json")
with open(config_path) as f:
    model_config = json.load(f)

# Define a pyfunc wrapper for SAM 3.1
class Sam31Wrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for SAM 3.1.
    Stores the checkpoint and config as artifacts.
    For inference, use the SAM 3 Python API directly — this wrapper
    serves as a model registry entry with proper signature metadata.
    """

    def load_context(self, context):
        """Load model artifacts when the model is loaded."""
        self.model_dir = context.artifacts["model_dir"]

    def predict(self, context, model_input, params=None):
        """
        Placeholder predict — SAM 3.1 inference should use the native
        Sam3Processor API for full functionality (masks, boxes, scores).
        This returns a status message for signature validation.
        """
        return {"status": "Use SAM 3 native API for inference. See sam3.model_builder.build_sam3_image_model()"}


# Define model signature
# Input: image path (string) + text prompt (string)
# Output: detection results (string/json)
signature = ModelSignature(
    inputs=Schema([
        ColSpec("string", "image_path"),
        ColSpec("string", "text_prompt"),
    ]),
    outputs=Schema([
        ColSpec("string", "status"),
    ]),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log and register in Unity Catalog

# COMMAND ----------

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

with mlflow.start_run(run_name="sam3.1_registration") as run:
    # Log config and metadata
    mlflow.log_dict(model_config, "model_config.json")
    mlflow.log_params({
        "hf_model_id": HF_MODEL_ID,
        "architecture": "Sam3VideoModel",
        "model_version": "3.1",
        "num_params": "848M",
    })

    # Log the model with pyfunc wrapper and proper signature
    model_info = mlflow.pyfunc.log_model(
        artifact_path="sam3_1_model",
        python_model=Sam31Wrapper(),
        artifacts={"model_dir": local_path},
        signature=signature,
        registered_model_name=FULL_MODEL_NAME,
        pip_requirements=[
            "torch>=2.7",
            "torchvision",
            "timm>=1.0.17",
            "huggingface_hub",
            "pillow",
        ],
        metadata={
            "hf_model_id": HF_MODEL_ID,
            "architecture": "Sam3VideoModel",
            "version": "3.1",
            "params": "848M",
        },
    )

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Set alias and verify

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Get the latest version
versions = client.search_model_versions(f"name='{FULL_MODEL_NAME}'")
if versions:
    latest = max(versions, key=lambda v: int(v.version))
    # Set alias
    client.set_registered_model_alias(
        name=FULL_MODEL_NAME,
        alias="base",
        version=latest.version,
    )
    # Update tags
    for key, value in tags.items():
        client.set_registered_model_tag(FULL_MODEL_NAME, key, value)

    print(f"Model version: {FULL_MODEL_NAME} v{latest.version}")
    print(f"Alias 'base' set to version {latest.version}")
    print(f"Status: {latest.status}")
else:
    print("No model versions found")

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
