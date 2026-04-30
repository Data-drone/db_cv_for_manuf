# Databricks notebook source
# MAGIC %md
# MAGIC # DINOv3 Image Embedding Serving on Databricks
# MAGIC
# MAGIC Deploys Meta's DINOv3 as a custom MLflow PyFunc model on GPU_LARGE (A100 80GB).
# MAGIC
# MAGIC DINOv3 is the successor to DINOv2, released August 2025. Trained on LVD-1689M
# MAGIC (1.7B curated images, 12x DINOv2's 142M). Same self-supervised ViT approach
# MAGIC but with significantly improved feature quality.
# MAGIC
# MAGIC **Variant:** `facebook/dinov3-vitl16-pretrain-lvd1689m` (~300M params, ViT-L/16)
# MAGIC
# MAGIC **Key differences from DINOv2:**
# MAGIC - Patch size 16 (vs 14 in DINOv2)
# MAGIC - 12x larger pretraining dataset
# MAGIC - Gated model (requires HF token with license accepted)
# MAGIC - Requires transformers >= 4.56.0 (custom `dinov3_vit` architecture)
# MAGIC - License: custom dinov3-license (non-Apache, requires acceptance)
# MAGIC
# MAGIC **Precision:** BF16 on A100 (FP16 causes NaN on T4, but BF16 has FP32's
# MAGIC exponent range and works on A100/A10G). `torch.compile` enabled for ~1.5-2x speedup.
# MAGIC
# MAGIC **Applied lessons from DINOv2 + SAM 3.1 deployments:**
# MAGIC - `torch==2.6.0` + `torchvision==0.21.0` (CUDA 12.4 compat)
# MAGIC - BF16 + torch.compile on A100 for optimal throughput
# MAGIC - Weights bundled as artifacts (no HF network at serve time)
# MAGIC - Robust DataFrame input parsing
# MAGIC - L2-normalized embeddings returned as JSON strings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Widget Definitions

# COMMAND ----------

dbutils.widgets.text("catalog", "brian_gen_ai")
dbutils.widgets.text("schema", "cv_manufacturing")
dbutils.widgets.text("hf_secret_scope", "cv-manufacturing")
dbutils.widgets.text("hf_secret_key", "hf-token")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install Dependencies

# COMMAND ----------

# MAGIC %pip install mlflow==2.19.0 "databricks-sdk>=0.55.0" "transformers>=4.56.0" huggingface_hub pillow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configuration

# COMMAND ----------

UC_CATALOG = dbutils.widgets.get("catalog")
UC_SCHEMA = dbutils.widgets.get("schema")
HF_SECRET_SCOPE = dbutils.widgets.get("hf_secret_scope")
HF_SECRET_KEY = dbutils.widgets.get("hf_secret_key")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download Model Weights

# COMMAND ----------

import os
from huggingface_hub import snapshot_download

HF_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
LOCAL_MODEL_DIR = "/local_disk0/dinov3_weights"

HF_TOKEN = dbutils.secrets.get(scope=HF_SECRET_SCOPE, key=HF_SECRET_KEY)
os.environ["HF_TOKEN"] = HF_TOKEN

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=LOCAL_MODEL_DIR,
    allow_patterns=["*.json", "*.safetensors"],
    token=HF_TOKEN,
)

print("Files:", os.listdir(LOCAL_MODEL_DIR))
total_size = sum(
    os.path.getsize(os.path.join(LOCAL_MODEL_DIR, f))
    for f in os.listdir(LOCAL_MODEL_DIR)
    if os.path.isfile(os.path.join(LOCAL_MODEL_DIR, f))
) / (1024 ** 2)
print(f"Total size: {total_size:.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sanity Check — Load and Run Locally

# COMMAND ----------

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import io, base64

processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModel.from_pretrained(LOCAL_MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()

test_img = Image.new("RGB", (256, 256), (128, 64, 200))
inputs = processor(images=test_img, return_tensors="pt")
pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)

with torch.inference_mode():
    out = model(pixel_values=pixel_values)
    emb = out.pooler_output.float()
    emb_norm = torch.nn.functional.normalize(emb, dim=-1)

print(f"Embedding shape: {emb.shape}")
print(f"L2-normalized norm: {emb_norm.norm(dim=-1).item():.4f}")
print(f"First 5 dims: {emb_norm[0, :5].cpu().numpy()}")
print(f"Model config hidden_size: {model.config.hidden_size}")
print(f"dtype: {next(model.parameters()).dtype}")

del model, processor
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define Custom PyFunc Wrapper
# MAGIC
# MAGIC Same interface as DINOv2 — input: base64 image, output: JSON-encoded embedding.

# COMMAND ----------

import base64
import io
import json
import os

import mlflow.pyfunc
import numpy as np
import pandas as pd


class Dinov3Embedder(mlflow.pyfunc.PythonModel):
    """DINOv3 image embedding model.

    Input: DataFrame/dict/list with base64-encoded image strings.
    Output: DataFrame with 'embedding' column containing JSON-encoded
    L2-normalized float lists.
    """

    MAX_BATCH_SIZE = 8
    MAX_B64_CHARS = 12 * 1024 * 1024  # ~9 MB decoded

    def load_context(self, context):
        import threading
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("DINOv3 serving requires a CUDA GPU")

        self.device = "cuda"
        self._gpu_lock = threading.Lock()
        # BF16 on A100/A10G — safe (same exponent range as FP32, no NaN).
        # FP16 causes NaN on T4 due to attention overflow.
        self.dtype = torch.bfloat16

        weights_path = context.artifacts["weights"]
        self.processor = AutoImageProcessor.from_pretrained(
            weights_path, local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            weights_path,
            torch_dtype=self.dtype,
            local_files_only=True,
        ).to(self.device).eval()

        # torch.compile (default mode supports dynamic batch sizes)
        self.model = torch.compile(self.model)

        # Warm up compile cache at common batch sizes
        for bs in [1, 4, 8]:
            dummy = torch.randn(bs, 3, 224, 224, device=self.device, dtype=self.dtype)
            with torch.inference_mode():
                self.model(pixel_values=dummy)
            del dummy
        torch.cuda.empty_cache()

        print(f"DINOv3 loaded on {self.device}, dtype={self.dtype}, "
              f"compiled=True, hidden_size={self.model.config.hidden_size}")

    def _decode_image(self, value):
        """Robustly decode a base64 image string from various wrapping formats."""
        import base64
        import io
        import json
        import numpy as np
        from PIL import Image

        if isinstance(value, (list, np.ndarray)):
            if len(value) == 0:
                raise ValueError("Empty image value")
            value = value[0]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if not isinstance(value, str):
            value = str(value)

        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list) and parsed:
                    s = parsed[0]
            except (json.JSONDecodeError, TypeError):
                pass

        if s.lower().startswith("data:") and "," in s:
            s = s.split(",", 1)[1]

        if len(s) > self.MAX_B64_CHARS:
            raise ValueError(f"Image payload too large ({len(s)} chars, max {self.MAX_B64_CHARS})")

        try:
            raw = base64.b64decode(s, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}") from e

        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot decode image: {e}") from e

    def _as_list(self, value):
        """Normalize a scalar or collection to a list (avoids splitting strings)."""
        import numpy as np
        import pandas as pd
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _extract_series(self, model_input):
        """Pull a list of image strings from whatever format Databricks serving sends."""
        import numpy as np
        import pandas as pd
        if isinstance(model_input, pd.DataFrame):
            col = "image" if "image" in model_input.columns else model_input.columns[0]
            return model_input[col].tolist()
        if isinstance(model_input, dict):
            if "image" in model_input:
                return self._as_list(model_input["image"])
            if not model_input:
                raise ValueError("Input dict is empty")
            return self._as_list(next(iter(model_input.values())))
        if isinstance(model_input, (list, tuple, np.ndarray)):
            return list(model_input)
        return [model_input]

    def predict(self, context, model_input, params=None):
        import json
        import torch
        import pandas as pd

        series = self._extract_series(model_input)

        if len(series) == 0:
            raise ValueError("No input images provided")
        if len(series) > self.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(series)} exceeds max {self.MAX_BATCH_SIZE}")

        # Decode images, tracking per-row errors
        results = [None] * len(series)
        valid_images = []
        valid_indices = []
        for i, v in enumerate(series):
            try:
                img = self._decode_image(v)
                valid_images.append(img)
                valid_indices.append(i)
            except Exception as e:
                results[i] = json.dumps({"error": str(e)[:200]})

        if valid_images:
            pixel_values = None
            out = None
            emb = None
            try:
                inputs = self.processor(images=valid_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

                with self._gpu_lock:
                    with torch.inference_mode():
                        out = self.model(pixel_values=pixel_values)
                        emb = out.pooler_output.float()
                        emb = torch.nn.functional.normalize(emb, dim=-1)
                        emb_cpu = emb.cpu().numpy().tolist()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise
            finally:
                del pixel_values, out, emb

            for idx, emb_vec in zip(valid_indices, emb_cpu):
                results[idx] = json.dumps(emb_vec)

        # Fill any remaining None slots (shouldn't happen, but safety)
        for i in range(len(results)):
            if results[i] is None:
                results[i] = json.dumps({"error": "unknown error"})

        return pd.DataFrame({"embedding": results})


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log Model to MLflow + Unity Catalog

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")

REGISTERED_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.dinov3_embedder"

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "image")]),
    outputs=Schema([ColSpec("string", "embedding")]),
)

_tiny_img = Image.new("RGB", (64, 64), (100, 150, 200))
_buf = io.BytesIO()
_tiny_img.save(_buf, format="PNG")
_tiny_b64 = base64.b64encode(_buf.getvalue()).decode()
input_example = pd.DataFrame({"image": [_tiny_b64]})

conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.12",
        "pip<=25.0.1",
        {
            "pip": [
                "mlflow==2.19.0",
                "torch==2.6.0+cu124",
                "torchvision==0.21.0+cu124",
                "--extra-index-url https://download.pytorch.org/whl/cu124",
                "transformers==4.56.0",
                "pillow",
                "numpy",
            ]
        },
    ],
    "name": "mlflow-env",
}

with mlflow.start_run(run_name="dinov3-vitl16-v1") as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Dinov3Embedder(),
        artifacts={"weights": LOCAL_MODEL_DIR},
        signature=signature,
        input_example=input_example,
        conda_env=conda_env,
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    print(f"Run ID: {run.info.run_id}")

print(f"Registered to: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy to Serving Endpoint
# MAGIC
# MAGIC Creates / updates the serving endpoint on GPU_MEDIUM (A10G 24GB).
# MAGIC DINOv3 ViT-L/16 in BF16 uses ~1.2GB VRAM — fits easily on A10G.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from databricks.sdk.errors import ResourceAlreadyExists

ENDPOINT_NAME = f"cv-dinov3-{UC_SCHEMA}"
w = WorkspaceClient()

from mlflow import MlflowClient
mc = MlflowClient(registry_uri="databricks-uc")
latest = max(int(v.version) for v in mc.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"))
print(f"Deploying v{latest}")

served_entity = ServedEntityInput(
    name="dinov3-a10g",
    entity_name=REGISTERED_MODEL_NAME,
    entity_version=str(latest),
    workload_type="GPU_MEDIUM",
    workload_size="Small",
    scale_to_zero_enabled=True,
)

try:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=[served_entity]),
        route_optimized=True,
    )
    print(f"Created endpoint {ENDPOINT_NAME}")
except ResourceAlreadyExists:
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
    )
    print(f"Updated endpoint {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sample REST Query

# COMMAND ----------

# MAGIC %md
# MAGIC ```python
# MAGIC import requests, json, base64
# MAGIC
# MAGIC URL = "https://<workspace>/serving-endpoints/<endpoint-name>/invocations"
# MAGIC HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
# MAGIC
# MAGIC with open("image.jpg", "rb") as f:
# MAGIC     img_b64 = base64.b64encode(f.read()).decode()
# MAGIC
# MAGIC payload = {"dataframe_records": [{"image": img_b64}]}
# MAGIC resp = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
# MAGIC resp.raise_for_status()
# MAGIC
# MAGIC result = resp.json()
# MAGIC embedding = json.loads(result["predictions"][0]["embedding"])
# MAGIC print(f"Embedding dim: {len(embedding)}")
# MAGIC
# MAGIC # For similarity, compare with dot product (L2-normalized)
# MAGIC import numpy as np
# MAGIC cosine_sim = float(np.dot(emb_a, emb_b))
# MAGIC ```
