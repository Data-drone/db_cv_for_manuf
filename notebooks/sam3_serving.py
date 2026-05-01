# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy SAM 3.1 (Segment Anything with Concepts) via Databricks Model Serving
# MAGIC
# MAGIC Wraps Meta's SAM 3.1 as a custom MLflow PyFunc model and deploys it to
# MAGIC Databricks GPU model serving on an A100 (GPU_LARGE).
# MAGIC
# MAGIC - *Model*: facebook/sam3.1 (Segment Anything with Concepts + Object Multiplex)
# MAGIC - *Architecture*: ViT backbone (1024 embed, 32 depth) + DETR detector + SAM2 tracker
# MAGIC - *Parameters*: 848M (~6.5 GB checkpoint)
# MAGIC - *Serving*: Databricks Model Serving with GPU_MEDIUM (A10G 24GB)
# MAGIC - *Precision*: FP16 autocast (BF16 hits unsupported ScalarType in SAM3 ops)
# MAGIC - *Compiled*: torch.compile enabled for ~1.5-2x speedup
# MAGIC - *VRAM*: ~5-7 GB for single image inference
# MAGIC - *Catalog*: `{catalog}.{schema}.sam3_1_serving` (parameterised via widgets)

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

# MAGIC %pip install mlflow==2.19.0 "databricks-sdk>=0.55.0" huggingface_hub pillow sam3 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configuration

# COMMAND ----------

UC_CATALOG = dbutils.widgets.get("catalog")
UC_SCHEMA = dbutils.widgets.get("schema")
HF_SECRET_SCOPE = dbutils.widgets.get("hf_secret_scope")
HF_SECRET_KEY = dbutils.widgets.get("hf_secret_key")

import os

MODEL_ID = "facebook/sam3.1"
REGISTERED_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.sam3_1_serving"
ENDPOINT_NAME = f"cv-sam31-{UC_SCHEMA}"
LOCAL_MODEL_DIR = "/local_disk0/models/sam3.1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download Model
# MAGIC
# MAGIC SAM 3.1 is gated on HuggingFace — requires license acceptance.
# MAGIC Token loaded from Databricks secrets (scope/key configured via widgets).

# COMMAND ----------

from huggingface_hub import snapshot_download

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

HF_TOKEN = dbutils.secrets.get(scope=HF_SECRET_SCOPE, key=HF_SECRET_KEY)
os.environ["HF_TOKEN"] = HF_TOKEN

local_path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_MODEL_DIR,
    token=HF_TOKEN,
    allow_patterns=["*.pt", "*.safetensors", "*.json", "*.yaml", "*.txt", "*.gz", "config*", "tokenizer*"],
    ignore_patterns=["*video*", "*.md", "*.git*"],
)
print(f"Downloaded to: {local_path}")

import glob
files = glob.glob(os.path.join(local_path, "*.pt")) + glob.glob(os.path.join(local_path, "*.safetensors"))
for f in files:
    size_gb = os.path.getsize(f) / 1e9
    print(f"  {os.path.basename(f)}: {size_gb:.2f} GB")

# Copy BPE tokenizer vocab into model dir (sam3 package may not include it in sdist builds)
import shutil
from importlib import resources as importlib_resources
bpe_dest = os.path.join(LOCAL_MODEL_DIR, "bpe_simple_vocab_16e6.txt.gz")
try:
    bpe_ref = importlib_resources.files("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    with importlib_resources.as_file(bpe_ref) as bpe_src:
        shutil.copy2(str(bpe_src), bpe_dest)
    print(f"Copied BPE vocab from sam3 package: {bpe_dest}")
except (ModuleNotFoundError, FileNotFoundError, TypeError):
    # Fallback: download from OpenAI CLIP repo (pinned commit)
    import urllib.request
    bpe_url = "https://github.com/openai/CLIP/raw/a1d071733d7111c9c014f024669f959182114e33/clip/bpe_simple_vocab_16e6.txt.gz"
    urllib.request.urlretrieve(bpe_url, bpe_dest)
    print(f"Downloaded BPE vocab from CLIP: {bpe_dest}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Custom PyFunc Wrapper
# MAGIC
# MAGIC SAM 3.1 is a vision segmentation model (not an LLM), so we wrap it in a
# MAGIC custom `mlflow.pyfunc.PythonModel`. The model accepts:
# MAGIC
# MAGIC - **image**: base64-encoded image
# MAGIC - **prompt**: text prompt for open-vocabulary segmentation
# MAGIC - **prompt_type**: "text" (default), "point", or "box"
# MAGIC - **points**: list of [x, y] pixel coords + optional **labels** (point prompts)
# MAGIC - **box**: [x1, y1, x2, y2] pixel coords (box prompt)
# MAGIC
# MAGIC Text prompts use open-vocabulary detection (DETR). Point/box prompts use
# MAGIC interactive instance segmentation (SAM2 tracker via `model.predict_inst()`).

# COMMAND ----------

import mlflow
import json
import base64
import io
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")


class SAM3ServingModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for SAM 3.1 image segmentation.

    Accepts JSON input with base64-encoded image and text/point/box prompts.
    Returns segmentation masks as RLE, plus bounding boxes and scores.
    """

    def load_context(self, context):
        import os
        import threading
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1 serving requires a CUDA GPU")

        self.device = "cuda"
        self._infer_lock = threading.Lock()

        model_dir = context.artifacts["model"]

        # Resolve BPE tokenizer path from artifacts (bundled during logging)
        bpe_path = os.path.join(model_dir, "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            bpe_path = None  # Fall back to sam3 package default

        # Find checkpoint deterministically — sort and require exactly one
        ckpt_files = sorted(
            f for f in os.listdir(model_dir)
            if f.endswith((".pt", ".safetensors"))
        )
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint (.pt/.safetensors) found in {model_dir}")
        if len(ckpt_files) > 1:
            print(f"Warning: multiple checkpoints found {ckpt_files}, using first: {ckpt_files[0]}")
        checkpoint_path = os.path.join(model_dir, ckpt_files[0])
        print(f"Checkpoint: {checkpoint_path}")

        # Build model with load_from_HF=False (no network access in serving container)
        # compile=True for ~1.5-2x throughput boost on A100
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
            compile=True,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )

        self.model.eval()

        # Don't force dtype conversion on the full model — the DETR text encoder
        # has layers that require FP32. Use FP16 autocast in predict() instead.
        # (BF16 autocast fails with "unsupported ScalarType BFloat16" in SAM3 ops.)
        self.processor = Sam3Processor(self.model, resolution=1008, device=self.device)
        print(f"SAM 3.1 loaded on {self.device} | compiled=True | "
              f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def _mask_to_rle(self, mask):
        """Convert binary mask to run-length encoding for compact JSON output."""
        import numpy as np
        pixels = mask.flatten()
        changes = np.diff(pixels.astype(int))
        runs = np.where(changes != 0)[0] + 1
        runs = np.concatenate([[0], runs, [len(pixels)]])
        lengths = np.diff(runs)
        starts = runs[:-1]
        # Only return runs where mask is True
        rle_pairs = []
        for start, length in zip(starts, lengths):
            if pixels[start]:
                rle_pairs.append({"start": int(start), "length": int(length)})
        return {
            "rle": rle_pairs,
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
        }

    def predict(self, context, model_input, params=None):
        import json
        import numpy as np
        import pandas as pd
        import torch

        # Parse input — handle all Databricks serving formats
        raw_strings = []
        if isinstance(model_input, pd.DataFrame):
            # DataFrame with Array(string) schema: single column, values may be lists
            for col in model_input.columns:
                for val in model_input[col]:
                    if isinstance(val, (list, np.ndarray)):
                        raw_strings.extend([str(v) for v in val])
                    else:
                        raw_strings.append(str(val))
        elif isinstance(model_input, dict):
            for val in model_input.values():
                if isinstance(val, (list, np.ndarray)):
                    raw_strings.extend([str(v) for v in val])
                else:
                    raw_strings.append(str(val))
        elif isinstance(model_input, list):
            raw_strings = [str(v) for v in model_input]
        else:
            raw_strings = [str(model_input)]

        # Parse JSON strings into dicts
        rows = []
        for s in raw_strings:
            try:
                rows.append(json.loads(s) if isinstance(s, str) else s)
            except (json.JSONDecodeError, TypeError):
                rows.append({"raw": s})

        results = []
        for row in rows:
            try:
                with self._infer_lock:
                    result = self._process_single(row)
                results.append(json.dumps(result))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                results.append(json.dumps({"error": str(e)}))

        return pd.DataFrame({"output": results})

    def _process_single(self, row):
        import base64
        import io
        import numpy as np
        import torch
        from PIL import Image

        # Decode image from base64
        image_b64 = row.get("image", "")
        if not image_b64:
            return {"error": "No 'image' field provided (expected base64-encoded image)"}

        # Strip data URI prefix if present
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt_type = row.get("prompt_type", "text")
        prompt_text = row.get("prompt", "")

        state = None
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                state = self.processor.set_image(image)

                if prompt_type == "text" and prompt_text:
                    output = self.processor.set_text_prompt(state=state, prompt=prompt_text)
                    return self._format_detection_output(output, image)

                elif prompt_type == "point" and "points" in row:
                    points = row["points"]
                    labels = row.get("labels", [1] * len(points))
                    masks, scores, logits = self.model.predict_inst(
                        state,
                        point_coords=np.array(points, dtype=np.float32),
                        point_labels=np.array(labels, dtype=np.int32),
                        multimask_output=True,
                    )
                    return self._format_inst_output(masks, scores, image)

                elif prompt_type == "box" and "box" in row:
                    box = row["box"]
                    masks, scores, logits = self.model.predict_inst(
                        state,
                        point_coords=None,
                        point_labels=None,
                        box=np.array(box, dtype=np.float32)[None, :],
                        multimask_output=False,
                    )
                    return self._format_inst_output(masks, scores, image)

                else:
                    return {"error": f"Invalid prompt_type '{prompt_type}' or missing prompt data"}
        finally:
            del state
            torch.cuda.empty_cache()

    def _format_detection_output(self, output, image):
        """Format output from processor (text prompts) — returns masks, boxes, scores dicts."""
        import numpy as np
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        response = {
            "num_detections": 0,
            "detections": [],
            "image_size": {"width": image.width, "height": image.height},
        }

        if masks is not None:
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            num = len(masks) if masks is not None else 0
            response["num_detections"] = int(num)

            for i in range(num):
                det = {}
                if masks is not None and i < len(masks):
                    mask_np = masks[i] if masks[i].ndim == 2 else masks[i][0]
                    det["mask_rle"] = self._mask_to_rle(mask_np > 0.5)
                if boxes is not None and i < len(boxes):
                    b = boxes[i].tolist()
                    det["box"] = {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                if scores is not None and i < len(scores):
                    det["score"] = float(scores[i])
                response["detections"].append(det)

        return response

    def _format_inst_output(self, masks, scores, image):
        """Format output from model.predict_inst() — returns (masks, scores, logits) arrays."""
        import numpy as np
        response = {
            "num_detections": 0,
            "detections": [],
            "image_size": {"width": image.width, "height": image.height},
        }

        if masks is not None:
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            num = len(masks)
            response["num_detections"] = int(num)

            for i in range(num):
                det = {}
                mask_np = masks[i] if masks[i].ndim == 2 else masks[i][0]
                det["mask_rle"] = self._mask_to_rle(mask_np > 0.5)
                if scores is not None and i < len(scores):
                    det["score"] = float(scores[i])
                response["detections"].append(det)

        return response


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log Model to Unity Catalog
# MAGIC
# MAGIC Logged with mlflow 2.19.0 to avoid cloudpickle compat issues.

# COMMAND ----------

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
                "sam3==0.1.0",
                "psutil",
                "iopath",
                "pillow",
                "numpy",
                "huggingface_hub",
            ]
        },
    ],
    "name": "mlflow-env",
}

input_schema = Schema([ColSpec("string", "input")])
output_schema = Schema([ColSpec("string", "output")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Build a sample input example so MLflow generates serving_input_example.json
import pandas as pd
from PIL import Image as _PILImage

_sample_img = _PILImage.new("RGB", (64, 64), (100, 150, 200))
_sample_buf = io.BytesIO()
_sample_img.save(_sample_buf, format="PNG")
_sample_b64 = base64.b64encode(_sample_buf.getvalue()).decode()
_sample_payload = json.dumps({
    "image": _sample_b64,
    "prompt_type": "text",
    "prompt": "object",
})
input_example = pd.DataFrame({"input": [_sample_payload]})

model = SAM3ServingModel()

print("Logging SAM 3.1 model to MLflow...")
with mlflow.start_run(run_name="sam3-1-t4-v1") as run:
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        artifacts={"model": LOCAL_MODEL_DIR},
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    print(f"Run ID: {run.info.run_id}")

print("Model registered!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Serving Endpoint
# MAGIC
# MAGIC Deploys on GPU_MEDIUM (A10G 24GB). SAM 3.1 with FP16 autocast + torch.compile
# MAGIC uses ~5-7 GB VRAM — fits on A10G with room for batch processing.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)
from databricks.sdk.errors import ResourceAlreadyExists

w = WorkspaceClient()

from mlflow import MlflowClient
mc = MlflowClient(registry_uri="databricks-uc")
versions_list = mc.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions_list)
print(f"Latest model version: {latest_version}")

served_entity = ServedEntityInput(
    name="sam31-a10g",
    entity_name=REGISTERED_MODEL_NAME,
    entity_version=str(latest_version),
    workload_size="Small",
    workload_type=ServingModelWorkloadType.GPU_MEDIUM,
    scale_to_zero_enabled=True,
)

try:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
        ),
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
# MAGIC ## 7. Test Endpoint
# MAGIC
# MAGIC Sends a test image with a text prompt. The endpoint may take a few minutes
# MAGIC to become READY after creation.

# COMMAND ----------

import json
import base64
import time
from PIL import Image
import io
import numpy as np

ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
ready = ep.state.ready.value if ep.state and ep.state.ready else "NOT_READY"
print(f"Endpoint state: ready={ready}")

if ready == "READY":
    arr = np.full((256, 256, 3), 255, dtype=np.uint8)
    arr[64:192, 64:192] = [255, 0, 0]
    test_img = Image.fromarray(arr)

    buf = io.BytesIO()
    test_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    request_payload = json.dumps({
        "image": img_b64,
        "prompt": "red square",
        "prompt_type": "text",
    })

    print("\nTest query: segment 'red square' in synthetic image...")
    start = time.time()
    try:
        response = w.serving_endpoints.query(
            name=ENDPOINT_NAME,
            inputs=[request_payload],
        )
        elapsed = time.time() - start
        predictions = response.predictions
        if predictions:
            result = json.loads(predictions[0])
            print(f"Detections: {result.get('num_detections', 0)}")
            print(f"Latency: {elapsed:.2f}s")
            for i, det in enumerate(result.get("detections", [])):
                print(f"  [{i}] score={det.get('score', 'N/A'):.3f} box={det.get('box', 'N/A')}")
        else:
            print(f"Unexpected response: {response.as_dict()}")
    except Exception as e:
        print(f"Test failed (endpoint may need warmup): {e}")
else:
    print(f"\nEndpoint not ready (state: {ready}). Container is still deploying.")
    print(f"Check the UI for endpoint: {ENDPOINT_NAME}")
    print("Run tests manually once READY.")
