# Databricks notebook source
# MAGIC %md
# MAGIC # Finetune SAM 3.1 for Detection on Databricks AI Runtime
# MAGIC
# MAGIC This notebook finetunes SAM 3.1 (Segment Anything with Concepts) for
# MAGIC object detection on custom COCO-format datasets using Databricks Serverless GPU
# MAGIC (AI Runtime).
# MAGIC
# MAGIC **Model:** `facebook/sam3.1` — 848M params
# MAGIC **Training:** DETR-based detector with text-conditioned detection
# MAGIC **Data format:** COCO JSON annotations (`_annotations.coco.json`)
# MAGIC **GPU:** Databricks AI Runtime — A10 or H100 (Serverless)
# MAGIC
# MAGIC **References:**
# MAGIC - SAM 3 GitHub: https://github.com/facebookresearch/sam3
# MAGIC - SAM 3.1 checkpoint: https://huggingface.co/facebook/sam3.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install SAM 3 and dependencies
# MAGIC
# MAGIC On AI Runtime, install dependencies via `%pip install`.
# MAGIC The AI environment (AI v4) comes with PyTorch + CUDA pre-installed.

# COMMAND ----------

# MAGIC %pip install huggingface_hub timm hydra-core fvcore fairscale tensorboard scipy torchmetrics scikit-image scikit-learn pycocotools iopath ftfy regex
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clone and install SAM 3

# COMMAND ----------

import subprocess
import os

# Clone SAM 3 repo
sam3_dir = "/tmp/sam3"
if not os.path.exists(sam3_dir):
    subprocess.run(
        ["git", "clone", "https://github.com/facebookresearch/sam3.git", sam3_dir],
        check=True,
    )
    # Install SAM 3 package
    subprocess.run(
        ["pip", "install", "-e", f"{sam3_dir}[train]"],
        check=True,
        capture_output=True,
    )
    print("SAM 3 installed successfully")
else:
    print("SAM 3 already cloned")

# Add to path
import sys
if sam3_dir not in sys.path:
    sys.path.insert(0, sam3_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configuration

# COMMAND ----------

import torch

# ── Paths ──────────────────────────────────────────────────────────────
# Unity Catalog volume paths for data (update these for your dataset)
UC_VOLUME_BASE = "/Volumes/brian_gen_ai/cv_manufacturing/raw"

# Local working directory
WORK_DIR = "/tmp/sam3_finetune"
os.makedirs(WORK_DIR, exist_ok=True)

# ── Dataset config (COCO format) ──────────────────────────────────────
# Point these to your COCO-format dataset
# Expected structure:
#   DATASET_ROOT/
#     train/
#       <images>
#       _annotations.coco.json
#     test/  (or valid/)
#       <images>
#       _annotations.coco.json
DATASET_ROOT = os.path.join(WORK_DIR, "dataset")

# ── Model config ──────────────────────────────────────────────────────
HF_MODEL_ID = "facebook/sam3.1"
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set via Databricks secrets or env var

# ── Training hyperparameters ──────────────────────────────────────────
NUM_EPOCHS = 20
LEARNING_RATE_SCALE = 0.1
TRAIN_BATCH_SIZE = 1
NUM_GPUS = 1  # Single GPU on A10, up to 8 on H100
RESOLUTION = 1008
NUM_TRAIN_IMAGES = None  # None = use all images
ENABLE_SEGMENTATION = False  # Set True for mask loss

# ── MLflow ────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "/Shared/cv_manufacturing/sam31_finetune"
UC_MODEL_NAME = "brian_gen_ai.cv_manufacturing.sam3_1"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Authenticate with HuggingFace and download model

# COMMAND ----------

from huggingface_hub import login, snapshot_download

login(token=HF_TOKEN)

# Download SAM 3.1 checkpoint
model_dir = os.path.join(WORK_DIR, "sam3.1_checkpoint")
if not os.path.exists(os.path.join(model_dir, "sam3.1_multiplex.pt")):
    print("Downloading SAM 3.1 checkpoint (~3.3GB)...")
    snapshot_download(
        repo_id=HF_MODEL_ID,
        token=HF_TOKEN,
        local_dir=model_dir,
        ignore_patterns=["*.md", "*.png", ".gitattributes"],
    )
    print("Download complete")
else:
    print("Checkpoint already downloaded")

# Verify checkpoint
ckpt_path = os.path.join(model_dir, "sam3.1_multiplex.pt")
ckpt_size = os.path.getsize(ckpt_path) / (1024**3)
print(f"Checkpoint: {ckpt_path} ({ckpt_size:.1f} GB)")

# BPE vocab path (from cloned repo)
bpe_path = os.path.join(sam3_dir, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
print(f"BPE path: {bpe_path} (exists: {os.path.exists(bpe_path)})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Prepare COCO-format dataset
# MAGIC
# MAGIC Copy or symlink your data from Unity Catalog volumes. The dataset must be in
# MAGIC COCO format with `_annotations.coco.json` files in train/ and test/ folders.
# MAGIC
# MAGIC Example COCO annotation format:
# MAGIC ```json
# MAGIC {
# MAGIC   "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],
# MAGIC   "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": 1234, "iscrowd": 0}],
# MAGIC   "categories": [{"id": 1, "name": "helmet", "supercategory": "safety"}]
# MAGIC }
# MAGIC ```

# COMMAND ----------

# Example: Copy SHWD dataset from UC Volume to local (adjust for your dataset)
# This cell shows how to prepare data — modify for your specific dataset

import shutil
import json

def prepare_coco_dataset(
    src_volume_path: str,
    dest_path: str,
    train_annotations: str = "_annotations.coco.json",
    test_annotations: str = "_annotations.coco.json",
):
    """
    Copy dataset from UC volume to local and verify COCO format.
    Expects src_volume_path to have train/ and test/ subdirectories.
    """
    os.makedirs(dest_path, exist_ok=True)

    for split in ["train", "test"]:
        src = os.path.join(src_volume_path, split)
        dst = os.path.join(dest_path, split)

        if os.path.exists(src):
            if not os.path.exists(dst):
                print(f"Copying {split} data from {src} to {dst}...")
                shutil.copytree(src, dst)
            else:
                print(f"{split} data already exists at {dst}")

            # Verify COCO annotations exist
            ann_file = os.path.join(dst, train_annotations)
            if os.path.exists(ann_file):
                with open(ann_file) as f:
                    coco = json.load(f)
                print(f"  {split}: {len(coco.get('images', []))} images, "
                      f"{len(coco.get('annotations', []))} annotations, "
                      f"{len(coco.get('categories', []))} categories")
            else:
                print(f"  WARNING: No annotations file at {ann_file}")
        else:
            print(f"  WARNING: Source {src} does not exist")

    return dest_path


# Uncomment and modify for your dataset:
# DATASET_ROOT = prepare_coco_dataset(
#     src_volume_path=f"{UC_VOLUME_BASE}/your_dataset",
#     dest_path=DATASET_ROOT,
# )

print(f"Dataset root: {DATASET_ROOT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Build training config
# MAGIC
# MAGIC SAM 3 uses Hydra configs. We build a config programmatically for our
# MAGIC custom dataset that follows the same structure as the Roboflow configs.

# COMMAND ----------

import yaml

def build_training_config(
    dataset_root: str,
    bpe_path: str,
    experiment_log_dir: str,
    num_epochs: int = 20,
    lr_scale: float = 0.1,
    resolution: int = 1008,
    num_train_images: int = None,
    batch_size: int = 1,
    num_gpus: int = 1,
    enable_segmentation: bool = False,
) -> dict:
    """Build a SAM 3 training config dict for a custom COCO dataset."""

    train_img_folder = os.path.join(dataset_root, "train")
    train_ann_file = os.path.join(dataset_root, "train", "_annotations.coco.json")
    val_img_folder = os.path.join(dataset_root, "test")
    val_ann_file = os.path.join(dataset_root, "test", "_annotations.coco.json")

    # Fall back to valid/ if test/ doesn't exist
    if not os.path.exists(val_img_folder):
        val_img_folder = os.path.join(dataset_root, "valid")
        val_ann_file = os.path.join(dataset_root, "valid", "_annotations.coco.json")

    config = {
        "paths": {
            "dataset_root": dataset_root,
            "experiment_log_dir": experiment_log_dir,
            "bpe_path": bpe_path,
        },
        "scratch": {
            "enable_segmentation": enable_segmentation,
            "resolution": resolution,
            "consistent_transform": False,
            "max_ann_per_img": 200,
            "train_norm_mean": [0.5, 0.5, 0.5],
            "train_norm_std": [0.5, 0.5, 0.5],
            "lr_scale": lr_scale,
            "wd": 0.1,
            "scheduler_timescale": 20,
            "scheduler_warmup": 20,
            "scheduler_cooldown": 20,
            "num_train_workers": 4,
            "num_val_workers": 0,
            "gradient_accumulation_steps": 1,
            "train_batch_size": batch_size,
            "val_batch_size": 1,
            "scale_by_find_batch_size": True,
            "use_presence_eval": True,
            "gather_pred_via_filesys": False,
        },
        "trainer": {
            "skip_saving_ckpts": False,
            "empty_gpu_mem_cache_after_eval": True,
            "skip_first_val": True,
            "max_epochs": num_epochs,
            "accelerator": "cuda",
            "seed_value": 42,
            "val_epoch_freq": 5,
            "mode": "train",
            "gradient_accumulation_steps": 1,
            "distributed": {
                "backend": "nccl",
                "find_unused_parameters": True,
                "gradient_as_bucket_view": True,
            },
            "checkpoint": {
                "save_dir": os.path.join(experiment_log_dir, "checkpoints"),
                "save_freq": 0,
            },
        },
        "launcher": {
            "num_nodes": 1,
            "gpus_per_node": num_gpus,
            "experiment_log_dir": experiment_log_dir,
        },
        "submitit": {
            "use_cluster": False,
        },
        "data": {
            "train_img_folder": train_img_folder,
            "train_ann_file": train_ann_file,
            "val_img_folder": val_img_folder,
            "val_ann_file": val_ann_file,
            "num_train_images": num_train_images,
        },
    }

    return config


# Build the config
experiment_dir = os.path.join(WORK_DIR, "experiment")
config = build_training_config(
    dataset_root=DATASET_ROOT,
    bpe_path=bpe_path,
    experiment_log_dir=experiment_dir,
    num_epochs=NUM_EPOCHS,
    lr_scale=LEARNING_RATE_SCALE,
    resolution=RESOLUTION,
    num_train_images=NUM_TRAIN_IMAGES,
    batch_size=TRAIN_BATCH_SIZE,
    num_gpus=NUM_GPUS,
    enable_segmentation=ENABLE_SEGMENTATION,
)

# Save config
config_path = os.path.join(WORK_DIR, "training_config.yaml")
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Config saved to: {config_path}")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Setup MLflow experiment tracking

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")
print(f"MLflow registry: {mlflow.get_registry_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run finetuning
# MAGIC
# MAGIC This cell executes the SAM 3 training loop. For the first run, it's
# MAGIC recommended to test with a small number of epochs and images to verify
# MAGIC the pipeline works end-to-end.

# COMMAND ----------

import sys
import importlib

# Ensure sam3 is importable
sys.path.insert(0, sam3_dir)

# Import SAM 3 training components
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("SAM 3 modules imported successfully")

# COMMAND ----------

# Load the SAM 3.1 model
print("Loading SAM 3.1 model...")
model = build_sam3_image_model(
    bpe_path=bpe_path,
    device="cuda",
    eval_mode=False,
    enable_segmentation=ENABLE_SEGMENTATION,
)
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8a. Direct training using SAM 3's built-in trainer
# MAGIC
# MAGIC SAM 3's training script (`sam3/train/train.py`) handles the full pipeline
# MAGIC including data loading, loss computation, optimization, and evaluation.
# MAGIC We invoke it with our custom config.

# COMMAND ----------

# Run training via SAM 3's training script
# This is the simplest approach — just call the training script
with mlflow.start_run(run_name="sam31_finetune_detection") as run:
    # Log config
    mlflow.log_params({
        "model": HF_MODEL_ID,
        "num_epochs": NUM_EPOCHS,
        "lr_scale": LEARNING_RATE_SCALE,
        "resolution": RESOLUTION,
        "batch_size": TRAIN_BATCH_SIZE,
        "num_gpus": NUM_GPUS,
        "enable_segmentation": ENABLE_SEGMENTATION,
        "num_train_images": str(NUM_TRAIN_IMAGES),
    })
    mlflow.log_artifact(config_path, "config")

    # Run training
    train_cmd = [
        sys.executable,
        os.path.join(sam3_dir, "sam3", "train", "train.py"),
        "-c", config_path,
        "--use-cluster", "0",
        "--num-gpus", str(NUM_GPUS),
    ]

    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(
        train_cmd,
        capture_output=True,
        text=True,
        cwd=sam3_dir,
    )

    print("STDOUT:", result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Log training outputs
    if os.path.exists(experiment_dir):
        mlflow.log_artifacts(experiment_dir, "experiment_outputs")

    # Log the best checkpoint
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if ckpts:
            best_ckpt = sorted(ckpts)[-1]
            mlflow.log_artifact(
                os.path.join(ckpt_dir, best_ckpt),
                "model/checkpoints",
            )
            print(f"Logged checkpoint: {best_ckpt}")

    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluate the finetuned model

# COMMAND ----------

# Run evaluation
eval_cmd = [
    sys.executable,
    os.path.join(sam3_dir, "sam3", "train", "train.py"),
    "-c", config_path,
    "--use-cluster", "0",
    "--num-gpus", "1",
]

# Override mode to val
os.environ["TRAINER_MODE"] = "val"

print("Running evaluation...")
eval_result = subprocess.run(
    eval_cmd,
    capture_output=True,
    text=True,
    cwd=sam3_dir,
    env={**os.environ, "TRAINER_MODE": "val"},
)

print("Eval output:", eval_result.stdout[-3000:] if len(eval_result.stdout) > 3000 else eval_result.stdout)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Register finetuned model in Unity Catalog

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Find the best checkpoint
ckpt_dir = os.path.join(experiment_dir, "checkpoints")
if os.path.exists(ckpt_dir):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if ckpts:
        best_ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])

        # Log as a new model version
        with mlflow.start_run(run_name="sam31_finetuned_registration") as reg_run:
            mlflow.log_artifact(best_ckpt_path, "model")
            mlflow.log_artifact(config_path, "config")

            artifact_uri = f"runs:/{reg_run.info.run_id}/model"

            # Create new version of the registered model
            mv = client.create_model_version(
                name=UC_MODEL_NAME,
                source=artifact_uri,
                run_id=reg_run.info.run_id,
                description="SAM 3.1 finetuned for detection on custom COCO dataset",
                tags={
                    "stage": "finetuned",
                    "finetuned": "true",
                    "task": "detection",
                    "num_epochs": str(NUM_EPOCHS),
                    "base_model": HF_MODEL_ID,
                },
            )

            # Set alias
            client.set_registered_model_alias(
                name=UC_MODEL_NAME,
                alias="finetuned-detection",
                version=mv.version,
            )

            print(f"Registered: {UC_MODEL_NAME} v{mv.version}")
            print(f"Alias: finetuned-detection -> v{mv.version}")
    else:
        print("No checkpoints found to register")
else:
    print(f"Checkpoint directory not found: {ckpt_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Quick inference test with finetuned model

# COMMAND ----------

from PIL import Image
import glob

# Find a test image
test_images = glob.glob(os.path.join(DATASET_ROOT, "test", "*.jpg"))
if not test_images:
    test_images = glob.glob(os.path.join(DATASET_ROOT, "valid", "*.jpg"))

if test_images:
    test_img_path = test_images[0]
    print(f"Test image: {test_img_path}")

    # Load finetuned model for inference
    model.eval()
    processor = Sam3Processor(model)

    image = Image.open(test_img_path)
    inference_state = processor.set_image(image)

    # Get categories from annotations
    ann_path = os.path.join(DATASET_ROOT, "test", "_annotations.coco.json")
    if not os.path.exists(ann_path):
        ann_path = os.path.join(DATASET_ROOT, "valid", "_annotations.coco.json")

    with open(ann_path) as f:
        coco = json.load(f)

    categories = [c["name"] for c in coco.get("categories", [])]
    print(f"Categories: {categories}")

    # Run detection for each category
    for cat_name in categories[:5]:  # First 5 categories
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=cat_name,
        )
        masks = output.get("masks", [])
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])
        print(f"  '{cat_name}': {len(boxes)} detections")
else:
    print("No test images found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Cleanup

# COMMAND ----------

# Optionally clean up local files
# import shutil
# shutil.rmtree(WORK_DIR, ignore_errors=True)
# shutil.rmtree(sam3_dir, ignore_errors=True)
print("Notebook complete. Check MLflow for experiment results.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Installed SAM 3 from GitHub (`facebookresearch/sam3`)
# MAGIC 2. Downloaded SAM 3.1 checkpoint from HuggingFace
# MAGIC 3. Configured training for a COCO-format detection dataset
# MAGIC 4. Ran finetuning with MLflow experiment tracking
# MAGIC 5. Registered the finetuned model in Unity Catalog as a new version
# MAGIC
# MAGIC **To use with your own data:**
# MAGIC - Set `DATASET_ROOT` to point to your COCO-format dataset
# MAGIC - Expected structure: `train/` and `test/` (or `valid/`) subdirectories
# MAGIC - Each subdirectory needs images + `_annotations.coco.json`
# MAGIC - Adjust `NUM_EPOCHS`, `LEARNING_RATE_SCALE`, `RESOLUTION` as needed
# MAGIC
# MAGIC **For segmentation tasks:**
# MAGIC - Set `ENABLE_SEGMENTATION = True`
# MAGIC - Your COCO annotations need RLE-encoded segmentation masks
# MAGIC
# MAGIC **Databricks AI Runtime hardware:**
# MAGIC - A10: Suitable for smaller datasets / lower resolution
# MAGIC - H100: Recommended for full-resolution (1008px) training
# MAGIC - H100 8xGPU: For multi-GPU distributed training
