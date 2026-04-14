# Databricks notebook source
# MAGIC %md
# MAGIC # Finetune SAM 3.1 for Detection on Databricks
# MAGIC
# MAGIC This notebook finetunes SAM 3.1 (Segment Anything with Concepts) for
# MAGIC object detection on the SHWD (Safety Helmet Wearing Dataset) using
# MAGIC SAM 3's built-in Hydra-based training pipeline.
# MAGIC
# MAGIC **Model:** `facebook/sam3.1` — 848M params, DETR detector + SAM 2 tracker
# MAGIC **Data format:** COCO JSON annotations (`_annotations.coco.json`)
# MAGIC **Training script:** `sam3/train/train.py` with Hydra configs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies

# COMMAND ----------

# MAGIC %pip install huggingface_hub timm hydra-core omegaconf fvcore fairscale submitit tensorboard scipy torchmetrics scikit-image scikit-learn pycocotools iopath ftfy regex einops
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clone and install SAM 3

# COMMAND ----------

import subprocess
import os
import sys
import shutil
import json
import tempfile

# --- AI Runtime /tmp workaround ---
# AI Runtime reuses /tmp across job runs. Files from previous runs have
# different ownership and cause PermissionError. Use a fresh unique dir
# for everything to guarantee writability.
def _is_writable(path):
    """Check if an existing path is writable by the current process."""
    try:
        test_file = os.path.join(path, f".write_test_{os.getpid()}")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except (PermissionError, OSError):
        return False

def _get_or_create_dir(preferred, fallback_prefix):
    """Return preferred path if writable, else create a fresh unique dir."""
    if os.path.exists(preferred) and _is_writable(preferred):
        return preferred
    if not os.path.exists(preferred):
        try:
            os.makedirs(preferred, exist_ok=True)
            return preferred
        except (PermissionError, OSError):
            pass
    # Fallback: unique dir
    new_dir = f"{fallback_prefix}_{os.getpid()}"
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

WORK_DIR = _get_or_create_dir("/tmp/sam3_finetune", "/tmp/sam3_finetune")
HF_CACHE_DIR = _get_or_create_dir("/tmp/hf_cache", "/tmp/hf_cache")
print(f"WORK_DIR: {WORK_DIR}")
print(f"HF_CACHE_DIR: {HF_CACHE_DIR}")

# Set HuggingFace cache env vars before any HF imports
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HF_TOKEN_PATH"] = ""  # Disable token file read/write

# Clone SAM3 — try reusing existing, else clone fresh
SAM3_DIR = "/tmp/sam3"
if os.path.exists(os.path.join(SAM3_DIR, "setup.py")) and _is_writable(SAM3_DIR):
    print("SAM 3 already cloned and writable")
else:
    SAM3_DIR = _get_or_create_dir(f"/tmp/sam3_{os.getpid()}", "/tmp/sam3_fresh")
    if not os.path.exists(os.path.join(SAM3_DIR, "setup.py")):
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/sam3.git", SAM3_DIR],
            check=True,
        )
    print(f"SAM 3 cloned to {SAM3_DIR}")

# Always ensure SAM3 is installed (handles retry after partial install)
result = subprocess.run(
    ["pip", "install", "-e", f"{SAM3_DIR}[train]"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("Failed to install SAM 3")
print(f"SAM 3 installed from {SAM3_DIR}")

if SAM3_DIR not in sys.path:
    sys.path.insert(0, SAM3_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configuration

# COMMAND ----------

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Paths ──────────────────────────────────────────────────────────────
UC_COCO_VOLUME = "/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets"
DATASET_NAME = "shwd"
DATASET_ROOT = os.path.join(WORK_DIR, "dataset")

# ── Model ──────────────────────────────────────────────────────────────
HF_MODEL_ID = "facebook/sam3.1"
# On AI Runtime, spark_env_vars are not available. Try env var first,
# then Databricks secrets, then widget parameter.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        HF_TOKEN = dbutils.secrets.get(scope="cv-manufacturing", key="hf-token")  # noqa: F821
        os.environ["HF_TOKEN"] = HF_TOKEN
        print("HF_TOKEN loaded from Databricks secrets (cv-manufacturing/hf-token)")
    except Exception:
        pass
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Set via env var or secrets scope 'cv-manufacturing' key 'hf-token'")

# ── Training hyperparameters ──────────────────────────────────────────
NUM_EPOCHS = 3        # Start small to validate pipeline
NUM_TRAIN_IMAGES = 100  # Subset for quick test; null for full
LEARNING_RATE_SCALE = 0.1
RESOLUTION = 1008  # A10 (24GB) can handle full resolution
NUM_GPUS = 1

# ── MLflow ────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "/Users/0def019e-c076-4fbf-9ab5-6f12c4b9396e/cv_manufacturing/sam31_finetune"
UC_MODEL_NAME = "brian_gen_ai.cv_manufacturing.sam3_1"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Download SAM 3.1 from HuggingFace

# COMMAND ----------

from huggingface_hub import snapshot_download

# Skip login() — it tries to write a stored_tokens file which fails on AI Runtime
# when /tmp/hf_cache is owned by a prior run. Just pass token= directly.

model_dir = os.path.join(WORK_DIR, "sam3.1_checkpoint")
os.makedirs(model_dir, exist_ok=True)
if not os.path.exists(os.path.join(model_dir, "sam3.1_multiplex.pt")):
    print("Downloading SAM 3.1 checkpoint (~3.3GB)...")
    downloaded_path = snapshot_download(
        repo_id=HF_MODEL_ID,
        token=HF_TOKEN,
        cache_dir=os.path.join(HF_CACHE_DIR, "hub"),
        ignore_patterns=["*.md", "*.png", ".gitattributes"],
    )
    # Copy checkpoint file from HF cache to our model_dir
    ckpt_src = os.path.join(downloaded_path, "sam3.1_multiplex.pt")
    ckpt_dst = os.path.join(model_dir, "sam3.1_multiplex.pt")
    if os.path.exists(ckpt_src):
        shutil.copy2(ckpt_src, ckpt_dst)
    else:
        shutil.copytree(downloaded_path, model_dir, dirs_exist_ok=True)
    print("Download complete")
else:
    print("Checkpoint already downloaded")

ckpt_path = os.path.join(model_dir, "sam3.1_multiplex.pt")
print(f"Checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / (1024**3):.1f} GB)")

bpe_path = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
print(f"BPE: {bpe_path} (exists: {os.path.exists(bpe_path)})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Copy COCO dataset from UC volume to local

# COMMAND ----------

src_volume = os.path.join(UC_COCO_VOLUME, DATASET_NAME)

for split in ["train", "test"]:
    src = os.path.join(src_volume, split)
    dst = os.path.join(DATASET_ROOT, split)
    if os.path.exists(dst):
        print(f"{split}/ already copied")
    elif os.path.exists(src):
        print(f"Copying {split}/ from UC volume...")
        shutil.copytree(src, dst)
        print(f"  Done")
    else:
        print(f"WARNING: {src} not found")

    ann = os.path.join(dst, "_annotations.coco.json")
    if os.path.exists(ann):
        with open(ann) as f:
            c = json.load(f)
        print(f"  {split}: {len(c['images'])} images, {len(c['annotations'])} annotations, categories: {[x['name'] for x in c['categories']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Hydra training config
# MAGIC
# MAGIC SAM 3 uses Hydra with `_target_` instantiation. We create a proper config
# MAGIC YAML based on the Roboflow reference config, adapted for our SHWD dataset.

# COMMAND ----------

EXPERIMENT_DIR = os.path.join(WORK_DIR, "experiment")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Paths for the config
train_img_folder = os.path.join(DATASET_ROOT, "train")
train_ann_file = os.path.join(DATASET_ROOT, "train", "_annotations.coco.json")
test_img_folder = os.path.join(DATASET_ROOT, "test")
test_ann_file = os.path.join(DATASET_ROOT, "test", "_annotations.coco.json")

# Build the full Hydra config as a YAML string
# This mirrors sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
# but with our SHWD dataset paths and without the Roboflow-specific job_array logic

num_images_str = str(NUM_TRAIN_IMAGES) if NUM_TRAIN_IMAGES else "null"

config_yaml = f"""# @package _global_
defaults:
  - _self_

paths:
  dataset_root: {DATASET_ROOT}
  experiment_log_dir: {EXPERIMENT_DIR}
  bpe_path: {bpe_path}

# Dataset-specific training config (replaces roboflow_train)
shwd_train:
  num_images: {num_images_str}

  train_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterCrowds
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes:
            _target_: sam3.train.transforms.basic.get_random_resize_scales
            size: ${{scratch.resolution}}
            min_size: 480
            rounded: false
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.PadToSizeAPI
          size: ${{scratch.resolution}}
          consistent_transform: ${{scratch.consistent_transform}}
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.train_norm_mean}}
          std: ${{scratch.train_norm_std}}
        - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
          query_filter:
            _target_: sam3.train.transforms.filter_query_transforms.FilterEmptyTargets
    - _target_: sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries
      query_filter:
        _target_: sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut
        max_num_objects: ${{scratch.max_ann_per_img}}

  val_transforms:
    - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
      transforms:
        - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
          sizes: ${{scratch.resolution}}
          max_size:
            _target_: sam3.train.transforms.basic.get_random_resize_max_size
            size: ${{scratch.resolution}}
          square: true
          consistent_transform: False
        - _target_: sam3.train.transforms.basic_for_api.ToTensorAPI
        - _target_: sam3.train.transforms.basic_for_api.NormalizeAPI
          mean: ${{scratch.train_norm_mean}}
          std: ${{scratch.train_norm_std}}

  loss:
    _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
    matcher: ${{scratch.matcher}}
    o2m_weight: 2.0
    o2m_matcher:
      _target_: sam3.train.matcher.BinaryOneToManyMatcher
      alpha: 0.3
      threshold: 0.4
      topk: 4
    use_o2m_matcher_on_o2m_aux: false
    loss_fns_find:
      - _target_: sam3.train.loss.loss_fns.Boxes
        weight_dict:
          loss_bbox: 5.0
          loss_giou: 2.0
      - _target_: sam3.train.loss.loss_fns.IABCEMdetr
        weak_loss: False
        weight_dict:
          loss_ce: 20.0
          presence_loss: 20.0
        pos_weight: 10.0
        alpha: 0.25
        gamma: 2
        use_presence: True
        pos_focal: false
        pad_n_queries: 200
        pad_scale_pos: 1.0
    loss_fn_semantic_seg: null
    scale_by_find_batch_size: ${{scratch.scale_by_find_batch_size}}

scratch:
  enable_segmentation: False
  d_model: 256
  pos_embed:
    _target_: sam3.model.position_encoding.PositionEmbeddingSine
    num_pos_feats: ${{scratch.d_model}}
    normalize: true
    scale: null
    temperature: 10000

  use_presence_eval: True
  original_box_postprocessor:
    _target_: sam3.eval.postprocessors.PostProcessImage
    max_dets_per_img: -1
    use_original_ids: true
    use_original_sizes_box: true
    use_presence: ${{scratch.use_presence_eval}}

  matcher:
    _target_: sam3.train.matcher.BinaryHungarianMatcherV2
    focal: true
    cost_class: 2.0
    cost_bbox: 5.0
    cost_giou: 2.0
    alpha: 0.25
    gamma: 2
    stable: False
  scale_by_find_batch_size: True

  resolution: {RESOLUTION}
  consistent_transform: False
  max_ann_per_img: 200

  train_norm_mean: [0.5, 0.5, 0.5]
  train_norm_std: [0.5, 0.5, 0.5]
  val_norm_mean: [0.5, 0.5, 0.5]
  val_norm_std: [0.5, 0.5, 0.5]

  num_train_workers: 4
  num_val_workers: 0
  max_data_epochs: {NUM_EPOCHS}
  target_epoch_size: 1500
  hybrid_repeats: 1
  context_length: 2
  gather_pred_via_filesys: false

  lr_scale: {LEARNING_RATE_SCALE}
  lr_transformer: ${{times:8e-4,${{scratch.lr_scale}}}}
  lr_vision_backbone: ${{times:2.5e-4,${{scratch.lr_scale}}}}
  lr_language_backbone: ${{times:5e-5,${{scratch.lr_scale}}}}
  lrd_vision_backbone: 0.9
  wd: 0.1
  scheduler_timescale: 20
  scheduler_warmup: 20
  scheduler_cooldown: 20

  val_batch_size: 1
  collate_fn_val:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: shwd
    with_seg_masks: ${{scratch.enable_segmentation}}

  gradient_accumulation_steps: 1
  train_batch_size: 1
  collate_fn:
    _target_: sam3.train.data.collator.collate_fn_api
    _partial_: true
    repeats: ${{scratch.hybrid_repeats}}
    dict_key: all
    with_seg_masks: ${{scratch.enable_segmentation}}

trainer:
  _target_: sam3.train.trainer.Trainer
  skip_saving_ckpts: false
  empty_gpu_mem_cache_after_eval: True
  skip_first_val: True
  max_epochs: {NUM_EPOCHS}
  accelerator: cuda
  seed_value: 42
  val_epoch_freq: 5
  mode: train
  gradient_accumulation_steps: ${{scratch.gradient_accumulation_steps}}

  distributed:
    backend: nccl
    find_unused_parameters: True
    gradient_as_bucket_view: True

  loss:
    all: ${{shwd_train.loss}}
    default:
      _target_: sam3.train.loss.sam3_loss.DummyLoss

  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        limit_ids: ${{shwd_train.num_images}}
        transforms: ${{shwd_train.train_transforms}}
        load_segmentation: ${{scratch.enable_segmentation}}
        max_ann_per_img: 500000
        multiplier: 1
        max_train_queries: 50000
        max_val_queries: 50000
        training: true
        use_caching: False
        img_folder: {train_img_folder}
        ann_file: {train_ann_file}

      shuffle: True
      batch_size: ${{scratch.train_batch_size}}
      num_workers: ${{scratch.num_train_workers}}
      pin_memory: True
      drop_last: True
      collate_fn: ${{scratch.collate_fn}}

    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        load_segmentation: ${{scratch.enable_segmentation}}
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          category_chunk_size: 2
          _partial_: true
        img_folder: {test_img_folder}
        ann_file: {test_ann_file}
        transforms: ${{shwd_train.val_transforms}}
        max_ann_per_img: 100000
        multiplier: 1
        training: false

      shuffle: False
      batch_size: ${{scratch.val_batch_size}}
      num_workers: ${{scratch.num_val_workers}}
      pin_memory: True
      drop_last: False
      collate_fn: ${{scratch.collate_fn_val}}

  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${{paths.bpe_path}}
    device: cpus
    eval_mode: false
    enable_segmentation: ${{scratch.enable_segmentation}}

  meters:
    val:
      shwd:
        detection:
          _target_: sam3.eval.coco_writer.PredictionDumper
          iou_type: "bbox"
          dump_dir: ${{launcher.experiment_log_dir}}/dumps/shwd
          merge_predictions: True
          postprocessor: ${{scratch.original_box_postprocessor}}
          gather_pred_via_filesys: ${{scratch.gather_pred_via_filesys}}
          maxdets: 100
          pred_file_evaluators:
            - _target_: sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators
              gt_path: {test_ann_file}
              tide: False
              iou_type: "bbox"

  optim:
    amp:
      enabled: True
      amp_dtype: bfloat16  # A10 supports bfloat16

    optimizer:
      _target_: torch.optim.AdamW

    gradient_clip:
      _target_: sam3.train.optim.optimizer.GradientClipper
      max_norm: 0.1
      norm_type: 2

    param_group_modifiers:
      - _target_: sam3.train.optim.optimizer.layer_decay_param_modifier
        _partial_: True
        layer_decay_value: ${{scratch.lrd_vision_backbone}}
        apply_to: 'backbone.vision_backbone.trunk'
        overrides:
          - pattern: '*pos_embed*'
            value: 1.0

    options:
      lr:
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_transformer}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_vision_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.vision_backbone.*'
        - scheduler:
            _target_: sam3.train.optim.schedulers.InverseSquareRootParamScheduler
            base_lr: ${{scratch.lr_language_backbone}}
            timescale: ${{scratch.scheduler_timescale}}
            warmup_steps: ${{scratch.scheduler_warmup}}
            cooldown_steps: ${{scratch.scheduler_cooldown}}
          param_names:
            - 'backbone.language_backbone.*'

      weight_decay:
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: ${{scratch.wd}}
        - scheduler:
            _target_: fvcore.common.param_scheduler.ConstantParamScheduler
            value: 0.0
          param_names:
            - '*bias*'
          module_cls_names: ['torch.nn.LayerNorm']

  checkpoint:
    save_dir: ${{launcher.experiment_log_dir}}/checkpoints
    save_freq: 0

  logging:
    tensorboard_writer:
      _target_: sam3.train.utils.logger.make_tensorboard_logger
      log_dir: ${{launcher.experiment_log_dir}}/tensorboard
      flush_secs: 120
      should_log: True
    wandb_writer: null
    log_dir: ${{launcher.experiment_log_dir}}/logs
    log_freq: 10

launcher:
  num_nodes: 1
  gpus_per_node: {NUM_GPUS}
  experiment_log_dir: ${{paths.experiment_log_dir}}
  multiprocessing_context: forkserver

submitit:
  account: null
  partition: null
  qos: null
  timeout_hour: 72
  use_cluster: False
  cpus_per_task: 4
  port_range: [10000, 65000]
  constraint: null
"""

# Write config to the SAM3 configs directory so Hydra can find it
config_dir = os.path.join(SAM3_DIR, "sam3", "train", "configs", "custom")
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, "shwd_finetune.yaml")

with open(config_path, "w") as f:
    f.write(config_yaml)

# Also save a copy in our work dir for MLflow logging
config_copy = os.path.join(WORK_DIR, "shwd_finetune.yaml")
with open(config_copy, "w") as f:
    f.write(config_yaml)

print(f"Config written to: {config_path}")
print(f"Config copy: {config_copy}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Setup MLflow

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
except Exception as e:
    print(f"Warning: {e}")
    mlflow.set_experiment("/Users/0def019e-c076-4fbf-9ab5-6f12c4b9396e/sam31_finetune_fallback")

print(f"MLflow experiment set")
print(f"Registry: {mlflow.get_registry_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7b. Validate SAM3 imports
# MAGIC
# MAGIC Ensure all key SAM3 modules can be imported before starting training.
# MAGIC This catches missing dependencies early with clear error messages.

# COMMAND ----------

import importlib
import traceback as _tb

# Diagnostic: check SAM3 is on the path
print(f"SAM3_DIR: {SAM3_DIR}")
print(f"SAM3_DIR in sys.path: {SAM3_DIR in sys.path}")
print(f"SAM3_DIR exists: {os.path.exists(SAM3_DIR)}")
print(f"sam3/ subdir exists: {os.path.exists(os.path.join(SAM3_DIR, 'sam3'))}")

# Ensure SAM3 is on sys.path (belt and suspenders with pip install -e)
if SAM3_DIR not in sys.path:
    sys.path.insert(0, SAM3_DIR)
    print(f"Added {SAM3_DIR} to sys.path")

# Try basic import first with full traceback
try:
    import sam3
    print(f"sam3 package imported from: {sam3.__file__}")
except Exception as e:
    err_detail = _tb.format_exc()
    print(f"CRITICAL: Cannot import sam3: {err_detail}")
    # Write to UC volume for debugging
    with open("/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets/shwd/import_error_log.txt", "w") as f:
        f.write(f"SAM3_DIR: {SAM3_DIR}\n")
        f.write(f"sys.path: {sys.path}\n\n")
        f.write(err_detail)
    raise

targets_to_check = [
    "sam3.train.transforms.basic_for_api",
    "sam3.train.transforms.basic",
    "sam3.train.transforms.filter_query_transforms",
    "sam3.train.data.sam3_image_dataset",
    "sam3.train.data.torch_dataset",
    "sam3.train.data.collator",
    "sam3.train.data.coco_json_loaders",
    "sam3.train.loss.sam3_loss",
    "sam3.train.loss.loss_fns",
    "sam3.train.matcher",
    "sam3.train.trainer",
    "sam3.train.train",
    "sam3.model_builder",
    "sam3.eval.coco_writer",
    "sam3.eval.coco_eval_offline",
    "sam3.eval.postprocessors",
    "sam3.train.optim.optimizer",
    "sam3.train.optim.schedulers",
    "sam3.train.utils.logger",
    "sam3.model.position_encoding",
]

failed = []
for mod_name in targets_to_check:
    try:
        importlib.import_module(mod_name)
        print(f"  OK: {mod_name}")
    except Exception as e:
        detail = _tb.format_exc()
        print(f"  FAIL: {mod_name} -> {detail[-500:]}")
        failed.append((mod_name, detail[-300:]))

if failed:
    # Write detailed errors to UC volume for debugging
    err_lines = [f"SAM3_DIR: {SAM3_DIR}", f"sys.path: {sys.path}", ""]
    for name, err in failed:
        err_lines.append(f"--- {name} ---")
        err_lines.append(err)
        err_lines.append("")
    with open("/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets/shwd/import_error_log.txt", "w") as f:
        f.write("\n".join(err_lines))

    print(f"\n{len(failed)} modules failed to import!")
    # Include first error detail in the exception for visibility
    raise ImportError(f"SAM3 module import failures ({len(failed)}/{len(targets_to_check)}). First error: {failed[0][1]}")
else:
    print(f"\nAll {len(targets_to_check)} SAM3 modules imported successfully")

# Monkey-patch addmm_act: the original in sam3/perflib/fused.py is inference-only
# (raises ValueError when gradients are enabled). During training we need a
# fallback that uses standard PyTorch ops.
import sam3.perflib.fused as _fused
import sam3.model.vitdet as _vitdet

_original_addmm_act = _fused.addmm_act

def _addmm_act_train_compatible(activation, linear, mat1):
    if not torch.is_grad_enabled():
        return _original_addmm_act(activation, linear, mat1)
    # Fallback: standard linear + activation
    x = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        x = torch.nn.functional.gelu(x)
    elif activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        x = torch.nn.functional.relu(x)
    else:
        raise ValueError(f"Unexpected activation {activation}")
    return x

_fused.addmm_act = _addmm_act_train_compatible
_vitdet.addmm_act = _addmm_act_train_compatible
print("Patched addmm_act for training mode (grad-enabled fallback)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run finetuning
# MAGIC
# MAGIC Invoke `sam3/train/train.py` with our custom Hydra config.
# MAGIC The `-c` arg is a Hydra config name relative to `sam3/train/configs/`.

# COMMAND ----------

import random
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Use initialize_config_dir with an absolute path to our config dir
# This avoids the Hydra module-based config resolution issue
os.environ["HYDRA_FULL_ERROR"] = "1"

# Write error output to UC volume for debugging (accessible outside cluster)
error_log_path = "/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets/shwd/train_error_log.txt"

try:
    from sam3.train.utils.train_utils import register_omegaconf_resolvers
    register_omegaconf_resolvers()
except Exception as e:
    print(f"Note: {e}")

# Load the config using Hydra with absolute config dir path
sam3_config_dir = os.path.join(SAM3_DIR, "sam3", "train")

# We need to use initialize_config_dir for absolute paths
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

with initialize_config_dir(config_dir=sam3_config_dir, version_base="1.2"):
    cfg = compose(config_name="configs/custom/shwd_finetune")

print("Config loaded successfully!")
print("Config keys:", list(cfg.keys()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8b. Run the training loop inline

# COMMAND ----------

with mlflow.start_run(run_name="sam31_finetune_shwd", log_system_metrics=True) as run:
    mlflow.log_params({
        "model": HF_MODEL_ID,
        "dataset": DATASET_NAME,
        "num_epochs": NUM_EPOCHS,
        "lr_scale": LEARNING_RATE_SCALE,
        "resolution": RESOLUTION,
        "num_train_images": str(NUM_TRAIN_IMAGES),
        "num_gpus": NUM_GPUS,
    })
    mlflow.log_artifact(config_copy, "config")

    try:
        # Call single_proc_run directly instead of single_node_runner.
        # single_node_runner calls torch.multiprocessing.set_start_method("spawn")
        # which fails on Databricks because the context is already set by Spark.
        # For single-GPU training we can bypass it entirely.
        from sam3.train.train import single_proc_run
        main_port = random.randint(10000, 65000)

        cfg.launcher.num_nodes = 1
        cfg.launcher.gpus_per_node = NUM_GPUS
        cfg.submitit.use_cluster = False

        print(f"Starting training: {NUM_EPOCHS} epochs, {NUM_TRAIN_IMAGES} images, port={main_port}")
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=1)
        print("Training completed successfully!")

    except Exception as e:
        # Write error to UC volume for debugging
        import traceback
        error_msg = f"Training error:\n{traceback.format_exc()}"
        print(error_msg)
        with open(error_log_path, "w") as f:
            f.write(error_msg)
        raise

    # Log training outputs
    if os.path.exists(EXPERIMENT_DIR):
        mlflow.log_artifacts(EXPERIMENT_DIR, "experiment_outputs")

    # Log the best checkpoint
    ckpt_dir = os.path.join(EXPERIMENT_DIR, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if ckpts:
            best_ckpt = sorted(ckpts)[-1]
            mlflow.log_artifact(os.path.join(ckpt_dir, best_ckpt), "model/checkpoints")
            print(f"Logged checkpoint: {best_ckpt}")

    print(f"MLflow run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Register finetuned model in Unity Catalog

# COMMAND ----------

from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

client = MlflowClient()

ckpt_dir = os.path.join(EXPERIMENT_DIR, "checkpoints")
if os.path.exists(ckpt_dir):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if ckpts:
        best_ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])

        class Sam31FtWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                self.ckpt_path = context.artifacts["checkpoint"]
            def predict(self, context, model_input, params=None):
                return {"status": "Use SAM 3 native API with this checkpoint for inference."}

        signature = ModelSignature(
            inputs=Schema([ColSpec("string", "image_path"), ColSpec("string", "text_prompt")]),
            outputs=Schema([ColSpec("string", "status")]),
        )

        with mlflow.start_run(run_name="sam31_finetuned_registration") as reg_run:
            model_info = mlflow.pyfunc.log_model(
                artifact_path="sam3_1_finetuned",
                python_model=Sam31FtWrapper(),
                artifacts={"checkpoint": best_ckpt_path},
                signature=signature,
                registered_model_name=UC_MODEL_NAME,
                pip_requirements=["torch>=2.7", "torchvision", "timm>=1.0.17", "huggingface_hub", "pillow"],
            )

            client.set_registered_model_alias(
                name=UC_MODEL_NAME,
                alias="finetuned-detection",
                version=model_info.registered_model_version,
            )

            print(f"Registered: {UC_MODEL_NAME} v{model_info.registered_model_version}")
            print(f"Alias: finetuned-detection")
    else:
        print("No checkpoints found to register")
else:
    print(f"Checkpoint directory not found: {ckpt_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary

# COMMAND ----------

print("Notebook complete.")
print(f"  Dataset: {DATASET_NAME} ({DATASET_ROOT})")
print(f"  Epochs: {NUM_EPOCHS}, Train images: {NUM_TRAIN_IMAGES}")
print(f"  Model: {UC_MODEL_NAME}")
print(f"  Experiment: {MLFLOW_EXPERIMENT}")
print("Check MLflow for experiment results and registered model versions.")
