# Image Library Design — Delta Tables for CV Training Data

Design document for a metadata-driven image library on Databricks, built on Unity Catalog volumes + Delta tables. The goal is a system that can ingest, tag, label, version, and serve images for model training and finetuning.

---

## High-Level Architecture

```
                    ┌─────────────────────────────────┐
                    │       Unity Catalog              │
                    │                                  │
  Ingest ──────►   │  Volumes (raw files)             │
                    │    /raw/         ← original data │
                    │    /processed/   ← resized/norm  │
                    │                                  │
                    │  Delta Tables (metadata)         │
                    │    images             ← registry │
                    │    tags               ← free-form│
                    │    classifications    ← img-level│
                    │    annotations        ← bbox det │
                    │    segmentation_masks ← pixel    │
                    │    label_tasks        ← queue    │
                    │    datasets           ← curated  │
                    │    dataset_images     ← splits   │
                    └─────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Label Studio     Training Job    MLflow Registry
        (annotate)       (consume)       (track lineage)
```

**Core principle:** Images live as files in UC Volumes. All metadata, tags, labels, and dataset membership live in Delta tables. This separates storage from governance and makes everything queryable, versionable, and auditable.

---

## 1. Schema Design

### 1.1 `images` — File Registry

The central table. One row per image file.

```sql
CREATE TABLE images (
  image_id        STRING NOT NULL,          -- UUID
  file_path       STRING NOT NULL,          -- Volume path: /Volumes/catalog/schema/vol/...
  original_name   STRING,                   -- Original filename before ingest
  source_dataset  STRING,                   -- e.g. "shwd", "deeppcb", "corrosion"
  source_url      STRING,                   -- Where the data originally came from

  -- Image properties
  width           INT,
  height          INT,
  channels        INT,                      -- 1=grayscale, 3=RGB, 4=RGBA
  format          STRING,                   -- "jpeg", "png", "tiff"
  file_size_bytes BIGINT,

  -- Thumbnails (optional — store as base64 or separate volume path)
  thumbnail_path  STRING,

  -- Ingest metadata
  ingested_at     TIMESTAMP,
  ingested_by     STRING,                   -- User or pipeline name
  checksum_sha256 STRING,                   -- Deduplication + integrity

  -- Soft delete
  is_active       BOOLEAN DEFAULT TRUE
)
USING DELTA
PARTITIONED BY (source_dataset)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'     -- Track row-level changes
);
```

**Why this matters:**
- `file_path` points to the actual file in a UC Volume — the image bytes never go in Delta
- `checksum_sha256` enables deduplication across sources (same image in Roboflow and GitHub)
- `source_dataset` partitioning keeps queries fast when filtering by origin
- Change Data Feed lets downstream systems react to new images

### 1.2 `tags` — Free-Form Tagging

Flexible key-value tags for organising images without rigid schema.

```sql
CREATE TABLE tags (
  image_id    STRING NOT NULL,
  tag_key     STRING NOT NULL,              -- e.g. "domain", "quality", "scene_type"
  tag_value   STRING NOT NULL,              -- e.g. "mining", "good", "outdoor"
  tagged_by   STRING,
  tagged_at   TIMESTAMP
)
USING DELTA;
```

Example tags:
- `domain`: "mining", "manufacturing", "construction"
- `quality`: "good", "blurry", "occluded", "dark"
- `scene_type`: "indoor", "outdoor", "aerial"
- `has_ppe`: "true", "false"
- `defect_present`: "true", "false"
- `review_status`: "pending", "approved", "rejected"

This approach is more flexible than adding columns. You can add new tag dimensions without schema migrations.

### 1.3 `classifications` — Image-Level Labels

For classification tasks (e.g. "is this image corroded?", "what type of defect?", "is PPE worn?"). One row per image-class assignment.

```sql
CREATE TABLE classifications (
  classification_id STRING NOT NULL,        -- UUID
  image_id        STRING NOT NULL,
  label_class     STRING NOT NULL,          -- e.g. "corroded", "good", "helmet_worn"
  taxonomy        STRING,                   -- Grouping: "defect_type", "ppe_status", "scene"

  -- Multi-class vs multi-label
  -- Multi-class: one row per image (mutually exclusive classes)
  -- Multi-label: multiple rows per image (independent labels)
  is_primary      BOOLEAN DEFAULT TRUE,     -- For multi-class: the winning label

  -- Soft labels / probabilities (optional — useful for distillation)
  probability     DOUBLE,                   -- Model confidence or annotator certainty

  -- Provenance
  annotation_source STRING,                 -- "original_dataset", "label_studio", "model_v2"
  annotated_by    STRING,                   -- Human annotator or model name
  annotated_at    TIMESTAMP,
  confidence      DOUBLE,                   -- Overall annotation confidence (0-1)
  is_verified     BOOLEAN DEFAULT FALSE     -- Human-reviewed flag
)
USING DELTA
PARTITIONED BY (taxonomy);
```

**Why a separate table from detection annotations?**
- Classification has no spatial component — mixing bbox columns with image-level labels creates confusion and wasted NULLs
- Partitioning by `taxonomy` keeps queries fast when you're working on one classification task
- Soft labels / probabilities are common in classification but rare in detection
- A single image can have labels from multiple taxonomies simultaneously (defect type AND severity AND ppe status)

Example classification use cases from our datasets:
- **Corrosion**: binary classification (corroded / not corroded)
- **DeepPCB**: defect type classification (open, short, mousebite, spur, copper, pin-hole)
- **SHWD**: PPE compliance classification (compliant / non-compliant)
- **General**: image quality classification (good / blurry / dark / occluded)

### 1.4 `annotations` — Bounding Boxes (Object Detection)

Stores object-level annotations for detection tasks. One row per detected object.

```sql
CREATE TABLE annotations (
  annotation_id   STRING NOT NULL,          -- UUID
  image_id        STRING NOT NULL,
  label_class     STRING NOT NULL,          -- e.g. "helmet", "no_helmet", "corrosion"

  -- Bounding box (normalised 0-1 relative to image dimensions)
  bbox_x          DOUBLE NOT NULL,          -- Top-left x
  bbox_y          DOUBLE NOT NULL,          -- Top-left y
  bbox_w          DOUBLE NOT NULL,          -- Width
  bbox_h          DOUBLE NOT NULL,          -- Height

  -- Provenance
  annotation_source STRING,                 -- "original_dataset", "label_studio", "model_v2"
  annotated_by    STRING,                   -- Human annotator or model name
  annotated_at    TIMESTAMP,
  confidence      DOUBLE,                   -- For model-generated annotations (0-1)
  is_verified     BOOLEAN DEFAULT FALSE     -- Human-reviewed flag
)
USING DELTA
PARTITIONED BY (label_class);
```

**Why normalised coordinates (0-1)?** Images may be resized/cropped during preprocessing. Normalised coords stay valid regardless of resolution.

### 1.5 `segmentation_masks` — Pixel-Level Annotations

For semantic and instance segmentation tasks. Masks are stored as files in a volume (not in Delta) since they're binary image data. The table tracks metadata and links to the mask files.

```sql
CREATE TABLE segmentation_masks (
  mask_id         STRING NOT NULL,          -- UUID
  image_id        STRING NOT NULL,
  label_class     STRING NOT NULL,          -- e.g. "corrosion", "crack", "weld_defect"
  mask_type       STRING NOT NULL,          -- "semantic", "instance", "panoptic"

  -- Mask storage (one of these will be populated)
  mask_file_path  STRING,                   -- Volume path to mask PNG/NPY file
  mask_rle        STRING,                   -- Run-length encoded mask (COCO-style, for smaller masks)
  polygon_points  ARRAY<DOUBLE>,            -- Polygon vertices [x1,y1,x2,y2,...] (normalised 0-1)

  -- Instance segmentation
  instance_id     INT,                      -- Distinguishes separate instances of same class

  -- Provenance
  annotation_source STRING,
  annotated_by    STRING,
  annotated_at    TIMESTAMP,
  confidence      DOUBLE,
  is_verified     BOOLEAN DEFAULT FALSE
)
USING DELTA
PARTITIONED BY (label_class);
```

Mask files live in a dedicated volume folder:
```
/Volumes/brian_gen_ai/cv_manufacturing/masks/
├── semantic/           ← Single-channel PNGs (pixel value = class ID)
│   └── {image_id}.png
└── instance/           ← Multi-channel or indexed PNGs
    └── {image_id}.png
```

**Why separate masks from detection annotations?**
- Mask data is fundamentally different — it's spatial (pixel-level) not geometric (boxes)
- Mask files can be large (same resolution as source image) and belong in volumes, not Delta
- Different tools consume masks differently (segmentation models expect mask images, not bbox coords)
- Some datasets have both detection boxes AND segmentation masks for the same objects — keeping them in separate tables avoids conflating the two

### 1.6 `datasets` — Curated Training Sets

Named, versioned collections of images for specific training runs.

```sql
CREATE TABLE datasets (
  dataset_id      STRING NOT NULL,          -- UUID
  dataset_name    STRING NOT NULL,          -- e.g. "helmet_detection_v3"
  description     STRING,
  task_type       STRING,                   -- "detection", "classification", "segmentation"
  label_classes   ARRAY<STRING>,            -- Classes included
  created_at      TIMESTAMP,
  created_by      STRING,
  is_frozen       BOOLEAN DEFAULT FALSE,    -- Lock after training starts
  mlflow_run_id   STRING                    -- Link to training run
)
USING DELTA;
```

### 1.7 `dataset_images` — Dataset Membership + Splits

```sql
CREATE TABLE dataset_images (
  dataset_id      STRING NOT NULL,
  image_id        STRING NOT NULL,
  split           STRING NOT NULL,          -- "train", "val", "test"
  assigned_at     TIMESTAMP
)
USING DELTA
PARTITIONED BY (dataset_id, split);
```

### 1.8 `label_tasks` — Labelling Queue

Tracks what needs human review.

```sql
CREATE TABLE label_tasks (
  task_id         STRING NOT NULL,
  image_id        STRING NOT NULL,
  task_type       STRING,                   -- "bbox", "classify", "verify", "segment"
  priority        INT DEFAULT 0,            -- Higher = more urgent
  status          STRING DEFAULT 'pending', -- "pending", "in_progress", "completed", "skipped"
  assigned_to     STRING,
  created_at      TIMESTAMP,
  completed_at    TIMESTAMP
)
USING DELTA;
```

---

## 2. Volume Layout

```
/Volumes/brian_gen_ai/cv_manufacturing/
├── raw/                          ← Original unmodified files
│   ├── shwd_safety_helmet/
│   ├── deep_pcb_defects/
│   └── corrosion_detection/
├── processed/                    ← Resized, normalised, augmented
│   ├── 512x512/                  ← Standard resolution
│   └── 224x224/                  ← Model input size
├── masks/                        ← Segmentation mask files
│   ├── semantic/                 ← Single-channel class masks
│   └── instance/                 ← Instance-level masks
├── thumbnails/                   ← Small previews for UI
│   └── 128x128/
└── exports/                      ← COCO/YOLO format exports for tools
```

**Why keep raw + processed separate?**
- Raw files are immutable — you never lose the original
- Processing is repeatable from raw if you change the pipeline
- Different model architectures need different input sizes

---

## 3. Ingestion Pipeline

### 3.1 Batch Ingest from Existing Datasets

For loading datasets like SHWD, DeepPCB, and Corrosion that we already have:

```python
# Pseudocode — runs as a Databricks notebook or DLT pipeline

def ingest_dataset(source_name, raw_volume_path, annotation_parser):
    """
    1. Walk files in the volume path
    2. Extract image metadata (dimensions, format, hash)
    3. Parse annotations using dataset-specific parser
    4. Write to images + annotations Delta tables
    """
    files = dbutils.fs.ls(raw_volume_path)

    for f in files:
        if is_image(f):
            img = Image.open(f.path)
            image_id = str(uuid4())

            # Register in images table
            spark.sql("""
              INSERT INTO images VALUES (
                :image_id, :file_path, :original_name,
                :source_dataset, :width, :height, ...
              )
            """)

            # Parse and insert annotations
            annotations = annotation_parser(f.path)
            for ann in annotations:
                spark.sql("""
                  INSERT INTO annotations VALUES (...)
                """)
```

Each source dataset needs its own annotation parser:
- **SHWD**: Parse Pascal VOC XML → bbox annotations with classes "hat"/"person"
- **DeepPCB**: Parse txt files → bbox annotations with 6 defect classes
- **Corrosion**: Read parquet objects column → bbox annotations

### 3.2 Streaming Ingest for New Data

Use Auto Loader to pick up new files dropped into a volume:

```python
# Auto Loader watches a volume path for new files
df = (spark.readStream
  .format("cloudFiles")
  .option("cloudFiles.format", "binaryFile")
  .option("cloudFiles.includeExistingFiles", "false")
  .load("/Volumes/brian_gen_ai/cv_manufacturing/raw/incoming/")
)

# Extract metadata and write to Delta
(df.writeStream
  .trigger(availableNow=True)
  .foreachBatch(process_and_register)
  .start()
)
```

---

## 4. Tagging & Labelling Workflow

### 4.1 Automated Tagging

Run classifiers to auto-tag images at scale:

```python
# Example: tag images by quality using a pretrained model
from pyspark.sql.functions import udf

@udf("string")
def assess_quality(image_bytes):
    """Classify image quality: good, blurry, dark, occluded"""
    img = decode(image_bytes)
    # Simple heuristics or pretrained model
    if laplacian_variance(img) < threshold:
        return "blurry"
    if mean_brightness(img) < dark_threshold:
        return "dark"
    return "good"

# Apply across all images using Spark
images_df = spark.read.format("binaryFile").load("/Volumes/.../raw/")
tagged = images_df.withColumn("quality", assess_quality("content"))
# Write tags to tags table
```

### 4.2 Label Studio Integration

For human labelling, export unlabelled images to Label Studio and import results back:

```
                                Export
    label_tasks  ──────────►  Label Studio
    (pending)                   (annotate)
                                   │
                                Import
    annotations  ◄──────────  Completed tasks
    (new rows)                (JSON export)
```

Workflow:
1. Query `label_tasks` for pending items
2. Generate a Label Studio project with image URLs (UC Volume paths served via a proxy/API)
3. Annotators work in Label Studio UI
4. Export completed annotations as JSON/COCO
5. Parse and insert into `annotations` table with `annotation_source = "label_studio"`
6. Update `label_tasks.status = "completed"`

### 4.3 Active Learning Loop

Use model uncertainty to prioritise labelling:

1. Train an initial model on existing annotations
2. Run inference on unlabelled images
3. Score each image by model uncertainty (entropy, margin sampling, etc.)
4. Insert high-uncertainty images into `label_tasks` with high priority
5. Human labels these → add to training set → retrain
6. Repeat

```python
# Pseudocode
predictions = model.predict(unlabelled_images)
uncertainties = compute_uncertainty(predictions)

# Send most uncertain to labelling queue
high_uncertainty = uncertainties.filter(col("entropy") > threshold)
high_uncertainty.select(
    lit(uuid4()).alias("task_id"),
    col("image_id"),
    lit("bbox").alias("task_type"),
    col("entropy").cast("int").alias("priority"),
    lit("pending").alias("status")
).write.mode("append").saveAsTable("label_tasks")
```

---

## 5. Dataset Versioning with Delta Time Travel

Delta's time travel is a killer feature for reproducible ML:

```sql
-- Create a frozen dataset for training
UPDATE datasets
SET is_frozen = TRUE
WHERE dataset_name = 'helmet_detection_v3';

-- Record the Delta version at freeze time
-- This lets you reproduce the exact training data later

-- Read the dataset as it was at a specific version
SELECT * FROM dataset_images VERSION AS OF 42
WHERE dataset_id = 'helmet_detection_v3';

-- Or by timestamp
SELECT * FROM annotations TIMESTAMP AS OF '2026-03-15T00:00:00'
WHERE image_id IN (SELECT image_id FROM dataset_images ...);
```

### Practical versioning strategy:

1. **Dataset creation** → record Delta table version numbers for `images`, `annotations`, `dataset_images`
2. **Log versions to MLflow** → tie exact data state to each training run
3. **Never mutate frozen datasets** → enforce via `is_frozen` flag
4. **Change Data Feed** → downstream dashboards react to new annotations

```python
import mlflow

with mlflow.start_run():
    # Log data version
    img_version = spark.sql("DESCRIBE HISTORY images").first().version
    ann_version = spark.sql("DESCRIBE HISTORY annotations").first().version

    mlflow.log_params({
        "dataset_name": "helmet_detection_v3",
        "images_delta_version": img_version,
        "annotations_delta_version": ann_version,
        "train_count": train_count,
        "val_count": val_count
    })

    # Train model...
    model = train(train_loader, val_loader)

    mlflow.log_metric("val_mAP", val_map)
    mlflow.pyfunc.log_model("model", python_model=model)
```

---

## 6. Serving Data for Training

The DataLoader pattern differs per task type. Below are examples for each.

### 6.1 Classification DataLoader

```python
from torch.utils.data import Dataset
from PIL import Image

class ClassificationDataset(Dataset):
    """Image classification from Delta — returns (image, label_index)."""

    def __init__(self, dataset_name, split, taxonomy, transform=None):
        self.df = spark.sql(f"""
            SELECT i.file_path,
                   c.label_class
            FROM dataset_images di
            JOIN images i ON di.image_id = i.image_id
            JOIN classifications c ON i.image_id = c.image_id
            WHERE di.dataset_id = '{dataset_name}'
              AND di.split = '{split}'
              AND c.taxonomy = '{taxonomy}'
              AND c.is_primary = TRUE
        """).toPandas()

        # Build class-to-index mapping
        self.classes = sorted(self.df['label_class'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['file_path']).convert('RGB')
        label = self.class_to_idx[row['label_class']]

        if self.transform:
            image = self.transform(image)

        return image, label
```

### 6.2 Object Detection DataLoader

```python
class DetectionDataset(Dataset):
    """Object detection from Delta — returns (image, target_dict)."""

    def __init__(self, dataset_name, split, transform=None):
        self.df = spark.sql(f"""
            SELECT i.file_path, i.width, i.height,
                   collect_list(struct(
                     a.label_class, a.bbox_x, a.bbox_y, a.bbox_w, a.bbox_h
                   )) as boxes
            FROM dataset_images di
            JOIN images i ON di.image_id = i.image_id
            LEFT JOIN annotations a ON i.image_id = a.image_id
            WHERE di.dataset_id = '{dataset_name}'
              AND di.split = '{split}'
            GROUP BY i.file_path, i.width, i.height
        """).toPandas()

        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['file_path']).convert('RGB')

        # Convert to format expected by torchvision detection models
        boxes = []
        labels = []
        for b in row['boxes']:
            if b['label_class'] is not None:
                # Convert normalised [x,y,w,h] to pixel [x1,y1,x2,y2]
                x1 = b['bbox_x'] * row['width']
                y1 = b['bbox_y'] * row['height']
                x2 = (b['bbox_x'] + b['bbox_w']) * row['width']
                y2 = (b['bbox_y'] + b['bbox_h']) * row['height']
                boxes.append([x1, y1, x2, y2])
                labels.append(b['label_class'])

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
```

### 6.3 Segmentation DataLoader

```python
import numpy as np

class SegmentationDataset(Dataset):
    """Semantic segmentation from Delta — returns (image, mask_tensor)."""

    def __init__(self, dataset_name, split, transform=None):
        self.df = spark.sql(f"""
            SELECT i.file_path,
                   first(sm.mask_file_path) as mask_path
            FROM dataset_images di
            JOIN images i ON di.image_id = i.image_id
            JOIN segmentation_masks sm ON i.image_id = sm.image_id
            WHERE di.dataset_id = '{dataset_name}'
              AND di.split = '{split}'
              AND sm.mask_type = 'semantic'
            GROUP BY i.file_path
        """).toPandas()

        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['file_path']).convert('RGB')

        # Load mask — single-channel PNG where pixel value = class ID
        mask = np.array(Image.open(row['mask_path']))

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
```

### 6.4 Multi-Task DataLoader

For models that do multiple tasks (e.g. classify + detect), query from all relevant tables:

```python
class MultiTaskDataset(Dataset):
    """Returns (image, {"classification": label, "boxes": [...], "mask": ...})"""

    def __init__(self, dataset_name, split, task_types, transform=None):
        # Base query: always start from dataset membership
        self.images = spark.sql(f"""
            SELECT di.image_id, i.file_path, i.width, i.height
            FROM dataset_images di
            JOIN images i ON di.image_id = i.image_id
            WHERE di.dataset_id = '{dataset_name}' AND di.split = '{split}'
        """).toPandas()

        self.task_types = task_types
        self.transform = transform

        # Pre-fetch task-specific data as lookups
        if "classification" in task_types:
            self.class_map = ...  # Query classifications table
        if "detection" in task_types:
            self.bbox_map = ...   # Query annotations table
        if "segmentation" in task_types:
            self.mask_map = ...   # Query segmentation_masks table
```

### 6.2 Mosaic StreamingDataset (for large-scale training)

For distributed training across multiple GPUs/nodes, convert Delta → MDS format:

```python
from streaming import MDSWriter

# Export a frozen dataset to MDS for efficient streaming
with MDSWriter(out="/Volumes/.../exports/mds/helmet_v3/", columns=...) as writer:
    for row in dataset_query.collect():
        writer.write({
            "image": open(row.file_path, "rb").read(),
            "boxes": serialize(row.boxes),
            "labels": row.labels
        })
```

---

## 7. Train/Val/Test Split Management

### 7.1 Stratified Splitting

```python
from sklearn.model_selection import train_test_split

# Get all images with their primary class
images_with_class = spark.sql("""
    SELECT i.image_id,
           first(a.label_class) as primary_class
    FROM images i
    JOIN annotations a ON i.image_id = a.image_id
    WHERE i.source_dataset = 'shwd'
    GROUP BY i.image_id
""").toPandas()

# Stratified split: 70/15/15
train_ids, temp_ids = train_test_split(
    images_with_class, test_size=0.3,
    stratify=images_with_class['primary_class'], random_state=42
)
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.5,
    stratify=temp_ids['primary_class'], random_state=42
)

# Insert into dataset_images
for split_name, split_df in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    spark.createDataFrame(split_df).select(
        lit(dataset_id).alias("dataset_id"),
        col("image_id"),
        lit(split_name).alias("split"),
        current_timestamp().alias("assigned_at")
    ).write.mode("append").saveAsTable("dataset_images")
```

### 7.2 Preventing Data Leakage

Critical for PCB-style datasets where multiple images come from the same physical item:

```sql
-- Group by source (e.g. PCB board ID) to prevent leakage
-- All images from the same board must be in the same split
SELECT source_group_id, split
FROM dataset_images di
JOIN images i ON di.image_id = i.image_id
GROUP BY source_group_id, split
HAVING count(DISTINCT split) > 1;  -- Should return 0 rows
```

---

## 8. Query Examples

### Classification queries

```sql
-- Class distribution for a classification dataset
SELECT c.label_class, di.split, count(*) as count
FROM dataset_images di
JOIN classifications c ON di.image_id = c.image_id
WHERE di.dataset_id = 'corrosion_binary_v1'
  AND c.taxonomy = 'corrosion_status'
GROUP BY c.label_class, di.split
ORDER BY di.split, count DESC;

-- Find images where model and human disagree (label review)
SELECT c_human.image_id, c_human.label_class as human_label,
       c_model.label_class as model_label, c_model.confidence
FROM classifications c_human
JOIN classifications c_model ON c_human.image_id = c_model.image_id
  AND c_human.taxonomy = c_model.taxonomy
WHERE c_human.annotation_source = 'label_studio'
  AND c_model.annotation_source LIKE 'model_%'
  AND c_human.label_class != c_model.label_class;

-- Multi-label: find images tagged with BOTH "corroded" and "outdoor"
SELECT i.image_id, i.file_path
FROM images i
JOIN classifications c1 ON i.image_id = c1.image_id
  AND c1.taxonomy = 'corrosion_status' AND c1.label_class = 'corroded'
JOIN classifications c2 ON i.image_id = c2.image_id
  AND c2.taxonomy = 'scene_type' AND c2.label_class = 'outdoor';
```

### Detection queries

```sql
-- Find all helmet images with high-quality verified annotations
SELECT i.file_path, a.label_class, a.bbox_x, a.bbox_y, a.bbox_w, a.bbox_h
FROM images i
JOIN annotations a ON i.image_id = a.image_id
JOIN tags t ON i.image_id = t.image_id AND t.tag_key = 'quality' AND t.tag_value = 'good'
WHERE a.label_class IN ('hat', 'person')
  AND a.is_verified = TRUE;

-- Detection dataset split statistics
SELECT di.split,
       count(DISTINCT di.image_id) as images,
       count(a.annotation_id) as annotations,
       count(DISTINCT a.label_class) as classes
FROM dataset_images di
JOIN annotations a ON di.image_id = a.image_id
WHERE di.dataset_id = 'helmet_detection_v3'
GROUP BY di.split;

-- Average objects per image (useful for anchor box tuning)
SELECT a.label_class,
       count(*) / count(DISTINCT a.image_id) as avg_objects_per_image,
       avg(a.bbox_w) as avg_width,
       avg(a.bbox_h) as avg_height
FROM annotations a
WHERE a.label_class IN ('hat', 'person')
GROUP BY a.label_class;
```

### Segmentation queries

```sql
-- Find images with both detection boxes and segmentation masks
SELECT i.image_id, i.file_path,
       count(DISTINCT a.annotation_id) as bbox_count,
       count(DISTINCT sm.mask_id) as mask_count
FROM images i
JOIN annotations a ON i.image_id = a.image_id
JOIN segmentation_masks sm ON i.image_id = sm.image_id
GROUP BY i.image_id, i.file_path;

-- Segmentation class pixel coverage (requires reading mask files, pseudocode)
-- Useful for checking class imbalance in segmentation datasets
SELECT sm.label_class, count(*) as mask_count
FROM segmentation_masks sm
WHERE sm.mask_type = 'semantic'
GROUP BY sm.label_class;
```

### Cross-cutting queries

```sql
-- Find unlabelled images for a given domain (no classification OR detection labels)
SELECT i.image_id, i.file_path
FROM images i
LEFT JOIN classifications c ON i.image_id = c.image_id
LEFT JOIN annotations a ON i.image_id = a.image_id
JOIN tags t ON i.image_id = t.image_id AND t.tag_key = 'domain' AND t.tag_value = 'mining'
WHERE c.classification_id IS NULL AND a.annotation_id IS NULL;

-- Annotation progress dashboard
SELECT lt.task_type, lt.status, count(*) as count
FROM label_tasks lt
GROUP BY lt.task_type, lt.status
ORDER BY lt.task_type, lt.status;

-- What label types exist for each image?
SELECT i.image_id,
       CASE WHEN count(DISTINCT c.classification_id) > 0 THEN TRUE ELSE FALSE END as has_classification,
       CASE WHEN count(DISTINCT a.annotation_id) > 0 THEN TRUE ELSE FALSE END as has_detection,
       CASE WHEN count(DISTINCT sm.mask_id) > 0 THEN TRUE ELSE FALSE END as has_segmentation
FROM images i
LEFT JOIN classifications c ON i.image_id = c.image_id
LEFT JOIN annotations a ON i.image_id = a.image_id
LEFT JOIN segmentation_masks sm ON i.image_id = sm.image_id
GROUP BY i.image_id;
```

---

## 9. Task Type Summary

| Task | Label Table | Label Granularity | Example |
|------|-------------|-------------------|---------|
| Classification | `classifications` | Image-level | "This image shows corrosion" |
| Object Detection | `annotations` | Object-level (bbox) | "Helmet at [x,y,w,h]" |
| Segmentation | `segmentation_masks` | Pixel-level (mask) | "These pixels are corrosion" |

A single image can participate in multiple task types simultaneously. For example, a corrosion image might have:
- A classification label: `corroded` (in `classifications`)
- Detection boxes around each corroded region (in `annotations`)
- A pixel-level mask of corroded areas (in `segmentation_masks`)

The `datasets` table's `task_type` field determines which label table(s) the DataLoader queries.

---

## 10. Implementation Phases

### Phase 1 — Foundation (now)
- Create Delta tables (`images`, `tags`, `classifications`, `annotations`)
- Write ingestion notebooks for SHWD, DeepPCB, Corrosion
- Register existing dataset files into the `images` table
- Parse and load existing annotations (detection boxes)
- Derive image-level classifications from existing annotations (e.g. SHWD → "has_helmet" / "no_helmet")

### Phase 2 — Classification Pipeline
- Build classification labelling workflow
- Automated classification via pretrained models (CLIP zero-shot, etc.)
- Create binary classification datasets (corroded/not, defect/clean, PPE/no-PPE)
- Train baseline classifiers and log to MLflow

### Phase 3 — Tagging & Curation
- Build automated quality tagging pipeline
- Create `datasets` and `dataset_images` tables
- Implement stratified train/val/test splitting per task type
- Build a dataset creation notebook

### Phase 4 — Labelling Pipeline
- Set up Label Studio (or Databricks-native UI)
- Build export/import notebooks for label tasks (classify, bbox, segment)
- Implement active learning loop with model uncertainty

### Phase 5 — Segmentation & Multi-Task
- Create `segmentation_masks` table and masks volume
- Build mask generation pipeline (from detection boxes or manual annotation)
- Multi-task DataLoaders for joint training
- MDS export for distributed training

### Phase 6 — Training Integration
- Task-specific PyTorch DataLoaders backed by Delta queries
- MLflow integration for data versioning + lineage
- Model registry with dataset version tracking

---

## 11. Open Questions

1. **Thumbnail storage** — Store as base64 in Delta (fast queries, larger tables) or as separate files in volumes (smaller tables, extra I/O)?
2. **Image embeddings** — Should we store CLIP/DINOv2 embeddings in Delta for similarity search and clustering? Could be a separate `embeddings` table with vector column. Useful for zero-shot classification and finding similar images for labelling.
3. **Multi-annotator agreement** — Do we need inter-annotator agreement tracking? (Multiple annotations per image from different annotators, with a consensus mechanism.)
4. **Access patterns** — Will training mostly happen on Databricks clusters (direct volume access) or external GPUs (need export/download)?
5. **Scale expectations** — Current datasets are ~18K images total. If this grows to 100K+ we may want to consider Photon-optimized tables and Z-ordering on frequently filtered columns.
6. **Classification taxonomy management** — Should taxonomies (class hierarchies, valid label values) be stored in a separate reference table? Useful for enforcing consistency and supporting hierarchical classification (e.g. defect → crack → fatigue_crack).
7. **Weak supervision / programmatic labelling** — For classification at scale, consider Snorkel-style labelling functions that generate noisy labels from heuristics, then combine them. This could rapidly bootstrap classification labels before investing in human annotation.
8. **Finetuning vs training from scratch** — Classification and segmentation will likely start from pretrained backbones (ImageNet, COCO). The dataset structure should support tracking which pretrained weights were used alongside the data version.
