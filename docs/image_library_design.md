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
                    │    images        ← file registry │
                    │    tags          ← free-form     │
                    │    annotations   ← bbox/segment  │
                    │    label_tasks   ← review queue  │
                    │    datasets      ← curated sets  │
                    │    dataset_images← membership    │
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

### 1.3 `annotations` — Bounding Boxes & Segmentation

Stores object-level annotations for detection/segmentation tasks.

```sql
CREATE TABLE annotations (
  annotation_id   STRING NOT NULL,          -- UUID
  image_id        STRING NOT NULL,
  label_class     STRING NOT NULL,          -- e.g. "helmet", "no_helmet", "corrosion"

  -- Bounding box (normalised 0-1 relative to image dimensions)
  bbox_x          DOUBLE,                   -- Top-left x
  bbox_y          DOUBLE,                   -- Top-left y
  bbox_w          DOUBLE,                   -- Width
  bbox_h          DOUBLE,                   -- Height

  -- Segmentation (optional)
  segmentation    ARRAY<DOUBLE>,            -- Polygon points [x1,y1,x2,y2,...]
  mask_rle        STRING,                   -- Run-length encoded mask

  -- Classification (for image-level labels, no bbox needed)
  is_image_level  BOOLEAN DEFAULT FALSE,

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

### 1.4 `datasets` — Curated Training Sets

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

### 1.5 `dataset_images` — Dataset Membership + Splits

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

### 1.6 `label_tasks` — Labelling Queue

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

### 6.1 Building a PyTorch/Lightning DataLoader

```python
from torch.utils.data import Dataset
from PIL import Image

class DeltaImageDataset(Dataset):
    """Read images + annotations from Delta tables."""

    def __init__(self, dataset_name, split, transform=None):
        # Query Delta for this dataset's images
        self.df = spark.sql(f"""
            SELECT i.file_path, i.width, i.height,
                   collect_list(struct(a.label_class, a.bbox_x, a.bbox_y, a.bbox_w, a.bbox_h)) as boxes
            FROM dataset_images di
            JOIN images i ON di.image_id = i.image_id
            LEFT JOIN annotations a ON i.image_id = a.image_id
            WHERE di.dataset_id = '{dataset_name}'
              AND di.split = '{split}'
            GROUP BY i.file_path, i.width, i.height
        """).toPandas()

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['file_path'])
        boxes = row['boxes']

        if self.transform:
            image, boxes = self.transform(image, boxes)

        return image, boxes
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

```sql
-- Find all helmet images with high-quality annotations
SELECT i.file_path, a.label_class, a.bbox_x, a.bbox_y, a.bbox_w, a.bbox_h
FROM images i
JOIN annotations a ON i.image_id = a.image_id
JOIN tags t ON i.image_id = t.image_id AND t.tag_key = 'quality' AND t.tag_value = 'good'
WHERE a.label_class IN ('hat', 'person')
  AND a.is_verified = TRUE;

-- Dataset statistics
SELECT di.split,
       count(DISTINCT di.image_id) as images,
       count(a.annotation_id) as annotations,
       count(DISTINCT a.label_class) as classes
FROM dataset_images di
JOIN annotations a ON di.image_id = a.image_id
WHERE di.dataset_id = 'helmet_detection_v3'
GROUP BY di.split;

-- Find unlabelled images for a given domain
SELECT i.image_id, i.file_path
FROM images i
LEFT JOIN annotations a ON i.image_id = a.image_id
JOIN tags t ON i.image_id = t.image_id AND t.tag_key = 'domain' AND t.tag_value = 'mining'
WHERE a.annotation_id IS NULL;

-- Annotation progress dashboard
SELECT lt.status, count(*) as count
FROM label_tasks lt
GROUP BY lt.status;
```

---

## 9. Implementation Phases

### Phase 1 — Foundation (now)
- Create Delta tables (`images`, `tags`, `annotations`)
- Write ingestion notebooks for SHWD, DeepPCB, Corrosion
- Register existing dataset files into the `images` table
- Parse and load existing annotations

### Phase 2 — Tagging & Curation
- Build automated quality tagging pipeline
- Create `datasets` and `dataset_images` tables
- Implement stratified train/val/test splitting
- Build a dataset creation notebook

### Phase 3 — Labelling Pipeline
- Set up Label Studio (or Databricks-native UI)
- Build export/import notebooks for label tasks
- Implement active learning loop with model uncertainty

### Phase 4 — Training Integration
- PyTorch DataLoader backed by Delta queries
- MLflow integration for data versioning + lineage
- MDS export for distributed training
- Model registry with dataset version tracking

---

## 10. Open Questions

1. **Thumbnail storage** — Store as base64 in Delta (fast queries, larger tables) or as separate files in volumes (smaller tables, extra I/O)?
2. **Image embeddings** — Should we store CLIP/DINOv2 embeddings in Delta for similarity search and clustering? Could be a separate `embeddings` table with vector column.
3. **Multi-annotator agreement** — Do we need inter-annotator agreement tracking? (Multiple annotations per image from different annotators, with a consensus mechanism.)
4. **Access patterns** — Will training mostly happen on Databricks clusters (direct volume access) or external GPUs (need export/download)?
5. **Scale expectations** — Current datasets are ~18K images total. If this grows to 100K+ we may want to consider Photon-optimized tables and Z-ordering on frequently filtered columns.
