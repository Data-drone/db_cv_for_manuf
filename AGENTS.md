# Project: db_cv_for_manuf

Computer vision for manufacturing — Databricks Asset Bundle (DAB) that provisions UC resources, deploys a labeling app, runs SAM 3.1 finetuning, and serves DINOv3 + SAM 3.1 models via GPU endpoints.

## Architecture

- **Bundle config:** `databricks.yml` + `resources/*.yml`. Two targets: `dev` and `prod`.
- **Notebooks:** `notebooks/*.py` — Databricks notebooks (`# MAGIC` cells, `dbutils.widgets`, `%pip`). Not standalone Python scripts.
- **App:** The `cv_explorer` app deploys from GitHub via `git_repository` + `git_source` (pinned tag). There is no local app code in this repo — the source lives at `https://github.com/Data-drone/db_image_labelling_app`. App changes belong in that upstream repo.
- **UC volumes:** `raw`, `labeling`, `imports`, `coco_datasets`, `exports` under `${var.catalog}.${var.schema}`.

## Conventions

- All notebooks use `dbutils.widgets` for parameterisation (`catalog`, `schema`, `dataset`, etc.) with sensible defaults matching `databricks.yml` variable defaults.
- Bundle variables are referenced as `${var.catalog}`, `${var.schema}` etc. in YAML resources. Never hardcode catalog/schema in resource definitions.
- Volume paths follow `/Volumes/{catalog}/{schema}/{volume_name}/...`.
- COCO JSON is the interchange format for annotations (images[], annotations[], categories[]).
- Notebooks are idempotent: skip existing files, use `on_existing_annotations: replace` for imports.
- App deployment is tag-pinned (currently `v0.0.2`). Bump the tag in `resources/cv_explorer.app.yml` when adopting a new app release.

## Tech stack

- **Bundle:** Databricks Asset Bundles (CLI >= 0.290.0)
- **Notebooks:** PySpark, Pillow, pycocotools, HuggingFace `datasets`/`transformers`, MLflow
- **App (upstream):** FastAPI + SQLAlchemy backend, React 19 + Vite + Tailwind frontend, Lakebase Postgres, react-konva for annotation canvas
- **ML:** SAM 3.1 (segment-anything-model), finetuned for object detection; DINOv3 (ViT-L/16) for image embeddings
- **Serving:** DINOv3 embedder + SAM 3.1 segmentation on GPU_MEDIUM (A10G) via Databricks Model Serving
- **Vector Search:** Databricks Vector Search endpoint for image similarity

## Key paths

| Path | Purpose |
|------|---------|
| `databricks.yml` | Bundle root config, variables, targets |
| `resources/catalog.yml` | Schema + UC volume definitions |
| `resources/cv_explorer.app.yml` | App resource (GitHub `git_source`, tag-pinned) |
| `resources/setup.yml` | Setup job (model serving, vector search, optional data download) |
| `resources/jobs.yml` | Job definitions (import, export, finetune, pipeline) |
| `notebooks/00a_download_raw_datasets.py` | Download raw datasets to `raw` volume |
| `notebooks/00_extract_raw_to_labeling_volume.py` | Extract flat JPEGs to `labeling` volume |
| `notebooks/03_finetune_sam31_detection.py` | Finetune SAM 3.1 on COCO detection data |
| `notebooks/04_convert_to_coco_and_finetune.py` | VOC→COCO conversion (legacy, SHWD-only) |
| `notebooks/05_import_existing_annotations.py` | Import annotations into CV Explorer via app API |
| `notebooks/06_app_export_to_coco_splits.py` | Export from app → train/test COCO splits |
| `notebooks/07_setup_vector_search.py` | Create/verify Vector Search endpoint |
| `notebooks/dinov3_serving.py` | Log DINOv3 embedder to UC + deploy serving endpoint |
| `notebooks/sam3_serving.py` | Log SAM 3.1 serving model to UC + deploy serving endpoint |

## Datasets

Three manufacturing/safety CV datasets:
- **SHWD** — safety helmet wearing detection (~7.5K images, VOC XML annotations, classes: hat/person)
- **DeepPCB** — PCB defect detection (~1.5K images, 640×640, 6 defect classes)
- **Corrosion** — corrosion detection (~840 images, HuggingFace Parquet with bbox annotations)

## Working with this repo

- Validate bundle changes: `databricks bundle validate -t dev`
- Deploy: `databricks bundle deploy -t dev`
- HF token is a manual secret: `databricks secrets put-secret cv-manufacturing hf-token`
- The plan in `.cursor/plans/stitching_plan.md` documents the original integration phases (all complete).
