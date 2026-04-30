# Computer Vision for Manufacturing

Databricks Asset Bundle (DAB) that provisions UC resources, deploys an image
labeling app, and finetunes SAM 3.1 for manufacturing / safety inspection
use cases.

## Architecture

```
Raw datasets ──► Extract (00/00a) ──► Labeling volume
                                          │
                                          ▼
                                    CV Explorer App ◄── Label in UI
                                          │
                                          ▼
                               Import annotations (05)
                                          │
                                          ▼
                              Export COCO splits (06) ──► coco_datasets volume
                                                              │
                                                              ▼
                                                    Finetune SAM 3.1 (03)
                                                              │
                                                              ▼
                                                    MLflow / UC model registry
```

## Datasets

| Dataset | Use case | Images | Format | Licence |
|---------|----------|--------|--------|---------|
| **SHWD** | Safety helmet detection | ~7.5K | VOC XML | MIT |
| **DeepPCB** | PCB defect inspection | ~1.5K | TXT bboxes (640×640) | MIT |
| **Corrosion** | Infrastructure corrosion | ~840 | HuggingFace Parquet | CC BY 4.0 |

## Prerequisites

- Databricks CLI >= 0.290.0
- A workspace with Unity Catalog enabled
- HuggingFace token with access to `facebook/sam3.1` (for finetuning)

## Deploy

```bash
# 1. Validate the bundle
databricks bundle validate -t dev

# 2. Deploy all resources (schema, volumes, app, jobs)
databricks bundle deploy -t dev

# 3. First-deploy: start the app (CLI workaround for initial deploy)
databricks bundle run cv_explorer

# 4. Store the HuggingFace token (manual, one-time)
databricks secrets put-secret cv-manufacturing hf-token
```

## Initial data load

Run in order — each step depends on the previous:

```bash
# Download raw datasets to the raw volume
databricks bundle run extract_raw -- --param dataset=all

# Import existing annotations into the app (one per dataset)
databricks bundle run import_annotations -- --param dataset=shwd
databricks bundle run import_annotations -- --param dataset=deeppcb
databricks bundle run import_annotations -- --param dataset=corrosion
```

After import, open the CV Explorer app UI to verify projects show the expected
sample counts and annotations.

## Retrain loop

Once you have labeled (or re-labeled) data in the app, run the end-to-end
pipeline to export COCO splits and finetune:

```bash
# One-shot: export from app → train/test splits → finetune
databricks bundle run pipeline_label_to_model -- --param project_name="SHWD Safety Helmets"
```

Or run the steps individually:

```bash
# Export labeled data from the app into COCO train/test splits
databricks bundle run export_to_coco_splits -- --param project_name="SHWD Safety Helmets"

# Register SAM 3.1 base model in MLflow (first time only)
databricks bundle run register_sam31

# Finetune SAM 3.1 on the exported splits
databricks bundle run finetune_sam31 -- --param dataset_name=shwd_safety_helmets
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `00a_download_raw_datasets.py` | Download raw datasets to `raw` volume |
| `00_extract_raw_to_labeling_volume.py` | Extract flat JPEGs to `labeling` volume |
| `02_log_sam31_to_mlflow.py` | Register SAM 3.1 in MLflow / UC |
| `03_finetune_sam31_detection.py` | Finetune SAM 3.1 on COCO detection data |
| `05_import_existing_annotations.py` | Import annotations into CV Explorer via API |
| `06_app_export_to_coco_splits.py` | Export from app → train/test COCO splits |

## Bundle structure

```
databricks.yml                    # Bundle root config, variables, targets
resources/
  catalog.yml                     # Schema + 5 UC volumes
  cv_explorer.app.yml             # App resource (GitHub git_source, tag-pinned)
  jobs.yml                        # 6 job definitions
notebooks/                        # Databricks notebooks (see table above)
```

## Targets

| Target | Workspace | Catalog |
|--------|-----------|---------|
| `dev` | FEVM Classic Stable | `classic_stable_hdtwu7_catalog` |
| `prod` | e2-demo-field-eng | `brian_gen_ai` |

## Known limitations

- The CV Explorer app uses Lakebase (auto-provisioned). Lakebase resources are
  not destroyed by `databricks bundle destroy` — delete manually if needed.
- The app service principal needs UC permissions (`USE CATALOG`, `READ_VOLUME`,
  `WRITE_VOLUME`) which may take ~60s to propagate after initial grant.
- No model-assisted pre-annotation (future work).
- Finetuning requires a GPU node (A10 24GB or better).

## Licence

See [LICENSE](LICENSE) for repository licence.
