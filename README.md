# Computer Vision for Manufacturing

Databricks Asset Bundle (DAB) that provisions UC resources, deploys two apps —
an image **labeling** app (CV Explorer) and an inference / **detection** app
(CV Inspect) — and finetunes SAM 3.1 for manufacturing / safety inspection
use cases.

## Architecture

```
┌─────────────────────── TRAINING PIPELINE ────────────────────────┐
│                                                                   │
│  Raw datasets                                                     │
│       │ Extract (00/00a)                                          │
│       ▼                                                           │
│  Labeling volume                                                  │
│       │                                                           │
│       ▼                                                           │
│  CV Explorer App  ◄── label in UI                                 │
│       │ Import (05)                                               │
│       ▼                                                           │
│  Export COCO (06) ──► coco_datasets volume                        │
│       │                                                           │
│       ▼                                                           │
│  Finetune SAM 3.1 (03)                                            │
│       │                                                           │
│       ▼                                                           │
│  MLflow / UC Model Registry  ──►  Model Serving endpoint          │
│                                          │                        │
└──────────────────────────────────────────┼────────────────────────┘
                                           │
                                           ▼
┌──────────────────── INFERENCE / DETECTION ───────────────────────┐
│                                                                   │
│  Image or video upload      OR      UC Volume path                │
│              │                              │                     │
│              └──────────────┬───────────────┘                     │
│                             ▼                                     │
│                     CV Inspect App  (Plotly Dash)                 │
│                             │                                     │
│                             ├──► Finetuned detector endpoint     │
│                             │      (databricks_endpoint backend)  │
│                             │                                     │
│                             └──► Vision LLM via FM API           │
│                                    (vlm_proxy backend, fallback)  │
│                             │                                     │
│                             ▼                                     │
│                Annotated image  +  detection table                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

The **finetuned detector** registered in UC Model Registry is served via a
Model Serving endpoint and called from the **CV Inspect App**. The app's
detector layer is pluggable: until a finetuned endpoint is deployed, the
`vlm_proxy` backend routes inference through a vision LLM (Gemini 2.5 Pro by
default) using a structured-output prompt — so the UI is usable end-to-end
before finetuning completes. Switching to the real model is a one-line config
change in `app/config.py`.

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

# 2. Deploy all resources (schema, volumes, apps, jobs)
databricks bundle deploy -t dev

# 3. First-deploy: start the apps
databricks bundle run cv_explorer
databricks bundle run cv_manuf_inspect

# 4. Store the HuggingFace token (manual, one-time)
databricks secrets put-secret cv-manufacturing hf-token
```

## CV Inspect App — deployment & iteration

The CV Inspect app source lives in `app/` and is referenced by
`resources/cv_manuf_inspect.app.yml` via `source_code_path: ../app`. The
included `Makefile` wraps the deploy / iteration loop with consistent flags.

### First deploy (full bundle)

```bash
make deploy CATALOG=<your_catalog>          # creates schema, volumes, apps, jobs
make post-deploy CATALOG=<your_catalog>     # captures app SP id + grants USE_CATALOG
```

`post-deploy` writes the app's service principal id into
`.databricks/bundle/<target>/variable-overrides.json` (gitignored) and runs
the one-time `USE_CATALOG` grant. After this, the app SP has
`USE_CATALOG`, `USE_SCHEMA`, and `READ_VOLUME` on the configured catalog.

### Iterating on app code

`bundle deploy` runs Terraform under the hood, which is occasionally blocked
by a CLI bundled-binary GPG-key issue on older CLI versions. For source-only
edits (Python / CSS / HTML), use the direct Apps API path which skips
Terraform entirely:

```bash
make sync                                   # re-import app/ → workspace; redeploy app
make logs-url                               # print live /logz URL for stdout/stderr
```

`make sync` is what to use during development — it takes a few seconds and
hot-replaces only the app source.

### Configure model backends

Open `app/config.py` and edit the `MODELS` list. Each `ModelEntry` declares:

- `family`: `"vlm"` (vision LLM) or `"detector"`
- `backend` (detectors only): `"vlm_proxy"` (today, routed through a vision
  LLM with a structured-output prompt) or `"databricks_endpoint"` (a
  Databricks Model Serving endpoint returning bounding boxes)
- `backend_config`: classes, instructions, and either `vlm_endpoint` or
  `endpoint_name`

Switching a detector entry from `vlm_proxy` to `databricks_endpoint` is the
one-line change that points the UI at a finetuned model once it's served.

### Common tasks

| Task | Command |
|------|---------|
| Validate bundle | `make validate` |
| Full bundle deploy | `make deploy CATALOG=...` |
| Post-deploy grants | `make post-deploy CATALOG=...` |
| Iterate on app source only | `make sync` |
| View live logs URL | `make logs-url` |
| Grant catalog access (one-off) | `make grant-catalog CATALOG=...` |

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
  cv_explorer.app.yml             # Labeling app (GitHub git_source, tag-pinned)
  cv_manuf_inspect.app.yml        # Detection / inference app (./app source)
  jobs.yml                        # 6 job definitions
app/                              # CV Inspect Dash app source
  app.py, config.py, inference.py
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
