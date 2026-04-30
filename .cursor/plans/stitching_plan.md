# Stitching Plan — DAB bundle for db_cv_for_manuf

End goal: `databricks bundle deploy -t dev` provisions everything, then a sequence of
`databricks bundle run` commands brings up the full pipeline from raw data → labelled
projects → COCO splits → finetuned model.

---

## Phase 0 — Bundle skeleton

**Files created:**

- `databricks.yml`
- `resources/catalog.yml`

**What they do:**

`databricks.yml` declares:
- `bundle.name: db-cv-for-manuf`
- Two targets: `dev` (current workspace, manual triggers) and `prod` (pinned app tag, bigger GPU).
- Variables: `catalog` (default `brian_gen_ai`), `schema` (default `cv_manufacturing`),
  `hf_secret_scope` (default `cv-manufacturing`), `hf_secret_key` (default `hf-token`).
- `include:` pointing at `resources/*.yml`.
- `workspace.root_path` set to `/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}`.

`resources/catalog.yml` declares:
- Schema `${var.catalog}.${var.schema}` (assumes catalog already exists).
- Five UC volumes: `raw`, `labeling`, `imports`, `coco_datasets`, `exports`.
  All MANAGED under the schema.

**Validation gate:** `databricks bundle validate -t dev` passes.

**Deploy gate:** `databricks bundle deploy -t dev` creates the schema and volumes
(or reports them as already existing).

---

## Phase 1 — App resource

**Files created:**

- `resources/apps.yml`

**What it does:**

Declares the `cv_explorer` app using `git_repository` + `git_source`:

```yaml
resources:
  apps:
    cv_explorer:
      name: cv-explorer-${bundle.target}
      description: "Image labeling & annotation for CV manufacturing datasets"
      git_repository:
        url: https://github.com/Data-drone/db_image_labelling_app
      git_source:
        branch: main
        source_code_path: .
      config:
        env:
          - name: DEMO_VOLUME_PATH
            value: /Volumes/${var.catalog}/${var.schema}/labeling
          - name: LAKEBASE_PROJECT_ID
            value: cv-explorer
```

Not setting `LAKEBASE_AUTO_PROVISION` — the default `true` is what we want.
Not setting `PGUSER` — intentionally left for future OBO migration.

**Deploy gate:**

```bash
databricks bundle deploy -t dev
databricks bundle run cv_explorer      # first-deploy workaround (CLI #4181)
```

Verify: the app URL loads, the React UI shows the Projects page, Lakebase
auto-provisions (visible on the Admin page `/admin`).

---

## Phase 2 — Extract notebook (new: `00_extract_raw_to_labeling_volume.py`)

**File created:**

- `notebooks/00_extract_raw_to_labeling_volume.py`

**Parameters (dbutils.widgets):**

| Widget | Default | Purpose |
|--------|---------|---------|
| `catalog` | `brian_gen_ai` | UC catalog |
| `schema` | `cv_manufacturing` | UC schema |
| `dataset` | `shwd` | Which dataset to extract |

**Logic per dataset:**

### SHWD (`dataset=shwd`)
1. Read `/Volumes/{catalog}/{schema}/raw/shwd_safety_helmet/shwd_voc2028.zip`.
2. Extract JPEGs from `VOC2028/JPEGImages/` to `/Volumes/{catalog}/{schema}/labeling/shwd/`.
3. Flat copy — filenames are already unique (`000001.jpg` … `007581.jpg`).

### DeepPCB (`dataset=deeppcb`)
1. Read `/Volumes/{catalog}/{schema}/raw/deep_pcb_defects/`.
   Structure: `group{NNNNN}/{NNNNN}NNN_test.jpg` + `…_temp.jpg`.
2. Only copy `*_test.jpg` files (the defect images, not the template references).
3. Prefix filenames with group folder: `group00041_00041000_test.jpg`.
   This avoids duplicate basenames and satisfies the import API's uniqueness constraint.
4. Write to `/Volumes/{catalog}/{schema}/labeling/deeppcb/`.

### Corrosion (`dataset=corrosion`)
1. Read `/Volumes/{catalog}/{schema}/raw/corrosion_detection/`.
   Format: HuggingFace Parquet with embedded image bytes + bbox annotations.
2. Decode the `image` column (PIL bytes) and write each as `{index:06d}.jpg`
   to `/Volumes/{catalog}/{schema}/labeling/corrosion/`.
3. Filenames are sequential so already unique.

**Idempotency:** Skip files that already exist at destination (check by name).

**Validation gate:** `dbutils.fs.ls` on the labeling volume shows the expected
file counts per dataset (SHWD ~7.5K, DeepPCB ~1.5K test images, Corrosion ~9.2K).

---

## Phase 3 — Import notebook (new: `05_import_existing_annotations.py`)

**File created:**

- `notebooks/05_import_existing_annotations.py`

**Parameters (dbutils.widgets):**

| Widget | Default | Purpose |
|--------|---------|---------|
| `catalog` | `brian_gen_ai` | UC catalog |
| `schema` | `cv_manufacturing` | UC schema |
| `dataset` | `shwd` | Which dataset to import |
| `app_url` | `""` | App URL (auto-discovered if blank) |

**How it works — the general pattern for every dataset:**

1. **Discover or receive the app URL.** If `app_url` is blank, use the Databricks SDK
   `w.apps.get("cv-explorer-dev")` to find the active deployment URL.
   Get an SP token for auth: `w.tokens.create(...)` or use the notebook's own
   ambient credential with `dbutils.notebook.entry_point...`.

2. **Create the project via the app API.**
   `POST /api/projects` with `name`, `task_type=detection`,
   `class_list=[...]`, `source_volume=/Volumes/{catalog}/{schema}/labeling/{dataset}`.
   On 409 (already exists), `GET /api/projects`, filter by name, use existing ID.
   The POST auto-scans the volume and creates `project_samples` rows.

3. **Convert source annotations to COCO JSON** (in-memory or temp file), including
   `images[]` with `width` and `height` (the COCO adapter needs these for
   pixel→normalised conversion).

4. **Stage the COCO JSON** to `/Volumes/{catalog}/{schema}/imports/{dataset}/labels.json`.

5. **Call the import endpoint.**
   `POST /api/projects/{id}/import` with:
   ```json
   {
     "volume_path": "/Volumes/{catalog}/{schema}/imports/{dataset}/labels.json",
     "format": "coco",
     "on_missing_sample": "error",
     "on_existing_annotations": "replace"
   }
   ```
   `replace` makes re-runs idempotent.

6. **Print counters** from the response (`samples_touched`, `annotations_created`, etc.).

### Dataset-specific conversion logic:

#### SHWD
- Read VOC XML files from `/Volumes/{catalog}/{schema}/raw/shwd_safety_helmet/`.
- Parse `<object>/<bndbox>` for each XML.
- Classes: `["hat", "person"]` (or whatever the VOC labels are — verify from the data).
- Image dimensions from `<size>/<width>` and `<size>/<height>` in the XML.
- Emit standard COCO JSON: `images[]` + `annotations[]` + `categories[]`.
  Bbox format: `[xmin, ymin, width, height]` in absolute pixels (COCO convention).

#### DeepPCB
- Read `.txt` annotation files from the raw volume.
- Each line: `x1 y1 x2 y2 defect_type`.
- Classes: `["open", "short", "mousebite", "spur", "copper", "pin-hole"]`
  (category IDs 1–6 per the original dataset spec).
- Image dimensions: all DeepPCB images are 640×640.
- Filenames must match the prefixed names from Phase 2
  (e.g. `group00041_00041000_test.jpg`).

#### Corrosion
- Read the HuggingFace Parquet from the raw volume.
- Bbox columns: `objects.bbox` is `[[x_min, y_min, x_max, y_max], ...]`
  and `objects.categories` is `[category_id, ...]`.
- Classes: `["corrosion"]` (single class).
- Image dimensions from the `image` column (decode to PIL, read `.size`).
- Filenames must match the sequential names from Phase 2 (`000000.jpg`, …).

### Limits to be aware of:
- 200 MB file cap per import. SHWD ~120K annotations → roughly 15–20 MB of JSON. Fine.
- 500K items per request. All three datasets are well under.
- 2M annotations per request. SHWD has ~120K. Fine.

**Validation gate:** After running for SHWD, open the CV Explorer app, navigate to
the SHWD project → all images show as "labeled", annotation counts match expectations.

---

## Phase 4 — Parameterise existing notebooks

**Files modified:**

- `notebooks/02_log_sam31_to_mlflow.py`
- `notebooks/03_finetune_sam31_detection.py`

### 02 — Log SAM 3.1

Replace the hardcoded constants at the top:
```python
UC_CATALOG = "brian_gen_ai"
UC_SCHEMA = "cv_manufacturing"
```
with:
```python
dbutils.widgets.text("catalog", "brian_gen_ai")
dbutils.widgets.text("schema", "cv_manufacturing")
UC_CATALOG = dbutils.widgets.get("catalog")
UC_SCHEMA = dbutils.widgets.get("schema")
```

Same for `UC_MODEL_NAME` → derive from widgets.

### 03 — Finetune SAM 3.1

Replace hardcoded values with widgets:

| Constant | Widget name | Default |
|----------|-------------|---------|
| `UC_COCO_VOLUME` | (derived) | `/Volumes/{catalog}/{schema}/coco_datasets` |
| `DATASET_NAME` | `dataset_name` | `shwd` |
| `MLFLOW_EXPERIMENT` | `mlflow_experiment` | `/Users/${current_user}/cv_manufacturing/sam31_finetune` |
| `UC_MODEL_NAME` | (derived) | `{catalog}.{schema}.sam3_1` |

Add MLflow tags for lineage:
```python
mlflow.log_params({
    ...,
    "project_id": project_id,       # from widget or "n/a"
    "project_version": project_ver,  # from widget or "n/a"
})
```

Replace the three hardcoded error-log paths
(`/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets/shwd/...`) with paths
derived from the widget values.

### 04 — Convert to COCO

This notebook is kept as-is for the "bypass the app" path. It works for SHWD only.
Add a header comment noting that the preferred path is now Phase 3 (import via app)
+ Phase 5 (export from app → COCO splits). Only parameterise if time allows.

---

## Phase 5 — Bridge notebook (new: `06_app_export_to_coco_splits.py`)

**File created:**

- `notebooks/06_app_export_to_coco_splits.py`

**Parameters (dbutils.widgets):**

| Widget | Default | Purpose |
|--------|---------|---------|
| `catalog` | `brian_gen_ai` | UC catalog |
| `schema` | `cv_manufacturing` | UC schema |
| `project_name` | `shwd` | App project name to export |
| `app_url` | `""` | App URL (auto-discovered if blank) |
| `split_ratio` | `0.8` | Train fraction (rest is test) |
| `seed` | `42` | Random seed for split |

**Logic:**

1. **Trigger export via app API.**
   `POST /api/projects/{id}/export` with
   `{"export_volume": "/Volumes/{catalog}/{schema}/exports"}`.
   Capture the returned `export_path` (timestamped directory).

2. **Read the exported COCO JSON** from `{export_path}/annotations.json`.

3. **Split images** deterministically (seed=42, ratio from widget).
   Assign each image ID to train or test. Use a hash-based or shuffle-based split.

4. **Write two COCO JSONs:**
   - `/Volumes/{catalog}/{schema}/coco_datasets/{project_name}/train/_annotations.coco.json`
   - `/Volumes/{catalog}/{schema}/coco_datasets/{project_name}/test/_annotations.coco.json`
   Filter `images[]` and `annotations[]` per split. `categories[]` is identical in both.

5. **Copy (or symlink) images** into `train/` and `test/` subdirectories.
   The export already copied images to `{export_path}/images/`; we redistribute them.

This bridges the app's single-file export into the `train/test/_annotations.coco.json`
layout that `03_finetune_sam31_detection.py` expects (lines ~198–215).

**Validation gate:** The COCO JSON files load cleanly with `pycocotools`;
image counts match the split ratio; finetune notebook 03 can consume them directly.

---

## Phase 6 — Jobs

**File created:**

- `resources/jobs.yml`

Declares:

### `extract_raw`
- Notebook: `notebooks/00_extract_raw_to_labeling_volume.py`
- Parameters: `catalog`, `schema`, `dataset` (default `shwd`).
- Cluster: small (single node, no GPU).
- Schedule: manual.

### `import_existing_annotations`
- Notebook: `notebooks/05_import_existing_annotations.py`
- Parameters: `catalog`, `schema`, `dataset`, `app_url`.
- Cluster: small (single node, no GPU).
- **Depends on:** app must be deployed and reachable (not enforced by DAB — documented).
- Schedule: manual.

### `register_sam31`
- Notebook: `notebooks/02_log_sam31_to_mlflow.py`
- Parameters: `catalog`, `schema`.
- Cluster: GPU (A10 or better, for downloading the 3.3GB checkpoint).
- Schedule: manual / on-demand.

### `app_export_to_coco_splits`
- Notebook: `notebooks/06_app_export_to_coco_splits.py`
- Parameters: `catalog`, `schema`, `project_name`, `app_url`, `split_ratio`, `seed`.
- Cluster: small (single node, no GPU).
- Schedule: manual.

### `finetune_sam31`
- Notebook: `notebooks/03_finetune_sam31_detection.py`
- Parameters: `catalog`, `schema`, `dataset_name`, `mlflow_experiment`.
- Cluster: GPU (A10 24GB or better). Single node.
- Schedule: manual.

### `pipeline_label_to_model` (parent orchestrator)
- Tasks (sequential):
  1. `app_export_to_coco_splits`
  2. `finetune_sam31`
- This is the "label more, then retrain" loop. User labels in the app, triggers
  this job, gets an updated model.
- Schedule: manual.

**Also created:**

- `resources/experiment.yml` — MLflow experiment for finetune runs.
- `resources/secrets.yml` — declares the `cv-manufacturing` secret scope.
  The HF token value is **not** in the bundle; it must be `databricks secrets put-secret`
  manually after deploy.

**Validation gate:** `databricks bundle validate -t dev` passes with all resources.

**Deploy gate:** `databricks bundle deploy -t dev` creates all jobs.

---

## Phase 7 — README update

**File modified:**

- `README.md`

Add a "Deploy" section covering:

1. Prerequisites (CLI >= 0.290.0, HF token access to `facebook/sam3.1`).
2. `databricks bundle deploy -t dev`
3. First-deploy app workaround: `databricks bundle run cv_explorer`.
4. Manual step: `databricks secrets put-secret cv-manufacturing hf-token`.
5. Run order for initial setup:
   - `databricks bundle run extract_raw -- --param dataset=shwd`
   - `databricks bundle run import_existing_annotations -- --param dataset=shwd`
   - (Repeat for `deeppcb`, `corrosion`)
   - Verify in the app UI.
6. Run order for the retrain loop:
   - Label in the app.
   - `databricks bundle run pipeline_label_to_model -- --param project_name=shwd`
7. Known limitations: Lakebase invisible to bundle destroy, always-on Lakebase cost,
   no model-assisted pre-annotation.

---

## Execution order

| Step | Phase | Depends on | Est. effort |
|------|-------|------------|-------------|
| 1 | Phase 0: bundle skeleton | nothing | small — YAML only |
| 2 | Phase 1: app resource | Phase 0 deployed | small — YAML only |
| 3 | Phase 2: extract notebook | Phase 0 deployed (volumes exist) | medium — 3 dataset parsers |
| 4 | Phase 3: import notebook | Phase 1 deployed (app running), Phase 2 run | medium — 3 converters + API calls |
| 5 | Phase 4: parameterise 02/03 | nothing (can run in parallel with 3–4) | small — widget wiring |
| 6 | Phase 5: bridge notebook | Phase 1 deployed (app running) | medium — split logic + file copies |
| 7 | Phase 6: jobs YAML | Phases 2–5 notebooks exist | small — YAML only |
| 8 | Phase 7: README | everything | small — docs |

Phases 2, 3, and 4 can be developed in parallel. Phase 5 depends on
Phase 1 (app deployed) but not on Phases 2–3. Phase 6 is purely declarative
and can be written any time after the notebooks exist.

---

## What's NOT in scope

- Upstream PRs to the labelling app (everything works as-is).
- Delta-table image library from `docs/image_library_design.md` (future work;
  Lakehouse Sync from the app's Lakebase can populate it later).
- Model-assisted pre-annotation (§4.5 in the original plan).
- Segmentation labeling (§4.6).
- OBO / multi-identity auth (§4.12 / §7.1).
- `prod` target tuning beyond the variable overrides.
