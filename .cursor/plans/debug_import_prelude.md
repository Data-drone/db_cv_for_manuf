# Debug Prelude: SHWD Import — 0 samples after project creation

## What we're debugging

Notebook `05_import_existing_annotations.py` successfully:
1. Authenticated to the CV Explorer app via OAuth token exchange
2. Created project "SHWD Safety Helmets" (id=1) via `POST /api/projects`
3. Converted VOC XML → COCO JSON and staged to imports volume
4. Called `POST /api/projects/1/import` — got HTTP 200

But the project shows **0 samples and 0 labeled**. The import succeeded (no error), but the project has no images to annotate.

## Root cause hypothesis

The `POST /api/projects` call includes `source_volume` which triggers `scan_volume_for_samples()` in the app backend ([backend/volumes.py](https://github.com/Data-drone/db_image_labelling_app/blob/main/backend/volumes.py)). This scan populates `project_samples`. If the scan found 0 files, either:

1. **The app SP can't list the labeling volume** — the app SP (`1aa73fba-ab76-4925-a094-d151a7b4b7de`) was granted `READ_VOLUME` on `labeling` during this session, but the grant may not have propagated before the project was created.
2. **The source_volume path is wrong** — the notebook passes `f"{LABELING_VOLUME}/shwd"` which expands to `/Volumes/classic_stable_hdtwu7_catalog/dev_brian_law_cv_manufacturing/labeling/shwd`. The app's `scan_volume_for_samples` uses the Databricks SDK `w.files.list_directory_contents()` which needs the full path to work.
3. **The import returned 200 but skipped all rows** — if `on_missing_sample=skip` and all filenames in the COCO JSON don't match `project_samples` (because there are 0 samples), every row gets skipped silently.

Hypothesis 3 is most likely: 0 samples → import skips everything → 200 with all counters at 0.

## Fix strategy

1. **Delete the existing project** (id=1) via `DELETE /api/projects/1`
2. **Verify the app SP can list the labeling volume** by calling the app's browse endpoint or checking permissions
3. **Re-create the project** — this time `scan_volume_for_samples` should find the images
4. **Re-run the import** — now samples exist, filenames match, annotations land

Alternative: change `on_missing_sample` from `skip` to `create` in the import call. But this creates samples without verifying the images exist on disk, which is less clean.

## Deployed state

### Workspace
- **Host:** `https://fevm-classic-stable-hdtwu7.cloud.databricks.com`
- **CLI profile:** `fevm-classic-stable`
- **Catalog:** `classic_stable_hdtwu7_catalog`
- **Schema:** `dev_brian_law_cv_manufacturing` (dev-mode prefixed)

### App
- **Name:** `cv-explorer-dev`
- **URL:** `https://cv-explorer-dev-7474649809026315.aws.databricksapps.com`
- **SP UUID:** `1aa73fba-ab76-4925-a094-d151a7b4b7de`
- **SP name:** `app-1z1r8z cv-explorer-dev`
- **Status:** RUNNING, healthy (`/api/health` returns 200)
- **Source:** git_source pinned to tag `v0.0.2` of `Data-drone/db_image_labelling_app`
- **OAuth client ID:** `fb5dc2cc-cc79-4717-b2fd-cefa9753c0d5` (needed for token exchange)

### UC permissions already granted to the app SP
```sql
GRANT USE CATALOG ON CATALOG classic_stable_hdtwu7_catalog TO `1aa73fba-ab76-4925-a094-d151a7b4b7de`;
GRANT USE SCHEMA ON SCHEMA classic_stable_hdtwu7_catalog.dev_brian_law_cv_manufacturing TO `1aa73fba-ab76-4925-a094-d151a7b4b7de`;
GRANT READ_VOLUME ON VOLUME classic_stable_hdtwu7_catalog.dev_brian_law_cv_manufacturing.imports TO `1aa73fba-ab76-4925-a094-d151a7b4b7de`;
GRANT READ_VOLUME ON VOLUME classic_stable_hdtwu7_catalog.dev_brian_law_cv_manufacturing.labeling TO `1aa73fba-ab76-4925-a094-d151a7b4b7de`;
GRANT READ_VOLUME, WRITE_VOLUME ON VOLUME classic_stable_hdtwu7_catalog.dev_brian_law_cv_manufacturing.exports TO `1aa73fba-ab76-4925-a094-d151a7b4b7de`;
```

### Data on disk
- `/Volumes/.../raw/shwd_safety_helmet/shwd_voc2028.zip` — downloaded
- `/Volumes/.../raw/deep_pcb_defects/` — downloaded (group folders)
- `/Volumes/.../raw/corrosion_detection/train.parquet` — downloaded
- `/Volumes/.../labeling/shwd/` — ~7.5K JPEGs extracted
- `/Volumes/.../labeling/deeppcb/` — ~1.5K prefixed test JPEGs
- `/Volumes/.../labeling/corrosion/` — ~1.2K decoded JPEGs
- `/Volumes/.../imports/shwd/labels.json` — COCO JSON staged

### Existing project in app
- Project id=1, name="SHWD Safety Helmets", 0 samples, 0 labeled

### Auth pattern for notebook → app
Token exchange per https://docs.databricks.com/aws/en/dev-tools/databricks-apps/connect-local:
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
app_info = w.api_client.do("GET", f"/api/2.0/apps/{APP_NAME}")
APP_CLIENT_ID = app_info["oauth2_app_client_id"]

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_token = ctx.apiToken().get()
workspace_url = w.config.host.rstrip("/")

token_resp = requests.post(
    f"{workspace_url}/oidc/v1/token",
    data={
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": notebook_token,
        "subject_token_type": "urn:databricks:params:oauth:token-type:personal-access-token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "scope": "all-apis",
        "audience": APP_CLIENT_ID,
    },
)
APP_TOKEN = token_resp.json()["access_token"]
```

### Key files
- `databricks.yml` — bundle config with per-target catalog
- `resources/cv_explorer.app.yml` — app via git_source tag v0.0.2
- `resources/catalog.yml` — schema + 5 volumes
- `resources/jobs.yml` — 6 job definitions
- `notebooks/05_import_existing_annotations.py` — the notebook to debug
- `.cursor/plans/stitching_plan.md` — the full implementation plan

### What to do
1. Read `notebooks/05_import_existing_annotations.py` to understand current state
2. Read the upstream app's `scan_volume_for_samples` in `backend/volumes.py` (at `/tmp/db_image_labelling_app/` or re-clone)
3. Delete project 1, verify labeling volume is listable by the app, recreate, re-import
4. If the volume scan still fails, check the app logs: `databricks apps logs cv-explorer-dev --profile fevm-classic-stable`
