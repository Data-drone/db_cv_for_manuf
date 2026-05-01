"""CV Manufacturing Inspection app.

A Plotly Dash app that lets a user upload an image or video, pick a deployed
model (VLM today, finetuned detector when added to config.MODELS), and view
inference results inline. Calls Databricks serving endpoints directly — no
file-arrival job, no polling.
"""

from __future__ import annotations

import base64
import json
import os
import traceback
import uuid
from datetime import datetime, timezone

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

import config
import inference


def _decode_upload(contents: str) -> bytes:
    if not contents or "," not in contents:
        raise ValueError("Empty upload payload")
    _, b64 = contents.split(",", 1)
    return base64.b64decode(b64)


def _data_url(content_type: str, b: bytes) -> str:
    return f"data:{content_type};base64,{base64.b64encode(b).decode('ascii')}"


def _model_options(family: str):
    return [{"label": m.label, "value": m.id} for m in config.for_family(family)]


import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("[boot] cv-manuf-inspect starting", flush=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="CV Inspect — Databricks")
server = app.server


@server.route("/_diag")
def _diag():
    print("[diag] /_diag hit", flush=True)
    return "diag-ok", 200


def _feature_pill(icon: str, label: str) -> html.Span:
    return html.Span(
        [html.Span(icon, className="feature-pill-icon"), label],
        className="feature-pill",
    )


header = html.Div(
    [
        html.Div(
            [
                html.Div("DATABRICKS", className="brand-eyebrow"),
                html.H1("Computer Vision for Manufacturing", className="hero-title"),
                html.P(
                    "Run object detection on images and video using finetuned models or vision LLMs — all served from Databricks.",
                    className="hero-subtitle",
                ),
                html.Div(
                    [
                        _feature_pill("◆", "Finetuned detectors"),
                        _feature_pill("✦", "Vision LLMs"),
                        _feature_pill("▶", "Image & video"),
                        _feature_pill("⚡", "Real-time inference"),
                    ],
                    className="feature-pills",
                ),
            ],
            className="hero-inner",
        ),
    ],
    className="hero",
)


sidebar = dbc.Card(
    [
        dbc.CardHeader("Inspection setup"),
        dbc.CardBody(
            [
                html.Label("Input"),
                dcc.Tabs(
                    id="input-mode",
                    value="image",
                    children=[
                        dcc.Tab(label="Image", value="image"),
                        dcc.Tab(label="Video", value="video"),
                        dcc.Tab(label="Batch", value="batch"),
                    ],
                    className="mb-3",
                ),
                dcc.Tabs(
                    id="source",
                    value="upload",
                    children=[
                        dcc.Tab(label="Upload", value="upload", className="source-tab", selected_className="source-tab--selected"),
                        dcc.Tab(label="UC Volume", value="volume", className="source-tab", selected_className="source-tab--selected"),
                    ],
                    className="source-tabs mb-2",
                ),
                html.Div(
                    id="upload-controls",
                    children=[
                        dcc.Upload(
                            id="upload",
                            children=html.Div(["Drag & drop or ", html.A("browse")]),
                            style={
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "6px",
                                "textAlign": "center",
                                "padding": "20px",
                                "marginBottom": "12px",
                                "cursor": "pointer",
                            },
                            multiple=False,
                        ),
                    ],
                ),
                html.Div(
                    id="volume-controls",
                    children=[
                        dcc.Input(
                            id="volume-path",
                            type="text",
                            value="/Volumes/ramcar_motolite_catalog/cv_manufacturing/raw",
                            placeholder="/Volumes/<catalog>/<schema>/<volume>/<path>",
                            debounce=True,
                            className="form-control mb-2",
                            style={"width": "100%"},
                        ),
                        dcc.Loading(
                            type="dot",
                            children=dcc.Dropdown(
                                id="volume-file",
                                options=[],
                                placeholder="Pick a file to load…",
                                clearable=False,
                                className="mb-2",
                            ),
                        ),
                    ],
                    style={"display": "none"},
                ),
                html.Div(
                    id="batch-controls",
                    children=[
                        html.Label("Input folder", className="small"),
                        dcc.Input(
                            id="batch-input-path",
                            type="text",
                            value="/Volumes/ramcar_motolite_catalog/cv_manufacturing/raw",
                            placeholder="/Volumes/<catalog>/<schema>/<volume>/<folder>",
                            className="form-control mb-2",
                            style={"width": "100%"},
                        ),
                        html.Label("Output folder", className="small"),
                        dcc.Input(
                            id="batch-output-path",
                            type="text",
                            value="",
                            placeholder="auto-derived from input + run_id",
                            className="form-control mb-2",
                            style={"width": "100%"},
                        ),
                        html.Div(id="batch-status", className="text-muted small mb-2"),
                    ],
                    style={"display": "none"},
                ),
                html.Div(id="upload-status", className="text-muted small mb-3"),
                html.Hr(),
                html.Label("Model family"),
                dcc.RadioItems(
                    id="family",
                    options=[
                        {"label": " Finetuned detector", "value": "detector"},
                        {"label": " Vision LLM", "value": "vlm"},
                    ],
                    value="detector",
                    inline=True,
                    className="mb-2",
                ),
                html.Label("Model"),
                dcc.Dropdown(
                    id="model",
                    options=_model_options("detector"),
                    value=(config.for_family("detector")[0].id if config.for_family("detector") else None),
                    clearable=False,
                    className="mb-3",
                ),
                html.Div(
                    id="vlm-controls",
                    children=[
                        html.Label("Prompt"),
                        dcc.Textarea(
                            id="prompt",
                            value="Describe what you see. Identify any safety hazards (missing PPE, unsafe behaviors), defects, or notable features.",
                            style={"width": "100%", "height": 90},
                            className="mb-3",
                        ),
                    ],
                    style={"display": "none"},
                ),
                html.Div(
                    id="detector-controls",
                    children=[
                        html.Label("Confidence threshold"),
                        dcc.Slider(
                            id="threshold",
                            min=0.1, max=0.9, step=0.05, value=0.5,
                            marks={0.1: "0.1", 0.5: "0.5", 0.9: "0.9"},
                            className="mb-3",
                        ),
                    ],
                ),
                html.Div(
                    id="video-controls",
                    children=[
                        html.Label("Frame stride"),
                        dcc.Slider(
                            id="frame-stride",
                            min=5, max=120, step=5, value=30,
                            marks={5: "5", 30: "30", 60: "60", 120: "120"},
                            className="mb-2",
                        ),
                        html.Label("Max frames"),
                        dcc.Slider(
                            id="max-frames",
                            min=1, max=30, step=1, value=8,
                            marks={1: "1", 8: "8", 15: "15", 30: "30"},
                            className="mb-3",
                        ),
                    ],
                    style={"display": "none"},
                ),
                dbc.Button("Run inspection", id="run-btn", color="primary", className="w-100"),
                dbc.Button("Start batch job", id="batch-run-btn", color="primary", className="w-100", style={"display": "none"}),
            ]
        ),
    ]
)

results_panel = dbc.Card(
    [
        dbc.CardHeader("Result"),
        dbc.CardBody(
            [
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=html.Div(id="result", children=html.Div("Upload media + click Run.", className="text-muted")),
                ),
            ]
        ),
    ]
)

history_panel = dbc.Card(
    [
        dbc.CardHeader([
            "Recent batch runs",
            dbc.Button("Refresh", id="history-refresh-btn", color="secondary", size="sm",
                       className="float-end", style={"padding": "2px 10px", "fontSize": "11px"}),
        ]),
        dbc.CardBody(
            dcc.Loading(type="dot", children=html.Div(id="history-table",
                children=html.Div("Click Refresh to load.", className="text-muted small"))),
        ),
    ],
    className="mt-3",
    id="history-card",
    style={"display": "none"},
)

app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(sidebar, md=4),
                        dbc.Col([results_panel, history_panel], md=8),
                    ],
                    className="main-row",
                ),
                dcc.Store(id="upload-store"),
            ],
            fluid=True,
            className="main-container",
        ),
    ]
)


@app.callback(
    Output("vlm-controls", "style"),
    Output("detector-controls", "style"),
    Output("video-controls", "style"),
    Output("model", "options"),
    Output("model", "value"),
    Output("source", "style"),
    Output("upload-controls", "style", allow_duplicate=True),
    Output("volume-controls", "style", allow_duplicate=True),
    Output("batch-controls", "style"),
    Output("run-btn", "style"),
    Output("batch-run-btn", "style"),
    Output("history-card", "style"),
    Input("family", "value"),
    Input("input-mode", "value"),
    State("source", "value"),
    State("model", "value"),
    prevent_initial_call="initial_duplicate",
)
def _toggle_controls(family, input_mode, source_value, current_model):
    is_batch = input_mode == "batch"
    is_video = input_mode == "video"
    # Batch can only run detectors (no per-frame text aggregation makes sense for VLM batch)
    effective_family = "detector" if is_batch else family
    vlm_style = {} if (effective_family == "vlm" and not is_batch) else {"display": "none"}
    detector_style = {} if effective_family == "detector" else {"display": "none"}
    video_style = {} if is_video else {"display": "none"}
    source_style = {"display": "none"} if is_batch else {}
    upload_style = (
        {"display": "none"} if is_batch
        else ({} if source_value == "upload" else {"display": "none"})
    )
    volume_style = (
        {"display": "none"} if is_batch
        else ({} if source_value == "volume" else {"display": "none"})
    )
    batch_style = {} if is_batch else {"display": "none"}
    run_btn_style = {"display": "none"} if is_batch else {}
    batch_run_btn_style = {} if is_batch else {"display": "none"}
    history_style = {} if is_batch else {"display": "none"}
    opts = _model_options(effective_family)
    if not opts:
        return (vlm_style, detector_style, video_style, [], None,
                source_style, upload_style, volume_style, batch_style,
                run_btn_style, batch_run_btn_style, history_style)
    new_value = current_model if any(o["value"] == current_model for o in opts) else opts[0]["value"]
    return (vlm_style, detector_style, video_style, opts, new_value,
            source_style, upload_style, volume_style, batch_style,
            run_btn_style, batch_run_btn_style, history_style)


@app.callback(
    Output("upload-store", "data", allow_duplicate=True),
    Output("upload-status", "children", allow_duplicate=True),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def _store_upload(contents, filename):
    if not contents:
        return no_update, no_update
    try:
        b = _decode_upload(contents)
    except Exception as e:
        return None, f"Upload error: {e}"
    return {"contents": contents, "filename": filename, "size": len(b)}, f"Loaded {filename} ({len(b)//1024} KB)"


@app.callback(
    Output("upload-controls", "style", allow_duplicate=True),
    Output("volume-controls", "style", allow_duplicate=True),
    Input("source", "value"),
    State("input-mode", "value"),
    prevent_initial_call=True,
)
def _toggle_source(source, input_mode):
    if input_mode == "batch":
        return {"display": "none"}, {"display": "none"}
    return (
        {} if source == "upload" else {"display": "none"},
        {} if source == "volume" else {"display": "none"},
    )


_MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".mov", ".avi", ".mkv")


def _mime_for(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "bmp": "image/bmp", "webp": "image/webp",
        "mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo", "mkv": "video/x-matroska",
    }.get(ext, "application/octet-stream")


@app.callback(
    Output("volume-file", "options"),
    Output("upload-status", "children", allow_duplicate=True),
    Input("source", "value"),
    Input("volume-path", "value"),
    prevent_initial_call=True,
)
def _list_volume(source, path):
    if source != "volume":
        return no_update, no_update
    if not path or not path.startswith("/Volumes/"):
        return [], "Enter a /Volumes/... path."
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        path = path.rstrip("/")
        entries = list(w.files.list_directory_contents(path))
        names = sorted(
            e.name for e in entries
            if e.name and e.name.lower().endswith(_MEDIA_EXTENSIONS) and not getattr(e, "is_directory", False)
        )
        opts = [{"label": n, "value": f"{path}/{n}"} for n in names]
        msg = f"{len(opts)} files in {path}" if opts else f"No media files in {path}"
        return opts, msg
    except Exception as e:
        return [], f"Volume error: {e}"


@app.callback(
    Output("upload-store", "data", allow_duplicate=True),
    Output("upload-status", "children", allow_duplicate=True),
    Input("volume-file", "value"),
    prevent_initial_call=True,
)
def _load_from_volume(file_path):
    if not file_path:
        return no_update, no_update
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        dl = w.files.download(file_path)
        try:
            data = dl.contents.read()
        finally:
            try: dl.contents.close()
            except Exception: pass
        filename = file_path.rsplit("/", 1)[-1]
        mime = _mime_for(filename)
        b64 = base64.b64encode(data).decode()
        return (
            {"contents": f"data:{mime};base64,{b64}", "filename": filename, "size": len(data)},
            f"Loaded {filename} from volume ({len(data)//1024} KB)",
        )
    except Exception as e:
        return None, f"Volume load error: {e}"


def _ws_client():
    if not hasattr(_ws_client, "_w"):
        from databricks.sdk import WorkspaceClient
        _ws_client._w = WorkspaceClient()
    return _ws_client._w


def _query_batch_log(limit: int = 20) -> list[dict]:
    catalog = os.environ.get("CATALOG", "ramcar_motolite_catalog")
    schema = os.environ.get("SCHEMA", "cv_manufacturing")
    warehouse_id = os.environ.get("WAREHOUSE_ID", "")
    if not warehouse_id:
        raise RuntimeError("WAREHOUSE_ID env var not set — needed to query batch_run_log")
    sql = (
        f"SELECT run_id, started_at, completed_at, status, "
        f"input_path, output_path, model_label, image_count, detection_count "
        f"FROM {catalog}.{schema}.batch_run_log "
        f"ORDER BY started_at DESC LIMIT {limit}"
    )
    w = _ws_client()
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=warehouse_id, wait_timeout="30s"
    )
    if resp.status and str(resp.status.state) not in ("StatementState.SUCCEEDED", "SUCCEEDED"):
        raise RuntimeError(f"SQL state: {resp.status.state} | {resp.status.error}")
    rows = []
    if resp.result and resp.result.data_array:
        cols = [c.name for c in resp.manifest.schema.columns]
        for row in resp.result.data_array:
            rows.append(dict(zip(cols, row)))
    return rows


def _trigger_batch_job(input_path: str, output_path: str, model_id: str) -> int:
    job_id = os.environ.get("BATCH_JOB_ID", "")
    if not job_id:
        raise RuntimeError("BATCH_JOB_ID env var not set — bundle deploy must register the cv-manuf-batch-detect job")
    model = config.by_id(model_id)
    if model.family != "detector":
        raise RuntimeError("Batch jobs only support detector-family models.")
    cfg = model.backend_config
    catalog = os.environ.get("CATALOG", "ramcar_motolite_catalog")
    schema = os.environ.get("SCHEMA", "cv_manufacturing")
    params = {
        "catalog": catalog,
        "schema": schema,
        "input_path": input_path.rstrip("/"),
        "output_path": output_path.rstrip("/"),
        "vlm_endpoint": cfg.get("vlm_endpoint", "databricks-gemini-2-5-pro"),
        "classes": json.dumps(list(cfg.get("classes", []))),
        "instructions": cfg.get("instructions", ""),
        "model_label": model.label,
        "model_id": model.id,
        "threshold": "0.0",
        "max_files": "200",
    }
    w = _ws_client()
    run = w.jobs.run_now(job_id=int(job_id), notebook_params=params)
    return int(run.run_id)


def _detection_table(detections):
    if not detections:
        return html.Div("No detections.", className="text-muted small mt-2")
    rows = [
        html.Tr([
            html.Td(d.label),
            html.Td(f"{d.confidence:.0%}"),
            html.Td(", ".join(f"{v:.2f}" for v in d.bbox_norm), className="text-muted small"),
        ])
        for d in detections
    ]
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Label"), html.Th("Conf."), html.Th("Bbox (x1,y1,x2,y2)")])),
         html.Tbody(rows)],
        striped=True, bordered=True, size="sm", className="mt-3",
    )


@app.callback(
    Output("result", "children"),
    Input("run-btn", "n_clicks"),
    State("input-mode", "value"),
    State("upload-store", "data"),
    State("family", "value"),
    State("model", "value"),
    State("prompt", "value"),
    State("threshold", "value"),
    State("frame-stride", "value"),
    State("max-frames", "value"),
    prevent_initial_call=True,
)
def _run(n_clicks, input_mode, upload, family, model_id, prompt, threshold, frame_stride, max_frames):
    print(f"[run] n_clicks={n_clicks} mode={input_mode} family={family} model={model_id} threshold={threshold} stride={frame_stride}")
    if not n_clicks:
        return no_update
    if not upload:
        print("[run] no upload")
        return dbc.Alert("Upload an image or video first.", color="warning")
    if not model_id:
        print("[run] no model_id")
        return dbc.Alert("Pick a model.", color="warning")

    try:
        model = config.by_id(model_id)
        contents = upload["contents"]
        head, b64 = contents.split(",", 1)
        media_bytes = base64.b64decode(b64)
        mime = head.split(":", 1)[1].split(";", 1)[0] if ":" in head else "application/octet-stream"
        print(f"[run] dispatching: model={model.label} family={model.family} bytes={len(media_bytes)}")

        if input_mode == "image":
            result = inference.run_image(model, media_bytes, prompt, threshold=threshold)
            if model.family == "detector":
                annotated_src = f"data:image/jpeg;base64,{result.annotated_jpeg_b64}"
                debug = []
                if not result.detections and result.raw_response:
                    debug = [
                        html.Details([
                            html.Summary("Raw model output (no detections parsed)"),
                            html.Pre(result.raw_response, style={"fontSize": "11px", "whiteSpace": "pre-wrap", "maxHeight": "300px", "overflow": "auto"}),
                        ], className="mt-3"),
                    ]
                return html.Div([
                    html.Img(src=annotated_src, style={"maxWidth": "100%", "display": "block", "marginBottom": "16px"}),
                    html.H6(f"{model.label} — {len(result.detections)} detections"),
                    _detection_table(result.detections),
                    *debug,
                ])
            return html.Div([
                html.Img(src=_data_url(mime, media_bytes),
                         style={"maxWidth": "100%", "display": "block", "marginBottom": "16px"}),
                html.H6(model.label),
                dcc.Markdown(result),
            ])

        # video path
        results = inference.run_video(
            model, media_bytes, prompt, frame_stride=frame_stride, max_frames=max_frames, threshold=threshold,
        )
        if model.family == "detector":
            cards = []
            for r in results:
                src = f"data:image/jpeg;base64,{r.detection.annotated_jpeg_b64}"
                cards.append(html.Div([
                    html.Div(f"{r.timestamp_s:.1f}s · frame #{r.frame_index} · {len(r.detection.detections)} detections",
                             className="text-muted small mb-1"),
                    html.Img(src=src, style={"maxWidth": "100%", "marginBottom": "10px"}),
                ], className="mb-3"))
            return html.Div([
                html.H6(f"{model.label} — {len(results)} frames"),
                html.Div(cards),
            ])

        rows = [
            html.Tr([
                html.Td(f"{r.timestamp_s:.1f}s"),
                html.Td(f"#{r.frame_index}"),
                html.Td(dcc.Markdown(r.text)),
            ])
            for r in results
        ]
        return html.Div([
            html.Video(src=_data_url(mime, media_bytes), controls=True, style={"maxWidth": "100%"}),
            html.H6(f"{model.label} — {len(results)} frames", className="mt-3"),
            dbc.Table(
                [html.Thead(html.Tr([html.Th("Time"), html.Th("Frame"), html.Th("Output")])),
                 html.Tbody(rows)],
                striped=True, bordered=True, size="sm",
            ),
        ])

    except Exception:
        tb = traceback.format_exc()
        print(f"[run] EXCEPTION:\n{tb}")
        return dbc.Alert([html.B("Error: "), html.Pre(tb)], color="danger")


@app.callback(
    Output("batch-output-path", "value", allow_duplicate=True),
    Input("batch-input-path", "value"),
    State("batch-output-path", "value"),
    prevent_initial_call=True,
)
def _autoderive_output_path(input_path, current_output):
    if current_output:
        return no_update
    if not input_path or not input_path.startswith("/Volumes/"):
        return no_update
    parts = input_path.rstrip("/").split("/")
    if len(parts) < 5:
        return no_update
    catalog, schema, volume = parts[2], parts[3], parts[4]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"/Volumes/{catalog}/{schema}/{volume}/_outputs/{ts}"


@app.callback(
    Output("result", "children", allow_duplicate=True),
    Output("history-table", "children", allow_duplicate=True),
    Output("batch-status", "children"),
    Input("batch-run-btn", "n_clicks"),
    State("batch-input-path", "value"),
    State("batch-output-path", "value"),
    State("model", "value"),
    prevent_initial_call=True,
)
def _start_batch(n_clicks, input_path, output_path, model_id):
    if not n_clicks:
        return no_update, no_update, no_update
    if not input_path or not output_path or not model_id:
        return no_update, no_update, dbc.Alert("Set input path, output path, and model.", color="warning")
    if not output_path.startswith("/Volumes/"):
        return no_update, no_update, dbc.Alert("Output path must start with /Volumes/", color="warning")
    try:
        run_id = _trigger_batch_job(input_path, output_path, model_id)
        print(f"[batch] triggered job_run_id={run_id}", flush=True)
        banner = dbc.Alert([
            html.B("Batch job submitted. "),
            f"Databricks job run_id={run_id}. ",
            html.Small("Click Refresh in Recent batch runs to see status."),
        ], color="success")
        # also refresh history immediately
        try:
            rows = _query_batch_log()
            history = _render_history_table(rows)
        except Exception as e:
            history = html.Div(f"history error: {e}", className="text-muted small")
        return banner, history, ""
    except Exception:
        tb = traceback.format_exc()
        print(f"[batch] EXCEPTION:\n{tb}", flush=True)
        return no_update, no_update, dbc.Alert([html.B("Error: "), html.Pre(tb, style={"fontSize": "11px"})], color="danger")


def _render_history_table(rows):
    if not rows:
        return html.Div("No runs yet.", className="text-muted small")
    headers = ["Started", "Completed", "Status", "Model", "Images", "Detections", "Output", ""]
    body_rows = []
    for r in rows:
        run_id = r.get("run_id", "")
        started = (r.get("started_at") or "")[:19].replace("T", " ")
        completed = (r.get("completed_at") or "—")
        if completed and completed != "—":
            completed = completed[:19].replace("T", " ")
        status = r.get("status", "?")
        status_color = {"SUCCESS": "success", "RUNNING": "warning", "FAILED": "danger"}.get(status, "secondary")
        body_rows.append(html.Tr([
            html.Td(started, style={"fontSize": "11px"}),
            html.Td(completed, style={"fontSize": "11px"}),
            html.Td(dbc.Badge(status, color=status_color)),
            html.Td(r.get("model_label") or "?", style={"fontSize": "12px"}),
            html.Td(r.get("image_count") or 0),
            html.Td(r.get("detection_count") or 0),
            html.Td(html.Code(r.get("output_path", "")[-50:], style={"fontSize": "10px"})),
            html.Td(dbc.Button("View ▸", id={"type": "view-run-btn", "run_id": run_id},
                               n_clicks=0, color="secondary", size="sm",
                               style={"fontSize": "11px", "padding": "2px 8px"})),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(h) for h in headers])), html.Tbody(body_rows)],
        striped=True, bordered=True, size="sm", className="mb-0", responsive=True,
    )


@app.callback(
    Output("history-table", "children", allow_duplicate=True),
    Input("history-refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _refresh_history(n_clicks):
    if not n_clicks:
        return no_update
    try:
        rows = _query_batch_log()
        return _render_history_table(rows)
    except Exception as e:
        return dbc.Alert(f"History query failed: {e}", color="danger")


@app.callback(
    Output("result", "children", allow_duplicate=True),
    Input({"type": "view-run-btn", "run_id": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _view_run(n_clicks_list):
    triggered = dash.callback_context.triggered_id
    if not triggered:
        return no_update
    if not any(n_clicks_list or []):
        return no_update
    run_id = triggered.get("run_id")
    if not run_id:
        return no_update
    try:
        # Re-query the row to get its output_path
        rows = _query_batch_log(limit=200)
        match = next((r for r in rows if r.get("run_id") == run_id), None)
        if not match:
            return dbc.Alert(f"Run {run_id} not found in log.", color="warning")
        output_path = match.get("output_path", "").rstrip("/")
        if not output_path:
            return dbc.Alert("Run has no output_path.", color="warning")
        w = _ws_client()
        annotated_dir = f"{output_path}/annotated"
        try:
            entries = list(w.files.list_directory_contents(annotated_dir))
        except Exception as e:
            return dbc.Alert(f"Could not list {annotated_dir}: {e}", color="danger")
        files = sorted([e.name for e in entries if e.name and not getattr(e, "is_directory", False)])
        thumbs = []
        for name in files[:24]:
            try:
                dl = w.files.download(f"{annotated_dir}/{name}")
                data = dl.contents.read()
                try: dl.contents.close()
                except Exception: pass
                b64 = base64.b64encode(data).decode()
                thumbs.append(html.Div([
                    html.Img(src=f"data:image/jpeg;base64,{b64}",
                             style={"width": "100%", "borderRadius": "6px"}),
                    html.Div(name, className="text-muted small mt-1", style={"fontSize": "10px", "wordBreak": "break-all"}),
                ], style={"width": "32%", "marginBottom": "12px"}))
            except Exception:
                continue
        header = html.Div([
            html.H6(f"Run {run_id[:8]}… — {match.get('model_label')}"),
            html.Div([
                f"{match.get('image_count')} images · {match.get('detection_count')} detections · ",
                html.Code(output_path, style={"fontSize": "11px"}),
            ], className="text-muted small mb-3"),
        ])
        gallery_note = html.Div(f"Showing {len(thumbs)} of {len(files)} annotated images",
                                className="text-muted small mb-2") if len(files) > len(thumbs) else None
        return html.Div([
            header,
            gallery_note,
            html.Div(thumbs, style={"display": "flex", "flexWrap": "wrap", "gap": "1%"}),
        ])
    except Exception:
        return dbc.Alert([html.B("View error: "), html.Pre(traceback.format_exc(), style={"fontSize": "11px"})], color="danger")


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("PORT", "8050")))
    app.run(host=host, port=port, debug=os.environ.get("DEBUG") == "1")
