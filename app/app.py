"""CV Manufacturing Inspection app.

A Plotly Dash app that lets a user upload an image or video, pick a deployed
model (VLM today, finetuned detector when added to config.MODELS), and view
inference results inline. Calls Databricks serving endpoints directly — no
file-arrival job, no polling.
"""

from __future__ import annotations

import base64
import os
import traceback

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

app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(sidebar, md=4),
                        dbc.Col(results_panel, md=8),
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
    Input("family", "value"),
    Input("input-mode", "value"),
    State("model", "value"),
)
def _toggle_controls(family, input_mode, current_model):
    vlm_style = {} if family == "vlm" else {"display": "none"}
    detector_style = {} if family == "detector" else {"display": "none"}
    video_style = {} if input_mode == "video" else {"display": "none"}
    opts = _model_options(family)
    if not opts:
        return vlm_style, detector_style, video_style, [], None
    new_value = current_model if any(o["value"] == current_model for o in opts) else opts[0]["value"]
    return vlm_style, detector_style, video_style, opts, new_value


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
    Output("upload-controls", "style"),
    Output("volume-controls", "style"),
    Input("source", "value"),
)
def _toggle_source(source):
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


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8050"))
    app.run(host=host, port=port, debug=os.environ.get("DEBUG") == "1")
