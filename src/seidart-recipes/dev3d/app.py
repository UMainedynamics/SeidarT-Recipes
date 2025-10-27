# app.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.io import FortranFile

import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ------- local file dialog (works when running Dash locally) -------
def browse_for_json() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.update()
        path = filedialog.askopenfilename(
            title="Select SeidarT JSON",
            filetypes=[("JSON files", "*.json")],
        )
        root.destroy()
        return path if path else None
    except Exception:
        # If tkinter not available (e.g., headless server), return None
        return None

# ------- helpers -------
def rgb_str_to_hex(s: str) -> str:
    r, g, b = [int(x) for x in s.split("/")]
    return f"#{r:02X}{g:02X}{b:02X}"

def load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)

def read_geometry_dat(dat_path: Path, nz: int, ny: int, nx: int, dtype=np.int32) -> np.ndarray:
    with FortranFile(dat_path, "r") as f:
        flat = f.read_record(dtype)
    return np.asarray(flat).reshape((nz, ny, nx), order="C")

def thicken_if_needed(vol: np.ndarray) -> np.ndarray:
    nz, ny, nx = vol.shape
    if ny == 1:
        vol = np.repeat(vol, 2, axis=1)
    if nx == 1:
        vol = np.repeat(vol, 2, axis=2)
    if nz == 1:
        vol = np.repeat(vol, 2, axis=0)
    return vol

def build_voigt6(row: dict) -> np.ndarray:
    M = np.zeros((6,6), dtype=float)
    for i in range(1,7):  # diagonals
        key = f"c{i}{i}"
        if key in row: M[i-1,i-1] = row[key]
    pairs = [(1,2),(1,3),(1,4),(1,5),(1,6),
             (2,3),(2,4),(2,5),(2,6),
             (3,4),(3,5),(3,6),
             (4,5),(4,6),(5,6)]
    for i,j in pairs:
        k = f"c{i}{j}"
        if k in row:
            M[i-1,j-1] = row[k]
            M[j-1,i-1] = row[k]
    return M

def table_from_matrix(mat: np.ndarray, rows, cols, title: str):
    df = [{"": rlab, **{clab: mat[i,j] for j,clab in enumerate(cols)}} for i, rlab in enumerate(rows)]
    return html.Div([
        html.H6(title, className="mt-2"),
        dash_table.DataTable(
            data=df,
            columns=[{"name": "", "id": ""}] + [{"name": c, "id": c} for c in cols],
            style_cell={"padding":"4px","fontFamily":"monospace","fontSize":12},
            style_table={"overflowX":"auto"},
        )
    ])

def table_3x3(vals: dict, prefix: str, title: str):
    M = np.array([
        [vals.get(f"{prefix}11",0.0), vals.get(f"{prefix}12",0.0), vals.get(f"{prefix}13",0.0)],
        [vals.get(f"{prefix}12",0.0), vals.get(f"{prefix}22",0.0), vals.get(f"{prefix}23",0.0)],
        [vals.get(f"{prefix}13",0.0), vals.get(f"{prefix}23",0.0), vals.get(f"{prefix}33",0.0)],
    ], dtype=float)
    return table_from_matrix(M, ["1","2","3"], ["1","2","3"], title)

def make_isosurface_traces(labels_zyx: np.ndarray, ids: list[int], colors: dict[int,str],
                           dx, dy, dz, other_opacity=0.2, active_id=None, shown_ids=None):
    z_coords, y_coords, x_coords = (
        np.arange(labels_zyx.shape[0]) * dz,
        np.arange(labels_zyx.shape[1]) * dy,
        np.arange(labels_zyx.shape[2]) * dx,
    )
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    val = labels_zyx.ravel()
    xx = xx.ravel(); yy = yy.ravel(); zz = zz.ravel()

    traces = []
    for mat_id in ids:
        visible = (shown_ids is None) or (mat_id in shown_ids)
        opacity = 1.0 if (active_id == mat_id) else other_opacity
        color = colors.get(mat_id, "#AAAAAA")
        traces.append(go.Isosurface(
            x=xx, y=yy, z=zz, value=val,
            isomin=mat_id - 0.49, isomax=mat_id + 0.49, surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False, opacity=opacity,
            colorscale=[[0, color], [1, color]],
            lighting=dict(ambient=0.35, diffuse=0.8, specular=0.2, roughness=0.8, fresnel=0.2),
            lightposition=dict(x=0, y=1000, z=1000),
            name=f"ID {mat_id}",
            visible=True if visible else "legendonly",
        ))
    return traces

# ------- Dash app -------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "SeidarT Voxel Viewer"

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([
            html.H4("SeidarT • Voxel Layers", className="mt-2"),
            dbc.InputGroup([
                dbc.Input(id="json-path", placeholder="Path to JSON (e.g., six_layer.json)", value=""),
                dbc.Button("Browse…", id="browse-btn", color="secondary"),
                dbc.Button("Load", id="load-btn", color="primary"),
            ], className="mb-2"),
            html.Div(id="load-status", className="text-muted"),
            html.Hr(),
            dbc.Label("Active Layer"),
            dcc.Dropdown(id="active-layer", options=[], value=None, clearable=True),
            dbc.Label("Show Layers"),
            dcc.Checklist(id="show-layers", options=[], value=[], inline=True),
            dbc.Label("Opacity for non-active layers"),
            dcc.Slider(id="other-opacity", min=0.05, max=0.8, step=0.05, value=0.2,
                       tooltip={"placement":"bottom"}),
        ], width=3),
        dbc.Col([ dcc.Graph(id="volume-fig", style={"height":"85vh"}) ], width=6),
        dbc.Col([
            html.Div([
                html.H5("Layer Info", className="mt-2"),
                html.Div(id="rhs-meta"),
                html.Hr(),
                html.H5("Tensors"),
                html.Div(id="rhs-tensors"),
            ])
        ], width=3),
    ], align="start"),
    dcc.Store(id="store-scene"),
])

# ------- callbacks -------

@app.callback(
    Output("json-path", "value"),
    Output("load-status", "children"),
    Input("browse-btn", "n_clicks"),
    prevent_initial_call=True
)
def browse_json(n):
    path = browse_for_json()
    if not path:
        return dash.no_update, "Browse canceled (or tkinter unavailable)."
    return path, f"Selected: {path}"

@app.callback(
    Output("store-scene", "data"),
    Output("load-status", "children"),
    Output("active-layer", "options"),
    Output("show-layers", "options"),
    Output("active-layer", "value"),
    Output("show-layers", "value"),
    Input("load-btn", "n_clicks"),
    State("json-path", "value"),
    prevent_initial_call=True
)
def load_scene(n, json_path):
    try:
        p = Path(json_path).expanduser().resolve()
        cfg = load_json(p)
        dom = cfg["Domain"]
        nx, ny, nz = int(dom["nx"]), int(dom["ny"]), int(dom["nz"])
        dx, dy, dz = float(dom["dx"]), float(dom["dy"]), float(dom["dz"])

        # **This is the bit you asked for**: use the JSON's folder for geometry.dat
        dat_path = p.parent / "geometry.dat"
        if not dat_path.exists():
            raise FileNotFoundError(f"geometry.dat not found next to JSON: {dat_path}")
        labels = read_geometry_dat(dat_path, nz=nz, ny=ny, nx=nx).astype(np.int32)
        labels_for_plot = thicken_if_needed(labels)

        materials = {m["id"]: m for m in cfg.get("Materials", [])}
        colors = {mid: rgb_str_to_hex(m["rgb"]) for mid, m in materials.items() if "rgb" in m}

        stiff_by_id = {row["id"]: row for row in cfg.get("Seismic", {}).get("Stiffness_Coefficients", [])}
        eps_by_id   = {row["id"]: row for row in cfg.get("Electromagnetic", {}).get("Permittivity_Coefficients", [])}
        sig_by_id   = {row["id"]: row for row in cfg.get("Electromagnetic", {}).get("Conductivity_Coefficients", [])}

        present_ids = sorted(list(np.unique(labels)))
        options = [{"label": f"{i} - {materials.get(i,{}).get('name','(unlabeled)')}", "value": i}
                   for i in present_ids]
        active_val = present_ids[0] if present_ids else None
        show_vals = present_ids

        scene = dict(
            cfg=cfg, json_path=str(p), dat_path=str(dat_path),
            nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
            labels=labels_for_plot, raw_labels=labels,
            materials=materials, colors=colors,
            stiff=stiff_by_id, eps=eps_by_id, sig=sig_by_id,
            present_ids=present_ids,
        )
        status = f"Loaded {p.name} and geometry.dat from {p.parent}"
        return scene, status, options, options, active_val, show_vals
    except Exception as e:
        return dash.no_update, f"Load error: {e}", [], [], None, []

@app.callback(
    Output("volume-fig", "figure"),
    Input("store-scene", "data"),
    Input("active-layer", "value"),
    Input("show-layers", "value"),
    Input("other-opacity", "value"),
    prevent_initial_call=True
)
def update_figure(scene, active_id, shown_ids, other_opacity):
    if not scene:
        return go.Figure()
    labels = np.array(scene["labels"])
    dx, dy, dz = scene["dx"], scene["dy"], scene["dz"]
    colors = scene["colors"]
    ids = scene["present_ids"]
    traces = make_isosurface_traces(labels, ids, colors, dx, dy, dz,
                                    other_opacity=other_opacity,
                                    active_id=active_id, shown_ids=shown_ids)
    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

@app.callback(
    Output("rhs-meta", "children"),
    Output("rhs-tensors", "children"),
    Input("store-scene", "data"),
    Input("active-layer", "value"),
    prevent_initial_call=True
)
def update_rhs(scene, active_id):
    if not scene or active_id is None:
        return html.Div("No layer selected."), html.Div()

    materials = scene["materials"]
    stiff = scene["stiff"]; eps = scene["eps"]; sig = scene["sig"]

    mat = materials.get(active_id, {})
    name = mat.get("name", "(unnamed)")
    rgb = mat.get("rgb", "—")
    hexcolor = rgb_str_to_hex(rgb) if isinstance(rgb, str) and "/" in rgb else "#888"

    meta = dbc.Card(dbc.CardBody([
        html.H6(f"ID {active_id}: {name}"),
        html.Div([html.Span("RGB: "), html.Span(rgb, style={"color":hexcolor,"fontWeight":"bold"})]),
        html.Div(f"Temperature: {mat.get('temperature','—')}"),
        html.Div(f"Density: {mat.get('density','—')}"),
        html.Div(f"Porosity: {mat.get('porosity','—')}"),
        html.Div(f"Water content: {mat.get('water_content','—')}"),
        html.Div(f"Anisotropic: {mat.get('is_anisotropic','—')}"),
        html.Div(f"Euler angles: {mat.get('euler_angles','—')}"),
    ]), className="mb-3")

    tensors = []
    if active_id in stiff:
        C = build_voigt6(stiff[active_id])
        tensors.append(table_from_matrix(C,
            ["11","22","33","23","13","12"],
            ["11","22","33","23","13","12"],
            "Stiffness C (Voigt 6×6)"))
    if active_id in eps:
        tensors.append(table_3x3(eps[active_id], "e", "Permittivity ε (3×3)"))
    if active_id in sig:
        tensors.append(table_3x3(sig[active_id], "s", "Conductivity σ (3×3)"))

    return meta, html.Div(tensors)

if __name__ == "__main__":
    app.run_server(debug=True)
