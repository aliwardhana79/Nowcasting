"""Surabaya Radar Nowcast Lab — Streamlit dashboard.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import csv
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from nowcast import geodata, plotting, radar as radar_core, store, viz
from nowcast.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_FSS_SCALES,
    DEFAULT_THRESHOLDS,
    MAP_THRESHOLDS,
    NowcastConfig,
)
from nowcast.models import NowcastResult
from nowcast.pipeline import run_demo_nowcast, run_radar_nowcast, workload_estimate

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Dr. AW Radar Lab",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

SOURCE_RADAR = "Radar archive"
SOURCE_DEMO = "Synthetic demo"

CSS = """
<style>
:root {
  --ink: #0f172a;
  --muted: #475569;
  --line: #e2e8f0;
  --brand: #0e7490;
  --brand-deep: #082f49;
}
.block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f8fafc 0%, #eef7f9 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}
h1, h2, h3, h4 {letter-spacing: -0.015em; color: var(--ink);}
p, li {color: #1e293b;}
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.85rem 1rem;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
  height: 100%;
}
div[data-testid="stMetric"] label p {
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  font-weight: 700;
}
div[data-testid="stMetricValue"] {font-size: 1.55rem !important; font-weight: 750; color: var(--ink);}
button[data-baseweb="tab"] {font-weight: 650; font-size: 0.93rem;}
button[data-baseweb="tab"][aria-selected="true"] {color: var(--brand);}
div.stButton > button[kind="primary"] {
  background: #0e7490; border: 1px solid #0e7490; font-weight: 700;
}
div.stButton > button[kind="primary"]:hover {
  background: #155e75; border-color: #155e75;
}
.stDownloadButton button {font-weight: 600;}
hr {margin: 1.1rem 0;}

.hero {
  position: relative; overflow: hidden; border-radius: 20px;
  padding: 2.1rem 2.2rem 2rem;
  background:
    radial-gradient(circle at 88% 45%, rgba(255,255,255,0.10) 0 1px, transparent 2px),
    linear-gradient(125deg, #082f49 0%, #0c4a6e 55%, #0e7490 100%);
  color: white;
  box-shadow: 0 18px 45px rgba(8, 47, 73, 0.20);
  margin-bottom: 1.2rem;
}
.hero:before {
  content: ""; position: absolute; right: -140px; top: -180px;
  width: 540px; height: 540px; border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.16);
  box-shadow: 0 0 0 55px rgba(255,255,255,0.045),
              0 0 0 110px rgba(255,255,255,0.032),
              0 0 0 165px rgba(255,255,255,0.022);
}
.hero .eyebrow {
  position: relative; font-size: 0.72rem; font-weight: 800;
  letter-spacing: 0.16em; text-transform: uppercase; color: #a5f3fc;
}
.hero h1 {
  position: relative; color: white; font-size: 2.25rem; line-height: 1.1;
  margin: 0.55rem 0 0.65rem; font-weight: 800; max-width: 900px;
}
.hero p {
  position: relative; color: #dbeafe; max-width: 780px;
  line-height: 1.55; margin: 0; font-size: 1rem;
}
.hero .chips {position: relative; display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1.1rem;}
.chip {
  display: inline-block; padding: 0.28rem 0.62rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.24);
  background: rgba(255,255,255,0.08); color: #e0f2fe;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
}
.side-brand {
  display: flex; align-items: center; gap: 0.7rem; padding: 0.15rem 0 0.8rem;
}
.side-mark {
  width: 42px; height: 42px; border-radius: 12px;
  background: linear-gradient(135deg, #082f49, #0e7490);
  color: white; display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em;
  box-shadow: 0 5px 14px rgba(8,47,73,0.22);
}
.side-brand b {font-size: 0.94rem; color: var(--ink);}
.side-brand span {font-size: 0.72rem; color: var(--muted);}
.sidebar-note {font-size: 0.74rem; color: #64748b; line-height: 1.4;}

.run-head {
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 1.2rem 1.6rem;
  align-items: flex-start;
  background: #ffffff; border: 1px solid var(--line); border-radius: 16px;
  padding: 1.1rem 1.35rem; margin-bottom: 0.9rem;
  box-shadow: 0 3px 12px rgba(15,23,42,0.04);
}
.run-head > div:first-child {flex: 1 1 340px; min-width: 0;}
.run-head h2 {margin: 0.45rem 0 0.25rem; font-size: 1.4rem; white-space: nowrap;}
.run-head p {margin: 0; color: var(--muted); font-size: 0.88rem;}
.run-meta {display: flex; gap: 1.25rem; flex-wrap: wrap; justify-content: flex-start;}
.run-meta div {min-width: 74px;}
.run-meta span {
  display: block; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.09em;
  text-transform: uppercase; color: #94a3b8; margin-bottom: 0.18rem;
}
.run-meta b {font-size: 1.02rem; color: var(--ink);}
.pill {
  display: inline-block; border-radius: 999px; padding: 0.22rem 0.58rem;
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.06em;
  text-transform: uppercase; margin-right: 0.3rem;
}
.pill.radar {background: #ecfeff; color: #0e7490; border: 1px solid #a5f3fc;}
.pill.demo {background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa;}
.pill.ok {background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;}
.pill.warn {background: #fffbeb; color: #b45309; border: 1px solid #fde68a;}
.pill.ghost {background: #f8fafc; color: #475569; border: 1px solid #e2e8f0;}

.step-card {
  background: #ffffff; border: 1px solid var(--line); border-radius: 14px;
  padding: 1rem 1.05rem; height: 100%;
}
.step-card .num {
  font-size: 0.68rem; font-weight: 800; color: var(--brand);
  letter-spacing: 0.1em; text-transform: uppercase;
}
.step-card h4 {margin: 0.35rem 0 0.35rem; font-size: 0.98rem;}
.step-card p {margin: 0; font-size: 0.82rem; color: var(--muted); line-height: 1.42;}

.callout {
  background: linear-gradient(90deg, #ecfeff, #f0fdfa);
  border: 1px solid #a5f3fc; border-radius: 14px; padding: 0.9rem 1.1rem;
}
.callout b {color: #155e75;}
.callout span {color: #475569; font-size: 0.86rem;}

.compass {
  width: 106px; height: 106px; border-radius: 50%; position: relative;
  border: 1px solid #cbd5e1; background:
    radial-gradient(circle, #ffffff 0 36%, transparent 37%),
    repeating-conic-gradient(from 0deg, #e2e8f0 0 1deg, transparent 1deg 30deg);
  margin: 0.2rem auto 0.4rem;
}
.compass .arrow {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-size: 2.6rem; line-height: 1; color: var(--brand); transform-origin: center;
  padding-bottom: 0.1rem;
}
.compass .n {position: absolute; top: 3px; left: 50%; transform: translateX(-50%); font-size: 0.6rem; color: #94a3b8; font-weight: 700;}
.compass .e {position: absolute; right: 5px; top: 50%; transform: translateY(-50%); font-size: 0.6rem; color: #94a3b8; font-weight: 700;}
.compass .s {position: absolute; bottom: 2px; left: 50%; transform: translateX(-50%); font-size: 0.6rem; color: #94a3b8; font-weight: 700;}
.compass .w {position: absolute; left: 5px; top: 50%; transform: translateY(-50%); font-size: 0.6rem; color: #94a3b8; font-weight: 700;}

.small-note {font-size: 0.78rem; color: #64748b; line-height: 1.45;}
.export-card {
  border: 1px dashed #cbd5e1; border-radius: 14px; padding: 0.9rem 1rem;
  background: #f8fafc;
}
/* Let paired map panels wrap to full width on narrow screens instead of
   squeezing two square maps into a few hundred pixels. */
.st-key-overview_maps div[data-testid="stHorizontalBlock"],
.st-key-ensemble_maps div[data-testid="stHorizontalBlock"] {
  flex-wrap: wrap;
}
.st-key-overview_maps div[data-testid="stColumn"],
.st-key-ensemble_maps div[data-testid="stColumn"] {
  flex: 1 1 430px !important;
  min-width: 340px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=120, show_spinner=False)
def cached_inventory(path_str: str) -> dict:
    return radar_core.inventory_archive(path_str)


def fmt(value, digits: int = 2, suffix: str = "") -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "—"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:  # noqa: BLE001
        return "—"


def nearest_lead_index(result: NowcastResult, minutes: float) -> int:
    return int(np.argmin(np.abs(np.asarray(result.lead_times_min) - float(minutes))))


def key_lead_indices(result: NowcastResult) -> list[int]:
    targets = [30, 60, 90, 120]
    indices: list[int] = []
    max_lead = float(result.lead_times_min[-1]) if result.n_steps else 0.0
    for target in targets:
        if target <= max_lead + 1:
            idx = nearest_lead_index(result, target)
            if idx not in indices:
                indices.append(idx)
    if not indices:
        indices = [result.n_steps - 1]
    return indices


def metric_key(threshold: float) -> str:
    return f"{float(threshold):g}"


def compass_name(degrees: float) -> str:
    names = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    try:
        return names[int((float(degrees) % 360 + 11.25) // 22.5) % 16]
    except Exception:  # noqa: BLE001
        return "—"


def get_district_masks(result: NowcastResult):
    cache = st.session_state.setdefault("district_masks", {})
    key = f"{result.run_id}_{result.extent_km}_{result.res_km}"
    if key not in cache:
        if not result.districts.get("polygons"):
            cache[key] = []
        else:
            with st.spinner("Rasterizing district boundaries…"):
                cache[key] = geodata.build_district_masks(
                    result.districts, result.extent_km, result.res_km
                )
    return cache[key]


def csv_bytes(result: NowcastResult) -> bytes:
    header, rows = store.metrics_csv_rows(result)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=2, default=str, allow_nan=True).encode("utf-8")


def hazard_table(result: NowcastResult) -> pd.DataFrame:
    rows = []
    for idx in key_lead_indices(result):
        prob = result.probability_field(idx, 5.0)
        rows.append(
            {
                "Lead": result.lead_label(idx),
                "Valid (UTC)": result.valid_time_label(idx),
                "Peak P≥5 (mm/h)": float(np.nanmax(prob)),
                "Surabaya mean P≥5": result.probability_around_surabaya(idx, 5.0),
                "Domain mean P≥5": float(np.nanmean(prob[result.valid_mask])),
            }
        )
    return pd.DataFrame(rows)


def verification_table(result: NowcastResult) -> pd.DataFrame | None:
    if not result.verified:
        return None
    metrics = result.metrics
    scales = [int(s) for s in metrics.get("fss_scales_km", [])]
    rows = []
    for idx in key_lead_indices(result):
        row = {
            "Lead": result.lead_label(idx),
            "Valid (UTC)": result.valid_time_label(idx),
            "CRPS": metrics["crps"][idx],
            "Brier@5": metrics["brier"].get("5", [np.nan] * result.n_steps)[idx],
            "AUC@5": metrics["roc_auc"].get("5", [np.nan] * result.n_steps)[idx],
        }
        for scale in scales:
            value = metrics["fss"].get(f"{scale:g}", [np.nan] * result.n_steps)[idx]
            row[f"FSS {scale} km"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def execute_run(config: NowcastConfig) -> None:
    status = st.status("Preparing nowcast…", expanded=True)
    with status:
        bar = st.progress(0.0, text="Starting")
        log_box = st.empty()
        logs: list[str] = []

        def progress(fraction: float, message: str) -> None:
            bar.progress(float(min(max(fraction, 0.0), 1.0)), text=message)
            if not logs or logs[-1] != message:
                logs.append(message)
                log_box.markdown(
                    "  \n".join(f"<span class='small-note'>• {html.escape(m)}</span>" for m in logs[-4:]),
                    unsafe_allow_html=True,
                )

        try:
            if config.mode == "demo":
                result = run_demo_nowcast(config, progress=progress)
            else:
                result = run_radar_nowcast(config, progress=progress)

            if config.save_run:
                bar.progress(0.99, text="Writing run to history…")
                with st.spinner("Saving ensemble, metrics and run metadata…"):
                    run_dir = store.save_run(result)
                st.caption(f"Saved to `{run_dir}`")
            status.update(
                label=f"Nowcast complete · {result.init_label}",
                state="complete",
                expanded=False,
            )
            st.session_state["result"] = result
            st.session_state.pop("district_masks", None)
            st.toast("Nowcast complete", icon="✅")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            status.update(label="Nowcast failed", state="error", expanded=True)
            st.error(f"Run failed: {exc}")
            with st.expander("Technical details", expanded=False):
                st.exception(exc)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="side-brand">
              <div class="side-mark">SBY</div>
              <div><b>Dr. AW Radar Lab</b><br><span>Nowcast dashboard</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_mode = st.radio(
            "Data source",
            [SOURCE_RADAR, SOURCE_DEMO],
            horizontal=True,
            key="source_mode",
        )

        archive_path = str(DEFAULT_DATA_DIR)
        inventory = None
        inventory_error = ""
        manual_ts = None
        if source_mode == SOURCE_RADAR:
            archive_path = st.text_input(
                "Rainbow5 archive folder",
                value=st.session_state.get("archive_path", str(DEFAULT_DATA_DIR)),
                key="archive_path",
                help="Folder containing *dBZ.vol, *RhoHV.vol, *ZDR.vol and *PhiDP.vol.",
            )
            if archive_path.strip():
                try:
                    inventory = cached_inventory(archive_path)
                except Exception as exc:  # noqa: BLE001
                    inventory_error = str(exc)
            if inventory:
                st.caption(
                    f"✓ {inventory['n_scans']} complete scans · "
                    f"{inventory['timestep_min']:.0f} min cadence"
                )
                if inventory["n_incomplete"]:
                    st.caption(f"⚠ {inventory['n_incomplete']} incomplete scans ignored")
            elif inventory_error:
                st.caption("⚠ Archive not ready")
        else:
            st.markdown(
                "<div class='sidebar-note'>Synthetic 06 Mar 2025 convective event. "
                "No radar files are read.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Nowcast settings")
        horizon = st.slider(
            "Forecast horizon (min)", 30, 180, 120, 6, key="cfg_horizon"
        )
        members = st.slider(
            "STEPS ensemble members", 5, 40, 20, 1, key="cfg_members"
        )
        grid_cols = st.columns(2)
        with grid_cols[0]:
            altitude = st.slider(
                "Pseudo-CAPPI altitude (m)", 500, 4000, 1500, 250, key="cfg_alt"
            )
        with grid_cols[1]:
            extent = st.slider(
                "Domain radius (km)", 75, 150, 150, 5, key="cfg_extent"
            )
        res = st.select_slider(
            "Grid resolution (km)",
            options=[1.0, 1.5, 2.0],
            value=1.0,
            key="cfg_res",
        )

        init_mode = "auto"
        if source_mode == SOURCE_RADAR:
            init_choice = st.radio(
                "Storm initialization",
                ["Auto peak activity", "Manual scan"],
                horizontal=True,
                key="cfg_init_mode",
            )
            init_mode = "auto" if init_choice.startswith("Auto") else "manual"
            if init_mode == "manual":
                options = []
                if inventory:
                    ts_list = inventory["timestamps"]
                    timestep = inventory["timestep_min"]
                    n_steps = max(1, int(round(horizon / timestep)))
                    last = max(3, len(ts_list) - n_steps - 1)
                    options = ts_list[2:last]
                if options:
                    manual_ts = st.selectbox(
                        "Initialization scan (UTC)",
                        options=options,
                        index=None,
                        placeholder="Choose a scan…",
                        format_func=radar_core.timestamp_label,
                        key="cfg_manual_ts",
                    )
                else:
                    st.caption("No scan has enough lead time for this horizon.")

        with st.expander("Advanced science settings", expanded=False):
            max_elev = st.slider(
                "Maximum elevation sweep (°)", 6.0, 20.0, 12.0, 0.5, key="cfg_maxelev"
            )
            zr_cols = st.columns(2)
            with zr_cols[0]:
                zr_a = st.number_input(
                    "Z–R a", min_value=50.0, max_value=600.0, value=250.0, step=10.0,
                    key="cfg_zra",
                )
            with zr_cols[1]:
                zr_b = st.number_input(
                    "Z–R b", min_value=1.0, max_value=2.0, value=1.2, step=0.05,
                    key="cfg_zrb",
                )
            cascade = st.slider(
                "STEPS cascade levels", 3, 8, 6, 1, key="cfg_cascade"
            )
            thresholds = st.multiselect(
                "Verification thresholds (mm/h)",
                options=list(MAP_THRESHOLDS),
                default=list(DEFAULT_THRESHOLDS),
                key="cfg_thresholds",
            )
            fss_scales = st.multiselect(
                "FSS neighbourhoods (km)",
                options=[5, 10, 15, 20, 30, 40],
                default=list(DEFAULT_FSS_SCALES),
                key="cfg_fss_scales",
            )
            fss_thr = st.number_input(
                "FSS threshold (mm/h)", min_value=1.0, max_value=30.0, value=5.0, step=1.0,
                key="cfg_fss_thr",
            )
            seed = st.number_input(
                "Random seed", min_value=0, max_value=100000, value=42, step=1,
                key="cfg_seed",
            )
            verify = st.toggle("Verify against future scans", value=True, key="cfg_verify")
            geojson_path = st.text_input(
                "Local ADM2 GeoJSON (optional)",
                value="",
                key="cfg_geojson",
                help="If empty, cached/downloaded geoBoundaries ADM2 is used.",
            )
            save_run = st.toggle("Save run to history", value=True, key="cfg_save")

        config: NowcastConfig | None = None
        config_error = ""
        if not thresholds:
            config_error = "Select at least one verification threshold."
        elif not fss_scales:
            config_error = "Select at least one FSS neighbourhood."
        elif init_mode == "manual" and not manual_ts:
            config_error = "Choose an initialization scan, or switch to auto."
        else:
            try:
                config = NowcastConfig(
                    data_dir=archive_path,
                    mode="demo" if source_mode == SOURCE_DEMO else "radar",
                    grid_extent_km=float(extent),
                    res_km=float(res),
                    target_alt_m=float(altitude),
                    max_elev_deg=float(max_elev),
                    zr_a=float(zr_a),
                    zr_b=float(zr_b),
                    horizon_min=float(horizon),
                    n_members=int(members),
                    n_cascade_levels=int(cascade),
                    seed=int(seed),
                    verify=bool(verify),
                    thresholds=tuple(float(t) for t in thresholds),
                    fss_scales_km=tuple(int(s) for s in fss_scales),
                    fss_thr_mmh=float(fss_thr),
                    init_mode=init_mode,
                    init_timestamp=manual_ts,
                    geojson_path=geojson_path.strip() or None,
                    save_run=bool(save_run),
                )
            except Exception as exc:  # noqa: BLE001
                config_error = str(exc)

        if config is not None:
            if source_mode == SOURCE_DEMO:
                st.caption("Demo compute · seconds · no radar archive needed")
            else:
                step_min = inventory["timestep_min"] if inventory else 6.0
                n_steps = config.n_steps(step_min)
                estimate = workload_estimate(config, n_steps)
                st.caption(
                    f"Estimated compute · {estimate['text']} · "
                    f"{estimate['n_cappi']} CAPPIs · {n_steps} lead times"
                )
            grid = config.grid_shape[0]
            mem_mb = (
                config.n_members
                * config.n_steps(inventory["timestep_min"] if inventory else 6.0)
                * grid
                * grid
                * 4
                / 1024**2
            )
            st.caption(
                f"Grid {grid} × {grid} · ensemble working set ≈ {mem_mb:,.0f} MB"
            )

        if config_error:
            st.caption(f"⚠ {config_error}")

        run_clicked = st.button(
            "Run synthetic demo" if source_mode == SOURCE_DEMO else "Run radar nowcast",
            type="primary",
            width="stretch",
            disabled=config is None,
            key="run_pipeline",
        )

        current = st.session_state.get("result")
        if current is not None:
            if st.button("Close current run", width="stretch", key="close_run"):
                st.session_state["result"] = None
                st.session_state.pop("district_masks", None)
                st.rerun()

        st.divider()
        st.markdown("#### Run history")
        runs = store.list_runs()
        if not runs:
            st.caption("No stored runs yet. Runs are saved locally after completion.")
        else:
            labels = []
            for item in runs:
                tag = "DEMO" if item.mode == "demo" else "RADAR"
                labels.append(f"{item.init_timestamp} · {tag}")
            selected = st.selectbox(
                "Stored runs",
                options=list(range(len(runs))),
                format_func=lambda i: labels[i],
                key="history_select",
                label_visibility="collapsed",
            )
            info = runs[selected]
            st.caption(
                f"{info.steps} leads · {info.size_mb:.1f} MB · "
                f"saved {info.created_at[:16].replace('T', ' ')} UTC"
            )
            hcols = st.columns(2)
            with hcols[0]:
                if st.button("Load run", width="stretch", key="load_run"):
                    try:
                        with st.spinner("Loading stored run…"):
                            loaded = store.load_run(info.path)
                        st.session_state["result"] = loaded
                        st.session_state.pop("district_masks", None)
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not load run: {exc}")
            with hcols[1]:
                with st.popover("Manage", width="stretch"):
                    st.caption(f"Delete `{info.run_id}`?")
                    confirm = st.checkbox("Confirm deletion", key=f"del_confirm_{info.run_id}")
                    if st.button(
                        "Delete run",
                        type="secondary",
                        disabled=not confirm,
                        key=f"del_{info.run_id}",
                    ):
                        store.delete_run(info.path)
                        st.rerun()

        st.caption(
            "Research and training prototype. Not an operational warning service."
        )

    return config, run_clicked, source_mode, inventory, inventory_error, archive_path


# --------------------------------------------------------------------------- #
# Landing page
# --------------------------------------------------------------------------- #
def render_landing(
    config: NowcastConfig | None,
    source_mode: str,
    inventory: dict | None,
    inventory_error: str,
    archive_path: str,
) -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Dr. AW Radar Lab · C-band dual-pol · STEPS</div>
          <h1>2-hour rainfall nowcasting,<br>with verification built in.</h1>
          <p>Turn BMKG Surabaya Rainbow5 volumes into a quality-controlled 1.5 km
          pseudo-CAPPI, a 20-member STEPS ensemble and a skill assessment against
          future radar scans — all from one dashboard.</p>
          <div class="chips">
            <span class="chip">Dual-pol QC + ΦDP attenuation correction</span>
            <span class="chip">1.5 km pseudo-CAPPI</span>
            <span class="chip">2-hour ensemble</span>
            <span class="chip">CRPS · Brier · ROC AUC · FSS</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_mode == SOURCE_DEMO:
        st.info(
            "**Synthetic demo selected.** The dashboard will generate an advecting "
            "storm-cell complex with a 20-member ensemble. Every map and download is "
            "labelled synthetic — use it to explore the workflow without radar files."
        )

    # Archive status / quick configuration
    st.markdown("### Workspace status")
    row1 = st.columns(2)
    row2 = st.columns(2)
    if source_mode == SOURCE_RADAR:
        if inventory:
            with row1[0]:
                st.metric("Volumes", f"{inventory['n_scans']:,}")
            with row1[1]:
                st.metric("Scan cadence", f"{inventory['timestep_min']:.0f} min")
            with row2[0]:
                st.metric("Archive size", f"{inventory['size_gb']:.1f} GB")
            with row2[1]:
                st.metric("Coverage window", "24 h")
            first_dt = radar_core.parse_timestamp(inventory["first"])
            last_dt = radar_core.parse_timestamp(inventory["last"])
            if first_dt.date() == last_dt.date():
                window_label = (
                    f"{first_dt:%d %b %Y} · {first_dt:%H:%M} → "
                    f"{last_dt:%H:%M} UTC"
                )
            else:
                window_label = (
                    f"{first_dt:%d %b %H:%M} → {last_dt:%d %b %H:%M} UTC"
                )
            st.caption(
                f"{window_label} · {inventory['n_incomplete']} incomplete scans ignored"
            )
        else:
            with row1[0]:
                st.metric("Archive", "Not ready")
            with row1[1]:
                st.metric("Path", "—")
            with row2[0]:
                st.metric("Cadence", "—")
            with row2[1]:
                st.metric("Complete volumes", "—")
            if inventory_error:
                st.warning(f"Archive not available: {inventory_error}")
            st.caption(f"Checked: `{archive_path}`")
    else:
        demo_cfg = config
        with row1[0]:
            st.metric("Event initialization", "06 Mar · 14:38 UTC")
        with row1[1]:
            st.metric("Ensemble members", demo_cfg.n_members if demo_cfg else "—")
        with row2[0]:
            st.metric("Forecast horizon", f"{demo_cfg.horizon_min:.0f} min" if demo_cfg else "—")
        with row2[1]:
            st.metric("Grid", f"{demo_cfg.grid_shape[0]}²" if demo_cfg else "—")

    # Pipeline explainer
    st.markdown("### How the pipeline works")
    steps = [
        ("01 · Ingest", "Rainbow5 ingest", "Open dBZ, RhoHV, ZDR and ΦDP volumes from the Baron radar archive."),
        ("02 · Build", "Dual-pol pseudo-CAPPI", "QC echoes, correct C-band attenuation, project to 1.5 km and interpolate vertically."),
        ("03 · Nowcast", "STEPS ensemble", "Lucas–Kanade motion + stochastic cascade nowcast, 2 hours ahead."),
        ("04 · Verify", "Radar-based skill", "Score against future pseudo-CAPPIs with CRPS, Brier, ROC AUC and FSS."),
    ]
    for pair in (steps[:2], steps[2:]):
        step_cols = st.columns(2)
        for col, (num, title, body) in zip(step_cols, pair):
            with col:
                st.markdown(
                    f"<div class='step-card'><div class='num'>{num}</div>"
                    f"<h4>{title}</h4><p>{body}</p></div>",
                    unsafe_allow_html=True,
                )

    # Activity / quick start
    left, right = st.container(), st.container()
    with left:
        st.markdown("### Storm activity in the archive")
        if source_mode == SOURCE_RADAR and inventory:
            activity = st.session_state.get("activity_profile")
            activity_path = st.session_state.get("activity_path")
            if activity is None or activity_path != archive_path:
                with st.spinner("Profiling convective activity (one-time; cached on disk)…"):
                    try:
                        activity = radar_core.activity_profile(
                            Path(archive_path), inventory["timestamps"]
                        )
                        st.session_state["activity_profile"] = activity
                        st.session_state["activity_path"] = archive_path
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Could not profile activity: {exc}")
                        activity = None
            if activity:
                peak_index = max(
                    range(len(activity)),
                    key=lambda i: activity[i]["convective_pixels"],
                )
                fig = viz.activity_chart(activity, init_index=peak_index)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
                peak = activity[peak_index]
                st.markdown(
                    f"<div class='callout'><b>Archive peak: {peak['label']}</b> · "
                    f"<span>{peak['convective_pixels']:,} pixels ≥ 25 dBZ in the "
                    f"first sweep. Auto mode picks the strongest scan that still has "
                    f"enough future scans for verification.</span></div>",
                    unsafe_allow_html=True,
                )
        elif source_mode == SOURCE_DEMO:
            st.markdown(
                "<div class='callout'><b>Demo storm:</b> a cluster of convective "
                "cells moves west-northwest across the Baron domain, with growth and "
                "decay over the two-hour window.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Connect a Rainbow5 archive to see the convective-activity timeline.")

    with right:
        st.markdown("### Start a run")
        if config is not None:
            if source_mode == SOURCE_DEMO:
                st.write(
                    "The demo uses the current sidebar settings and writes a fully "
                    "interactive result in seconds."
                )
                if st.button("Generate synthetic demo", type="primary", width="stretch", key="landing_demo"):
                    execute_run(config)
            else:
                if inventory:
                    st.write(
                        f"Configured for **{config.n_members} members**, "
                        f"**{config.horizon_min:.0f} min**, "
                        f"**{config.grid_shape[0]}² grid** at "
                        f"**{config.target_alt_m / 1000:.1f} km**."
                    )
                    disabled = config.init_mode == "manual" and not config.init_timestamp
                    if st.button(
                        "Run nowcast with current settings",
                        type="primary",
                        width="stretch",
                        key="landing_run",
                        disabled=disabled,
                    ):
                        execute_run(config)
                    st.caption("The same control is available in the sidebar.")
                else:
                    st.warning("Fix the archive path in the sidebar to enable the run.")
        else:
            st.warning("Complete the sidebar settings to enable a run.")

        runs = store.list_runs()
        if runs:
            st.markdown("#### Recent runs")
            for item in runs[:3]:
                title = "Synthetic demo" if item.mode == "demo" else "Radar run"
                rc = st.columns([3, 1])
                with rc[0]:
                    st.caption(
                        f"**{title}** · {item.init_timestamp}  \n"
                        f"{item.steps} leads · {item.size_mb:.1f} MB"
                    )
                with rc[1]:
                    if st.button("Open", key=f"landing_open_{item.run_id}", width="stretch"):
                        try:
                            with st.spinner("Loading stored run…"):
                                loaded = store.load_run(item.path)
                            st.session_state["result"] = loaded
                            st.session_state.pop("district_masks", None)
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Could not load run: {exc}")

    st.divider()
    st.caption(
        "Method references: Seed (2003) STEPS; Lucas & Kanade (1981) optical flow; "
        "Roberts & Lean (2008) FSS; geoBoundaries ADM2 boundaries. "
        "Radar-derived rain rates are indirect estimates, and verification against "
        "radar is not the same as verification against rain gauges."
    )


# --------------------------------------------------------------------------- #
# Result header
# --------------------------------------------------------------------------- #
def render_result_header(result: NowcastResult) -> None:
    mode_pill = (
        "<span class='pill demo'>Synthetic demo</span>"
        if result.is_demo
        else "<span class='pill radar'>Radar archive</span>"
    )
    verify_pill = (
        "<span class='pill ok'>Verified</span>"
        if result.verified
        else "<span class='pill warn'>Nowcast only</span>"
    )
    saved_pill = (
        "<span class='pill ghost'>Saved run</span>" if result.run_dir else ""
    )
    lead_min = int(round(float(result.lead_times_min[-1]))) if result.n_steps else 0
    ny, nx = result.ensemble.shape[-2:]
    st.markdown(
        f"""
        <div class="run-head">
          <div>
            <div>{mode_pill}{verify_pill}{saved_pill}</div>
            <h2>{html.escape(result.init_label)}</h2>
            <p>BMKG Surabaya Baron GEN3-350CM · {result.config.target_alt_m / 1000:.1f} km
            pseudo-CAPPI · {result.n_members}-member STEPS · {lead_min} min horizon</p>
          </div>
          <div class="run-meta">
            <div><span>Grid</span><b>{nx} × {ny}</b></div>
            <div><span>Domain</span><b>{2 * result.extent_km:.0f} km</b></div>
            <div><span>Resolution</span><b>{result.res_km:g} km</b></div>
            <div><span>Lead times</span><b>{result.n_steps}</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def render_overview(result: NowcastResult) -> None:
    if result.is_demo:
        st.info(
            "**Synthetic demonstration.** The maps and scores below come from a "
            "generated storm, not from observed rainfall. Switch to a radar archive "
            "to run the real pipeline."
        )

    idx60 = nearest_lead_index(result, 60)
    with st.container(key="overview_maps"):
        map_cols = st.columns(2, gap="large")
        with map_cols[0]:
            st.plotly_chart(
                viz.initial_map(result, show_labels=True),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"overview_init_{result.run_id}",
            )
        with map_cols[1]:
            st.plotly_chart(
                viz.probability_map(
                    result, idx60, 5.0, min_probability=2.0, show_labels=True
                ),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"overview_prob_{result.run_id}",
            )

    motion = result.motion_summary()
    lead60 = result.lead_times_min[idx60]
    lead60_label = f"+{lead60:.0f}"
    peak60 = result.domain_peak_probability(idx60, 5.0)
    sby60 = result.probability_around_surabaya(idx60, 5.0)
    kpi_row1 = st.columns(2)
    kpi_row2 = st.columns(2)
    with kpi_row1[0]:
        if motion:
            st.metric(
                "Storm motion",
                f"{motion['speed_kmh']:.0f} km/h",
                f"towards {compass_name(motion['direction_deg'])} "
                f"({motion['direction_deg']:.0f}°)",
            )
        else:
            st.metric("Storm motion", "—")
    with kpi_row1[1]:
        st.metric(f"Peak P ≥ 5 mm/h at {lead60_label}", f"{fmt(peak60, 0)}%")
    with kpi_row2[0]:
        st.metric("Surabaya mean P ≥ 5 mm/h", f"{fmt(sby60, 0)}%")
    with kpi_row2[1]:
        if result.verified:
            idx = nearest_lead_index(result, 60)
            auc = result.metrics["roc_auc"].get("5", [np.nan] * result.n_steps)[idx]
            st.metric(f"ROC AUC at {lead60_label}", fmt(auc, 2), help="Above 0.70 is generally useful.")
        else:
            st.metric("Verification", "Off", help="Enable verification before running.")

    st.markdown("### Forecast and skill at a glance")
    left, right = st.container(), st.container()
    with left:
        st.markdown("**Exceedance probability · ≥ 5 mm/h**")
        hazard = hazard_table(result)
        st.dataframe(
            hazard,
            hide_index=True,
            width="stretch",
            column_config={
                "Peak P≥5 (mm/h)": st.column_config.ProgressColumn(
                    "Peak P≥5 (mm/h)", min_value=0, max_value=100, format="%.0f%%"
                ),
                "Surabaya mean P≥5": st.column_config.ProgressColumn(
                    "Surabaya mean P≥5", min_value=0, max_value=100, format="%.0f%%"
                ),
                "Domain mean P≥5": st.column_config.NumberColumn(
                    "Domain mean P≥5", format="%.1f%%"
                ),
            },
        )
        st.caption(
            "Peak is the maximum grid-cell probability in the 150 km domain; "
            "Surabaya is the mean over an 18 km disc around the city centre."
        )
    with right:
        st.markdown("**Verification against future pseudo-CAPPIs**")
        table = verification_table(result)
        if table is not None:
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "CRPS": st.column_config.NumberColumn(format="%.3f"),
                    "Brier@5": st.column_config.NumberColumn(format="%.3f"),
                    "AUC@5": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(
                f"FSS is computed at ≥ {result.config.fss_thr_mmh:g} mm/h. "
                "AUC above 0.70 and FSS above 0.50 are common usefulness benchmarks."
            )
        else:
            st.caption(
                "Verification was disabled for this run. Enable it under Advanced "
                "science settings and run again to score the ensemble."
            )

    if result.verified:
        masks = get_district_masks(result)
        if masks:
            st.markdown(f"### District exposure at {lead60_label} · ≥ 5 mm/h")
            rows = geodata.district_probability_table(
                result, idx60, 5.0, masks, top_n=8
            )
            if rows:
                df = pd.DataFrame(rows).drop(columns=["Area (km²)"], errors="ignore")
                st.dataframe(
                    df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Mean probability (%)": st.column_config.ProgressColumn(
                            "Mean probability (%)", min_value=0, max_value=100, format="%.0f%%"
                        ),
                        "Max probability (%)": st.column_config.NumberColumn(format="%.0f%%"),
                        "Area ≥ 50% (km²)": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
                st.caption(f"Boundaries: {result.districts.get('source', '')}")


# --------------------------------------------------------------------------- #
# Probability map tab
# --------------------------------------------------------------------------- #
def render_probability(result: NowcastResult) -> None:
    row1 = st.columns([1, 2])
    row2 = st.columns([2, 1, 1])
    with row1[0]:
        choice = st.selectbox(
            "Threshold (mm/h)",
            options=[f"{t:g}" for t in MAP_THRESHOLDS] + ["Custom"],
            index=2,
            key="prob_threshold_select",
        )
    if choice == "Custom":
        with row1[0]:
            threshold = st.number_input(
                "Custom threshold", 0.5, 100.0, 12.0, 0.5, key="prob_custom_thr"
            )
    else:
        threshold = float(choice)
    default_idx = nearest_lead_index(result, 60)
    with row1[1]:
        lead_idx = st.select_slider(
            "Lead time",
            options=list(range(result.n_steps)),
            value=default_idx,
            format_func=result.lead_label,
            key="prob_lead",
        )
    with row2[0]:
        min_prob = st.slider("Hide below (%)", 0, 25, 1, key="prob_min")
    with row2[1]:
        labels = st.toggle("Labels", value=True, key="prob_labels")
    with row2[2]:
        rings = st.toggle("Range rings", value=True, key="prob_rings")

    st.plotly_chart(
        viz.probability_map(result, lead_idx, threshold, min_prob, labels, rings),
        width="stretch",
        config=PLOT_CONFIG,
        key=f"prob_map_{result.run_id}",
    )

    prob = result.probability_field(lead_idx, threshold)
    area50 = float(np.sum(prob[result.valid_mask] >= 50.0) * result.res_km**2)
    stats1 = st.columns(2)
    stats2 = st.columns(2)
    with stats1[0]:
        st.metric("Peak probability", f"{fmt(np.nanmax(prob), 0)}%")
    with stats1[1]:
        st.metric(
            "Surabaya area mean",
            f"{fmt(result.probability_around_surabaya(lead_idx, threshold), 0)}%",
        )
    with stats2[0]:
        st.metric("Domain mean probability", f"{fmt(np.nanmean(prob[result.valid_mask]), 1)}%")
    with stats2[1]:
        st.metric("Area with P ≥ 50%", f"{area50:,.0f} km²")

    st.caption(
        "Probability is the fraction of ensemble members exceeding the threshold "
        "at each 1 km grid cell. Transparent areas are below the display cutoff."
    )

    masks = get_district_masks(result)
    if masks:
        st.markdown("### District exposure")
        rows = geodata.district_probability_table(
            result, lead_idx, threshold, masks, top_n=12
        )
        if rows:
            bar_col, table_col = st.container(), st.container()
            with bar_col:
                fig = viz.district_bars(rows, threshold, result.lead_label(lead_idx))
                if fig is not None:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config=PLOT_CONFIG,
                        key=f"district_bars_{result.run_id}",
                    )
            with table_col:
                df = pd.DataFrame(rows).drop(columns=["Area (km²)"], errors="ignore")
                st.dataframe(
                    df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Mean probability (%)": st.column_config.ProgressColumn(
                            "Mean probability (%)", min_value=0, max_value=100, format="%.0f%%"
                        ),
                        "Max probability (%)": st.column_config.NumberColumn(format="%.0f%%"),
                        "Area ≥ 50% (km²)": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
            st.caption(f"Boundaries: {result.districts.get('source', '')}")
    else:
        st.caption(
            "District polygons are unavailable. Add an ADM2 GeoJSON in Advanced "
            "settings to enable district exposure statistics."
        )

    with st.expander("Publication figure · 4-threshold probability panel", expanded=False):
        png_key = f"png_prob_{result.run_id}_{lead_idx}"
        if st.button("Build probability PNG", key=f"build_prob_png_{lead_idx}"):
            with st.spinner("Rendering publication figure…"):
                fig = plotting.figure_probability_maps(result, lead_idx)
                st.session_state[png_key] = plotting.fig_to_png_bytes(fig)
        if png_key in st.session_state:
            st.image(st.session_state[png_key], width="stretch")
            st.download_button(
                "Download PNG",
                data=st.session_state[png_key],
                file_name=f"surabaya_probability_{result.init_timestamp[:12]}_{lead_idx + 1:02d}.png",
                mime="image/png",
                key=f"dl_prob_png_{lead_idx}",
            )


# --------------------------------------------------------------------------- #
# Observed vs forecast
# --------------------------------------------------------------------------- #
def render_obs_forecast(result: NowcastResult) -> None:
    if not result.verified:
        st.info(
            "This run has no verification. Enable **Verify against future scans** "
            "in the sidebar and run again to compare observations and forecasts."
        )
        return
    default_idx = nearest_lead_index(result, 60)
    controls = st.columns([2, 1, 1])
    with controls[0]:
        lead_idx = st.select_slider(
            "Lead time",
            options=list(range(result.n_steps)),
            value=default_idx,
            format_func=result.lead_label,
            key="obs_lead",
        )
    with controls[1]:
        labels = st.toggle("Labels", value=False, key="obs_labels")
    with controls[2]:
        st.caption(f"Valid {result.valid_time_label(lead_idx)}")

    st.plotly_chart(
        viz.obs_vs_forecast(result, lead_idx, labels),
        width="stretch",
        config=PLOT_CONFIG,
        key=f"obs_map_{result.run_id}",
    )

    metrics = result.metrics
    idx = lead_idx
    thr_key = "5"
    scale_key = min(
        [f"{s:g}" for s in metrics.get("fss_scales_km", [])],
        key=lambda key: abs(float(key) - 5.0),
        default="5",
    )
    values = [
        ("CRPS", metrics["crps"][idx], "mm/h"),
        ("MAE", metrics["mae"][idx], "mm/h"),
        ("Bias", metrics["bias"][idx], "mm/h"),
        ("Spatial correlation", metrics["corr"][idx], ""),
        ("Brier @ 5 mm/h", metrics["brier"].get(thr_key, [np.nan] * result.n_steps)[idx], ""),
        ("ROC AUC @ 5 mm/h", metrics["roc_auc"].get(thr_key, [np.nan] * result.n_steps)[idx], ""),
        (f"FSS {scale_key} km @ 5 mm/h", metrics["fss"].get(scale_key, [np.nan] * result.n_steps)[idx], ""),
    ]
    cols = st.columns(2)
    cols2 = st.columns(2)
    for col, (label, value, suffix) in zip(cols, values[:2]):
        with col:
            st.metric(label, f"{fmt(value, 2)} {suffix}".strip())
    for col, (label, value, suffix) in zip(cols2, values[2:4]):
        with col:
            st.metric(label, f"{fmt(value, 2)} {suffix}".strip())
    extra = st.columns(3)
    for col, (label, value, suffix) in zip(extra, values[4:]):
        with col:
            st.metric(label, fmt(value, 2))

    scatter_col, info_col = st.container(), st.container()
    with scatter_col:
        scatter = viz.scatter_obs_forecast(result, lead_idx)
        if scatter is not None:
            st.plotly_chart(
                scatter,
                width="stretch",
                config=PLOT_CONFIG,
                key=f"scatter_{result.run_id}",
            )
    with info_col:
        obs = result.observed_field(lead_idx)
        mean = result.mean_field(lead_idx)
        threshold_area = 5.0
        obs_area = float(np.nanmean(obs[result.valid_mask] >= threshold_area) * 100)
        fcst_area = float(np.nanmean(mean[result.valid_mask] >= threshold_area) * 100)
        peak_prob = float(np.nanmax(result.probability_field(lead_idx, threshold_area)))
        st.markdown("**At this lead time**")
        st.metric("Observed area ≥ 5 mm/h", f"{obs_area:.2f}% of domain")
        st.metric("Forecast area ≥ 5 mm/h", f"{fcst_area:.2f}% of domain")
        st.metric("Peak ensemble probability", f"{peak_prob:.0f}%")
        st.metric("Ensemble spread", f"{metrics['spread'][idx]:.2f} mm/h")
        st.caption(
            "Bias = ensemble mean minus observed. Positive values mean the "
            "forecast is too wet on average."
        )

    with st.expander("Publication figure · observed versus forecast", expanded=False):
        key = f"png_obs_{result.run_id}"
        if st.button("Build comparison PNG", key="build_obs_png"):
            with st.spinner("Rendering comparison figure…"):
                fig = plotting.figure_obs_vs_mean(result)
                if fig is not None:
                    st.session_state[key] = plotting.fig_to_png_bytes(fig)
        if key in st.session_state:
            st.image(st.session_state[key], width="stretch")
            st.download_button(
                "Download PNG",
                data=st.session_state[key],
                file_name=f"surabaya_obs_vs_steps_{result.init_timestamp[:12]}.png",
                mime="image/png",
            )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def render_verification(result: NowcastResult) -> None:
    if not result.verified:
        st.info(
            "No verification metrics are stored for this run. Enable **Verify "
            "against future scans** before running to compute CRPS, Brier, ROC AUC "
            "and FSS."
        )
        return
    metrics = result.metrics
    mapping = {
        "CRPS · continuous": "crps",
        "Brier score · threshold": "brier",
        "ROC AUC · threshold": "auc",
        "FSS · neighbourhood": "fss",
        "MAE · ensemble mean": "mae",
        "Bias · ensemble mean": "bias",
        "Spatial correlation": "corr",
        "Ensemble spread": "spread",
    }
    c1, c2 = st.columns([1.4, 2.6])
    with c1:
        label = st.selectbox(
            "Verification view", list(mapping), index=0, key="ver_metric"
        )
    with c2:
        st.caption(
            "All scores are computed inside the 150 km radar domain and compared "
            "with the pseudo-CAPPI at the nearest future scan."
        )
    metric = mapping[label]
    chart = viz.verification_chart(result, metric)
    if chart is not None:
        st.plotly_chart(
            chart,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"ver_chart_{result.run_id}",
        )

    coverage_thr = st.selectbox(
        "Coverage threshold (mm/h)",
        options=[1.0, 2.0, 5.0, 10.0, 20.0],
        index=2,
        key="ver_coverage_thr",
    )
    coverage = viz.coverage_chart(result, coverage_thr)
    if coverage is not None:
        st.plotly_chart(
            coverage,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"coverage_{result.run_id}",
        )

    st.markdown("### Score table at key lead times")
    control_cols = st.columns([1, 1, 2])
    thresholds = [float(t) for t in metrics.get("thresholds", [])]
    scales = [int(s) for s in metrics.get("fss_scales_km", [])]
    with control_cols[0]:
        threshold_pick = st.selectbox(
            "Brier / AUC threshold",
            options=thresholds,
            index=thresholds.index(5.0) if 5.0 in thresholds else 0,
            format_func=lambda value: f"≥ {value:g} mm/h",
            key="ver_threshold",
        )
    with control_cols[1]:
        scale_pick = st.selectbox(
            "FSS neighbourhood",
            options=scales,
            index=scales.index(5) if 5 in scales else 0,
            format_func=lambda value: f"{value} km",
            key="ver_scale",
        )
    rows = []
    for idx in key_lead_indices(result):
        rows.append(
            {
                "Lead": result.lead_label(idx),
                "Valid (UTC)": result.valid_time_label(idx),
                "CRPS": metrics["crps"][idx],
                f"Brier ≥{threshold_pick:g}": metrics["brier"][f"{threshold_pick:g}"][idx],
                f"AUC ≥{threshold_pick:g}": metrics["roc_auc"][f"{threshold_pick:g}"][idx],
                f"FSS {scale_pick} km": metrics["fss"][f"{scale_pick:g}"][idx],
                "MAE": metrics["mae"][idx],
                "Bias": metrics["bias"][idx],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "CRPS": st.column_config.NumberColumn(format="%.3f"),
            f"Brier ≥{threshold_pick:g}": st.column_config.NumberColumn(format="%.4f"),
            f"AUC ≥{threshold_pick:g}": st.column_config.NumberColumn(format="%.3f"),
            f"FSS {scale_pick} km": st.column_config.NumberColumn(format="%.3f"),
            "MAE": st.column_config.NumberColumn(format="%.3f"),
            "Bias": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "CRPS is in mm/h. Brier and AUC are computed for the selected rain-rate "
        "threshold; FSS uses the selected neighbourhood."
    )

    dl_cols = st.columns(2)
    with dl_cols[0]:
        st.download_button(
            "Metrics CSV",
            data=csv_bytes(result),
            file_name=f"surabaya_verification_{result.init_timestamp[:12]}.csv",
            mime="text/csv",
            width="stretch",
            key="dl_metrics",
        )
    with dl_cols[1]:
        st.download_button(
            "Config JSON",
            data=json_bytes(result.config.to_dict()),
            file_name=f"surabaya_config_{result.init_timestamp[:12]}.json",
            mime="application/json",
            width="stretch",
        )

    with st.expander("Publication figure · 2 × 2 verification panel", expanded=False):
        key = f"png_verification_{result.run_id}"
        if st.button("Build verification PNG", key="build_verification_png"):
            with st.spinner("Rendering verification figure…"):
                fig = plotting.figure_verification(result)
                if fig is not None:
                    st.session_state[key] = plotting.fig_to_png_bytes(fig)
        if key in st.session_state:
            st.image(st.session_state[key], width="stretch")
            st.download_button(
                "Download PNG",
                data=st.session_state[key],
                file_name=f"surabaya_verification_{result.init_timestamp[:12]}.png",
                mime="image/png",
            )


# --------------------------------------------------------------------------- #
# Ensemble explorer
# --------------------------------------------------------------------------- #
def render_ensemble(result: NowcastResult) -> None:
    default_idx = nearest_lead_index(result, 60)
    controls = st.columns([2, 1])
    with controls[0]:
        lead_idx = st.select_slider(
            "Lead time",
            options=list(range(result.n_steps)),
            value=default_idx,
            format_func=result.lead_label,
            key="ens_lead",
        )
    with controls[1]:
        member = st.slider(
            "Ensemble member", 1, result.n_members, 1, key="ens_member"
        ) - 1
    st.caption(
        "STEPS members are stochastic realizations, not a deterministic control."
    )

    member_field = np.asarray(result.ensemble[member, lead_idx], dtype=np.float32).copy()
    member_field[~result.valid_mask] = np.nan
    with st.container(key="ensemble_maps"):
        map_cols = st.columns(2, gap="large")
        with map_cols[0]:
            st.plotly_chart(
                viz.rain_map(
                    result,
                    member_field,
                    f"Member {member + 1} rain rate",
                    f"{result.lead_label(lead_idx)} · {result.valid_time_label(lead_idx)}",
                    vmax=40,
                    height=540,
                ),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"ens_member_{result.run_id}",
            )
        with map_cols[1]:
            st.plotly_chart(
                viz.rain_map(
                    result,
                    result.mean_field(lead_idx),
                    "Ensemble mean rain rate",
                    f"{result.lead_label(lead_idx)} · {result.n_members} members",
                    vmax=40,
                    height=540,
                ),
                width="stretch",
                config=PLOT_CONFIG,
                key=f"ens_mean_{result.run_id}",
            )

    valid = result.valid_mask
    member_fields = np.asarray(result.ensemble[:, lead_idx], dtype=np.float32)
    means = member_fields[:, valid].mean(axis=1)
    maxima = member_fields[:, valid].max(axis=1)
    fractions = (member_fields[:, valid] >= 5.0).mean(axis=1) * 100.0
    table = pd.DataFrame(
        {
            "Member": np.arange(1, result.n_members + 1),
            "Domain mean (mm/h)": means,
            "Domain max (mm/h)": maxima,
            "Area ≥ 5 mm/h (%)": fractions,
        }
    )
    bottom_cols = st.container(), st.container()
    with bottom_cols[0]:
        fig = go.Figure(
            go.Bar(
                x=np.arange(1, result.n_members + 1),
                y=means,
                marker_color=[
                    "#0e7490" if i == member else "rgba(14,116,144,0.35)"
                    for i in range(result.n_members)
                ],
                hovertemplate="Member %{x} · %{y:.2f} mm/h<extra></extra>",
            )
        )
        fig.add_hline(
            y=float(np.mean(means)),
            line=dict(color="#dc2626", width=1, dash="dash"),
            annotation_text="ensemble mean",
        )
        fig.update_layout(
            template="plotly_white",
            height=330,
            margin=dict(l=48, r=12, t=40, b=40),
            title=dict(text=f"<b>Member domain means · {result.lead_label(lead_idx)}</b>", x=0.01),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#eef4f8",
            font=dict(color="#0f172a", size=12),
            showlegend=False,
        )
        fig.update_xaxes(title="Member", dtick=max(1, result.n_members // 10))
        fig.update_yaxes(title="Domain mean (mm/h)")
        st.plotly_chart(
            fig,
            width="stretch",
            config=PLOT_CONFIG,
            key=f"ens_hist_{result.run_id}",
        )
    with bottom_cols[1]:
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=330,
            column_config={
                "Domain mean (mm/h)": st.column_config.NumberColumn(format="%.2f"),
                "Domain max (mm/h)": st.column_config.NumberColumn(format="%.1f"),
                "Area ≥ 5 mm/h (%)": st.column_config.ProgressColumn(
                    "Area ≥ 5 mm/h (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )
    st.caption(
        "A wide spread between member means indicates high flow-dependent "
        "uncertainty; a narrow spread indicates the ensemble is confident."
    )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def render_diagnostics(result: NowcastResult) -> None:
    activity = result.meta.get("activity")
    init_index = result.meta.get("init_index")
    if activity:
        st.markdown("### Archive storm activity")
        fig = viz.activity_chart(activity, init_index=init_index)
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                config=PLOT_CONFIG,
                key=f"diag_activity_{result.run_id}",
            )

    st.markdown("### Motion field and initial rain")
    st.plotly_chart(
        viz.motion_map(result, show_labels=True),
        width="stretch",
        config=PLOT_CONFIG,
        key=f"diag_motion_{result.run_id}",
    )

    st.markdown("### Pseudo-CAPPI processing")
    cappi = result.meta.get("cappi", [])
    if cappi:
        rows = []
        for item in cappi:
            elevations = item.get("elevations_deg", [])
            rows.append(
                {
                    "Scan (UTC)": radar_core.timestamp_label(item.get("timestamp", "")),
                    "Sweeps": item.get("n_sweeps", 0),
                    "Elevations (°)": " · ".join(f"{e:g}" for e in elevations),
                    "Meteo pixels (%)": 100.0 * float(item.get("met_fraction", 0.0)),
                    "Max rain (mm/h)": item.get("max_rain_mmh", np.nan),
                    "Coverage (%)": 100.0 * float(item.get("coverage_fraction", 0.0)),
                    "Read / project (s)": item.get("elapsed_s", np.nan),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Meteo pixels (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Max rain (mm/h)": st.column_config.NumberColumn(format="%.1f"),
                "Coverage (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Read / project (s)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        if result.is_demo:
            st.caption(
                "Demo runs bypass the radar ingest/CAPPI chain; switch to a radar "
                "archive to inspect sweep counts, meteorological-echo fractions and "
                "per-scan processing time."
            )
        else:
            st.caption("CAPPI metadata is not stored for this run.")

    with st.expander("Run metadata and software versions", expanded=False):
        meta_payload = {
            "run_id": result.run_id,
            "created_at": result.created_at,
            "mode": result.mode,
            "init": result.init_label,
            "init_window": result.timestamps,
            "lead_times_min": result.lead_times_min.tolist(),
            "observations": result.obs_timestamps,
            "config": result.config.to_dict(),
            "run_meta": result.meta,
        }
        st.json(meta_payload, expanded=False)
        st.download_button(
            "Download full run metadata (JSON)",
            data=json_bytes(meta_payload),
            file_name=f"surabaya_run_{result.init_timestamp[:12]}.json",
            mime="application/json",
        )

    st.markdown("### Method and caveats")
    with st.expander("Read the method notes", expanded=False):
        st.markdown(
            """
**Processing chain**

1. **Rainbow5 ingest.** Each volume is decoded from the Baron Rainbow5 format.
   The 16-bit start-angle field is converted from its 0–65535 encoding to
   0–360°; this is essential for correct polar georeferencing.
2. **Dual-pol quality control.** Echoes are retained when dBZ ≥ 5, ρHV ≥ 0.85
   and |ZDR| ≤ 5 dB. ΦDP is unwrapped, smoothed and converted to path-integrated
   attenuation with a conservative 10 dB cap.
3. **Pseudo-CAPPI.** Sweeps up to 12° are projected to a local tangent plane and
   nearest-neighbour sampled onto a 1 km Cartesian grid. Rain rate is linearly
   interpolated vertically to 1.5 km ASL using the 4/3-Earth beam-height model.
4. **Z–R conversion.** $Z = 250 R^{1.2}$ (tropical convective relation), with
   $R$ in mm/h and $Z$ in mm⁶ m⁻³.
5. **Motion and ensemble.** Lucas–Kanade optical flow provides the advection
   field; STEPS decomposes the field into cascade levels, evolves them with an
   AR(2) model and adds stochastic perturbations. Twenty members are generated
   over the two-hour horizon.
6. **Verification.** Forecasts are compared with pseudo-CAPPIs from the nearest
   future scans. CRPS measures the full distribution; Brier and ROC AUC measure
   threshold events; FSS measures neighbourhood structure; MAE, bias and
   correlation describe the ensemble mean.

**Caveats**

- Rain rates are radar-derived and sensitive to the Z–R relation, calibration,
  beam blockage and the vertical interpolation to 1.5 km.
- Verification uses radar as “truth”; it is not gauge verification.
- STEPS assumes approximately Lagrangian persistence of the echo field.
- District exposure is a spatial average of gridded probabilities, not a
  hydrological flood forecast.
- This is a research and training prototype, not an operational warning product.
            """
        )
        st.caption(
            "References: Seed (2003); Lucas & Kanade (1981); Roberts & Lean (2008); "
            "Pulkkinen et al. (2019) pysteps; geoBoundaries ADM2."
        )


# --------------------------------------------------------------------------- #
# Result page
# --------------------------------------------------------------------------- #
def render_result(result: NowcastResult) -> None:
    render_result_header(result)
    tabs = st.tabs(
        [
            "Overview",
            "Probability",
            "Observed vs forecast",
            "Verification",
            "Ensemble",
            "Diagnostics",
        ]
    )
    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_probability(result)
    with tabs[2]:
        render_obs_forecast(result)
    with tabs[3]:
        render_verification(result)
    with tabs[4]:
        render_ensemble(result)
    with tabs[5]:
        render_diagnostics(result)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    config, run_clicked, source_mode, inventory, inventory_error, archive_path = (
        render_sidebar()
    )

    if run_clicked:
        if config is None:
            st.error("Fix the configuration before running.")
        else:
            execute_run(config)
            return

    result = st.session_state.get("result")
    if result is None:
        render_landing(
            config, source_mode, inventory, inventory_error, archive_path
        )
    else:
        render_result(result)


if __name__ == "__main__":
    main()
