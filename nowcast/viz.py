"""Plotly figures for the Streamlit dashboard."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import NowcastResult

PRIMARY = "#0e7490"
ACCENT = "#f59e0b"
PALETTE = ["#0e7490", "#f59e0b", "#dc2626", "#2563eb", "#059669", "#7c3aed", "#db2777"]
# Light-at-low-end rain scale: avoids a dark purple wash over the whole radar disc.
RAIN_SCALE = [
    [0.00, "#f8fafc"],
    [0.04, "#dbeafe"],
    [0.12, "#93c5fd"],
    [0.24, "#22d3ee"],
    [0.38, "#34d399"],
    [0.55, "#facc15"],
    [0.72, "#fb923c"],
    [0.88, "#dc2626"],
    [1.00, "#7e22ce"],
]
DIFF_SCALE = [[0.0, "#2166ac"], [0.5, "#f8fafc"], [1.0, "#b2182b"]]
TEXT = "#0f172a"
MUTED = "#475569"
PLOT_BG = "#eef4f8"


def _base_layout(fig: go.Figure, height: int = 620, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12, color=TEXT),
        margin=dict(l=56, r=18, t=100 if title else 38, b=48),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.7)",
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#cbd5e1"),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=17)))
    return fig


def _km_axes(fig: go.Figure, extent: float) -> None:
    fig.update_xaxes(
        range=[-extent, extent],
        title="East–west distance from radar (km)",
        gridcolor="rgba(148,163,184,0.25)",
        zeroline=False,
    )
    fig.update_yaxes(
        range=[-extent, extent],
        title="North–south distance from radar (km)",
        gridcolor="rgba(148,163,184,0.25)",
        zeroline=False,
    )


def _segments_as_xy(segments) -> tuple[list, list]:
    xs: list = []
    ys: list = []
    for segment in segments:
        arr = np.asarray(segment, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        xs.extend(arr[:, 0].tolist())
        ys.extend(arr[:, 1].tolist())
        xs.append(None)
        ys.append(None)
    return xs, ys


def _add_district_layers(
    fig: go.Figure,
    result: NowcastResult,
    show_labels: bool = False,
    show_rings: bool = True,
    row: int | None = None,
    col: int | None = None,
) -> None:
    kwargs = {"row": row, "col": col} if row is not None else {}
    districts = result.districts or {}
    xs, ys = _segments_as_xy(districts.get("other_segments", []))
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="rgba(51,65,85,0.55)", width=1),
                hoverinfo="skip",
                showlegend=False,
                name="District boundaries",
            ),
            **kwargs,
        )
    xs, ys = _segments_as_xy(districts.get("surabaya_segments", []))
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#0f172a", width=2),
                hoverinfo="skip",
                showlegend=False,
                name="Surabaya",
            ),
            **kwargs,
        )

    if show_rings:
        theta = np.linspace(0, 2 * np.pi, 240)
        for radius in (50, 100, result.extent_km):
            if radius > result.extent_km * 1.01:
                continue
            fig.add_trace(
                go.Scatter(
                    x=radius * np.cos(theta),
                    y=radius * np.sin(theta),
                    mode="lines",
                    line=dict(color="rgba(51,65,85,0.38)", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                **kwargs,
            )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(symbol="star", size=10, color="#0f172a", line=dict(color="white", width=1)),
            hovertemplate="Surabaya Baron radar<extra></extra>",
            showlegend=False,
        ),
        **kwargs,
    )

    if show_labels:
        labels = districts.get("labels", [])
        # Keep the densest part of the domain readable
        for cx, cy, name in labels:
            if np.hypot(cx, cy) > result.extent_km * 0.94:
                continue
            fig.add_annotation(
                x=float(cx),
                y=float(cy),
                text=str(name),
                showarrow=False,
                font=dict(
                    size=9,
                    color=TEXT,
                    family="Inter, sans-serif",
                ),
                bgcolor="rgba(255,255,255,0.58)",
                borderpad=1,
                **kwargs,
            )


def _heatmap(
    z: np.ndarray,
    coords: np.ndarray,
    colorscale,
    zmin: float,
    zmax: float,
    colorbar_title: str,
    hover: str,
    zmid: float | None = None,
    showscale: bool = True,
) -> go.Heatmap:
    return go.Heatmap(
        z=z,
        x=coords,
        y=coords,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,
        showscale=showscale,
        connectgaps=False,
        hoverongaps=False,
        colorbar=dict(
            title=dict(text=colorbar_title, side="top"),
            thickness=14,
            len=0.78,
            outlinewidth=0,
            tickfont=dict(size=11),
        ),
        hovertemplate=hover,
    )


# ------------------------------------------------------------------- maps
def probability_map(
    result: NowcastResult,
    lead_idx: int,
    threshold: float,
    min_probability: float = 1.0,
    show_labels: bool = True,
    show_rings: bool = True,
) -> go.Figure:
    prob = result.probability_field(lead_idx, threshold)
    plot = np.where(np.isfinite(prob) & (prob >= min_probability), prob, np.nan)
    coords = result.coords_km
    lead = result.lead_label(lead_idx)
    valid = result.valid_time_label(lead_idx)
    title = (
        f"<b>Rain ≥ {threshold:g} mm/h</b> · exceedance probability"
        f"<br><span style='font-size:12px;color:{MUTED}'>STEPS ensemble · "
        f"{lead} ({valid}) · init {result.init_label}</span>"
    )
    fig = go.Figure(
        _heatmap(
            plot,
            coords,
            "YlOrRd",
            0,
            100,
            "Probability (%)",
            "<b>%{z:.0f}%</b> chance of ≥ "
            f"{threshold:g} mm/h<br>x %{{x:.0f}} km · y %{{y:.0f}} km<extra></extra>",
        )
    )
    _add_district_layers(fig, result, show_labels, show_rings)
    _base_layout(fig, 640, title)
    _km_axes(fig, result.extent_km)
    fig.update_layout(uirevision=f"{result.run_id}-prob")
    return fig


def rain_map(
    result: NowcastResult,
    field: np.ndarray,
    title: str,
    subtitle: str = "",
    vmax: float = 40.0,
    colorscale=RAIN_SCALE,
    unit: str = "mm/h",
    show_labels: bool = False,
    show_rings: bool = True,
    height: int = 560,
) -> go.Figure:
    plot = np.where(np.isfinite(field) & (field >= 0.1), field, np.nan)
    main = f"<b>{title}</b>"
    if subtitle:
        main += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
    fig = go.Figure(
        _heatmap(
            plot,
            result.coords_km,
            colorscale,
            0,
            vmax,
            unit,
            "<b>%{z:.1f}</b> " + unit + "<br>x %{x:.0f} km · y %{y:.0f} km<extra></extra>",
        )
    )
    _add_district_layers(fig, result, show_labels, show_rings)
    _base_layout(fig, height, main)
    _km_axes(fig, result.extent_km)
    fig.update_layout(uirevision=f"{result.run_id}-rain")
    return fig


def initial_map(result: NowcastResult, show_labels: bool = True) -> go.Figure:
    return rain_map(
        result,
        result.init_rain,
        f"{result.config.target_alt_m / 1000:.1f} km pseudo-CAPPI at initialization",
        f"{result.init_label} · observed rain rate",
        vmax=40,
        show_labels=show_labels,
        height=560,
    )


def obs_vs_forecast(result: NowcastResult, lead_idx: int, show_labels: bool = False) -> go.Figure:
    coords = result.coords_km
    mean = result.mean_field(lead_idx)
    lead = result.lead_label(lead_idx)
    valid = result.valid_time_label(lead_idx)
    obs = result.observed_field(lead_idx)
    extent = result.extent_km

    if obs is None:
        spread = result.spread_field(lead_idx)
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.06,
            subplot_titles=("Ensemble mean rain rate", "Ensemble spread (std. dev.)"),
        )
        fig.add_trace(
            _heatmap(
                np.where(mean >= 0.1, mean, np.nan),
                coords,
                RAIN_SCALE,
                0,
                40,
                "mm/h",
                "<b>%{z:.1f}</b> mm/h<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            _heatmap(
                np.where(spread >= 0.05, spread, np.nan),
                coords,
                "Magma",
                0,
                max(1.0, float(np.nanpercentile(spread, 99)) if np.isfinite(spread).any() else 1.0),
                "mm/h",
                "<b>%{z:.1f}</b> mm/h<extra></extra>",
            ),
            row=1,
            col=2,
        )
        _add_district_layers(fig, result, show_labels, True, row=1, col=1)
        _add_district_layers(fig, result, False, False, row=1, col=2)
        _base_layout(
            fig,
            560,
            f"<b>STEPS ensemble at {lead}</b><br><span style='font-size:12px;color:{MUTED}'>"
            f"Init {result.init_label} · verification disabled</span>",
        )
        for axis in ("xaxis", "xaxis2"):
            fig.layout[axis].update(
                range=[-extent, extent],
                title="East–west (km)",
                gridcolor="rgba(148,163,184,0.25)",
            )
        for name in ("yaxis", "yaxis2"):
            fig.layout[name].update(
                range=[-extent, extent],
                title="North–south (km)",
                gridcolor="rgba(148,163,184,0.25)",
            )
        return fig

    diff = mean - obs
    vmax = 40.0
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        horizontal_spacing=0.07,
        vertical_spacing=0.16,
        subplot_titles=(
            f"Observed · {valid}",
            f"Ensemble mean · {lead}",
            "Mean − observed",
        ),
    )
    fig.add_trace(
        _heatmap(
            np.where(obs >= 0.1, obs, np.nan),
            coords,
            RAIN_SCALE,
            0,
            vmax,
            "mm/h",
            "<b>%{z:.1f}</b> mm/h<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _heatmap(
            np.where(mean >= 0.1, mean, np.nan),
            coords,
            RAIN_SCALE,
            0,
            vmax,
            "mm/h",
            "<b>%{z:.1f}</b> mm/h<extra></extra>",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    diff_plot = np.where(np.isfinite(diff), diff, np.nan)
    fig.add_trace(
        _heatmap(
            diff_plot,
            coords,
            DIFF_SCALE,
            -10,
            10,
            "mm/h",
            "<b>%{z:+.1f}</b> mm/h<extra></extra>",
            zmid=0,
        ),
        row=2,
        col=1,
    )
    _add_district_layers(fig, result, show_labels, True, row=1, col=1)
    _add_district_layers(fig, result, False, False, row=1, col=2)
    _add_district_layers(fig, result, False, True, row=2, col=1)
    _base_layout(
        fig,
        900,
        f"<b>Observed versus STEPS forecast</b> · {lead}<br>"
        f"<span style='font-size:12px;color:{MUTED}'>Init {result.init_label} · "
        f"difference field capped at ±10 mm/h</span>",
    )
    for ann in fig.layout.annotations:
        if ann.xref == "paper" and ann.yref == "paper":
            ann.font.size = 12
    fig.update_layout(margin=dict(t=132, r=100))
    # Place the rain and difference colour bars in separate vertical bands
    # instead of letting Plotly stack both at the same right-hand position.
    rain_bar = fig.data[0].colorbar
    rain_bar.x = 1.015
    rain_bar.y = 0.78
    rain_bar.len = 0.36
    rain_bar.title.text = "Rain (mm/h)"
    diff_bar = fig.data[2].colorbar
    diff_bar.x = 1.015
    diff_bar.y = 0.12
    diff_bar.len = 0.36
    diff_bar.title.text = "Diff (mm/h)"
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        fig.layout[axis].update(
            range=[-extent, extent],
            gridcolor="rgba(148,163,184,0.25)",
            zeroline=False,
        )
    for axis in ("yaxis", "yaxis2", "yaxis3"):
        fig.layout[axis].update(
            range=[-extent, extent],
            gridcolor="rgba(148,163,184,0.25)",
            zeroline=False,
        )
    fig.layout["xaxis"].update(title="East–west (km)")
    fig.layout["xaxis3"].update(title="East–west (km)")
    fig.layout["yaxis"].update(title="North–south (km)")
    fig.layout["yaxis3"].update(title="North–south (km)")
    return fig


# ----------------------------------------------------------- verification
def verification_chart(
    result: NowcastResult,
    metric: str = "crps",
) -> go.Figure | None:
    metrics = result.metrics or {}
    if not metrics:
        return None
    leads = np.asarray(result.lead_times_min, dtype=float)
    metric = metric.lower()

    if metric == "crps":
        fig = go.Figure(
            go.Scatter(
                x=leads,
                y=metrics["crps"],
                mode="lines+markers",
                name="CRPS",
                line=dict(color=PRIMARY, width=2.4),
                marker=dict(size=6),
                hovertemplate="%{x:.0f} min · CRPS %{y:.3f} mm/h<extra></extra>",
            )
        )
        title = "Continuous Ranked Probability Score (lower is better)"
        ytitle = "CRPS (mm/h)"
        yrange = None
    elif metric == "brier":
        fig = go.Figure()
        for i, thr in enumerate(metrics.get("thresholds", [])):
            key = f"{float(thr):g}"
            fig.add_trace(
                go.Scatter(
                    x=leads,
                    y=metrics["brier"].get(key, []),
                    mode="lines+markers",
                    name=f"≥ {float(thr):g} mm/h",
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                    marker=dict(size=5),
                    hovertemplate="%{x:.0f} min · BS %{y:.3f}<extra></extra>",
                )
            )
        title = "Brier score by rain-rate threshold (lower is better)"
        ytitle = "Brier score"
        yrange = None
    elif metric == "auc":
        fig = go.Figure()
        for i, thr in enumerate(metrics.get("thresholds", [])):
            key = f"{float(thr):g}"
            fig.add_trace(
                go.Scatter(
                    x=leads,
                    y=metrics["roc_auc"].get(key, []),
                    mode="lines+markers",
                    name=f"≥ {float(thr):g} mm/h",
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                    marker=dict(size=5),
                    hovertemplate="%{x:.0f} min · AUC %{y:.3f}<extra></extra>",
                )
            )
        fig.add_hline(y=0.7, line=dict(color="#b91c1c", width=1, dash="dash"), annotation_text="useful (0.70)")
        fig.add_hline(y=0.5, line=dict(color="#64748b", width=1, dash="dot"), annotation_text="no skill (0.50)")
        title = "ROC area under curve by threshold (higher is better)"
        ytitle = "ROC AUC"
        yrange = [0.45, 1.0]
    elif metric == "fss":
        fig = go.Figure()
        for i, scale in enumerate(metrics.get("fss_scales_km", [])):
            key = f"{float(scale):g}"
            fig.add_trace(
                go.Scatter(
                    x=leads,
                    y=metrics["fss"].get(key, []),
                    mode="lines+markers",
                    name=f"{float(scale):g} km neighbourhood",
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                    marker=dict(size=5),
                    hovertemplate="%{x:.0f} min · FSS %{y:.3f}<extra></extra>",
                )
            )
        fig.add_hline(y=0.5, line=dict(color="#b91c1c", width=1, dash="dash"), annotation_text="skillful (0.50)")
        title = (
            f"Fractions Skill Score at ≥ {float(metrics.get('fss_thr', 5)):g} mm/h "
            "(higher is better)"
        )
        ytitle = "FSS"
        yrange = [0.0, 1.0]
    elif metric in {"mae", "bias", "corr", "spread"}:
        labels = {
            "mae": ("Mean absolute error", "MAE (mm/h)"),
            "bias": ("Signed bias of the ensemble mean", "Bias (mm/h)"),
            "corr": ("Spatial correlation of ensemble mean", "Pearson r"),
            "spread": ("Ensemble spread", "Std. dev. (mm/h)"),
        }
        label, ytitle = labels[metric]
        fig = go.Figure(
            go.Scatter(
                x=leads,
                y=metrics.get(metric, []),
                mode="lines+markers",
                name=label,
                line=dict(color=PRIMARY, width=2.4),
                marker=dict(size=6),
                hovertemplate="%{x:.0f} min · %{y:.3f}<extra></extra>",
            )
        )
        title = label
        yrange = None
    else:
        return None

    _base_layout(fig, 460, f"<b>{title}</b><br><span style='font-size:12px;color:{MUTED}'>"
                          f"Init {result.init_label} · {result.n_members} members</span>")
    fig.update_xaxes(title="Lead time (min)", gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(title=ytitle, range=yrange, gridcolor="rgba(148,163,184,0.25)")
    fig.update_layout(hovermode="x unified", margin=dict(l=64, r=24, t=90, b=52))
    return fig


def coverage_chart(result: NowcastResult, threshold: float = 5.0) -> go.Figure | None:
    if result.obs is None or not result.metrics:
        return None
    leads = result.lead_times_min
    prob_area = []
    mean_area = []
    obs_area = []
    peak_prob = []
    for step in range(result.n_steps):
        prob = result.probability_field(step, threshold)
        mean_field = result.mean_field(step)
        obs = result.observed_field(step)
        prob_area.append(float(np.nanmean(prob[result.valid_mask]) / 100.0 * 100.0))
        mean_area.append(float(np.nanmean((mean_field >= threshold)[result.valid_mask]) * 100.0))
        if obs is not None:
            obs_area.append(float(np.nanmean((obs >= threshold)[result.valid_mask]) * 100.0))
        peak_prob.append(float(np.nanmax(prob)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=leads,
            y=prob_area,
            mode="lines+markers",
            name="Domain-mean probability",
            line=dict(color=PRIMARY, width=2.4),
            hovertemplate="%{x:.0f} min · %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=leads,
            y=mean_area,
            mode="lines+markers",
            name="Forecast area ≥ threshold",
            line=dict(color=ACCENT, width=2.2),
            hovertemplate="%{x:.0f} min · %{y:.1f}% of domain<extra></extra>",
        )
    )
    if obs_area:
        fig.add_trace(
            go.Scatter(
                x=leads,
                y=obs_area,
                mode="lines+markers",
                name="Observed area ≥ threshold",
                line=dict(color="#111827", width=2.2, dash="dot"),
                hovertemplate="%{x:.0f} min · %{y:.1f}% of domain<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=leads,
            y=peak_prob,
            mode="lines+markers",
            name="Peak probability",
            line=dict(color="#dc2626", width=1.8, dash="dash"),
            hovertemplate="%{x:.0f} min · peak %{y:.0f}%<extra></extra>",
        )
    )
    _base_layout(
        fig,
        420,
        f"<b>Rain coverage and probability evolution · ≥ {threshold:g} mm/h</b>",
    )
    fig.update_layout(margin=dict(t=128))
    fig.update_xaxes(title="Lead time (min)", gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(title="Percent / area fraction (%)", gridcolor="rgba(148,163,184,0.25)")
    fig.update_layout(hovermode="x unified")
    return fig


def activity_chart(activity: list[dict], init_index: int | None = None) -> go.Figure | None:
    if not activity:
        return None
    from datetime import datetime

    times = []
    for row in activity:
        try:
            times.append(datetime.strptime(row["timestamp"][:14], "%Y%m%d%H%M%S"))
        except Exception:  # noqa: BLE001
            times.append(None)
    if any(t is None for t in times):
        times = list(range(len(activity)))
    pixels = [row.get("convective_pixels", 0) for row in activity]
    max_dbz = [row.get("max_dbz", np.nan) for row in activity]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=times,
            y=pixels,
            name="Pixels ≥ 25 dBZ",
            marker_color="rgba(14,116,144,0.55)",
            hovertemplate="%{x|%H:%M} UTC · %{y:,} pixels<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=max_dbz,
            mode="lines",
            name="Max dBZ (first sweep)",
            line=dict(color="#f59e0b", width=2),
            hovertemplate="%{x|%H:%M} UTC · %{y:.1f} dBZ<extra></extra>",
        ),
        secondary_y=True,
    )
    if init_index is not None and 0 <= init_index < len(times):
        fig.add_vline(
            x=times[init_index],
            line=dict(color="#dc2626", width=2, dash="dash"),
            annotation_text="selected init",
            annotation_position="top left",
        )
    _base_layout(fig, 330, "<b>Storm activity through the archive</b>")
    fig.update_xaxes(title="Time (UTC)", gridcolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(title="Pixels ≥ 25 dBZ (first sweep)", secondary_y=False)
    fig.update_yaxes(title="Max dBZ", secondary_y=True, range=[0, 70])
    fig.update_layout(bargap=0.05, hovermode="x unified")
    return fig


def scatter_obs_forecast(result: NowcastResult, lead_idx: int) -> go.Figure | None:
    obs = result.observed_field(lead_idx)
    mean = result.mean_field(lead_idx)
    if obs is None:
        return None
    mask = result.valid_mask & np.isfinite(obs) & np.isfinite(mean)
    x = obs[mask].astype(float)
    y = mean[mask].astype(float)
    if x.size == 0:
        return None
    if x.size > 5000:
        idx = np.linspace(0, x.size - 1, 5000).astype(int)
        x, y = x[idx], y[idx]
    x_clip = np.clip(x, 0, 40)
    y_clip = np.clip(y, 0, 40)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_clip,
            y=y_clip,
            mode="markers",
            marker=dict(size=4, color="rgba(14,116,144,0.35)", line=dict(width=0)),
            name="Grid pixels",
            hovertemplate="obs %{x:.1f} · mean %{y:.1f} mm/h<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 40],
            y=[0, 40],
            mode="lines",
            line=dict(color="#dc2626", width=1.5, dash="dash"),
            name="1:1 line",
            hoverinfo="skip",
        )
    )
    _base_layout(
        fig,
        360,
        f"<b>Observed vs ensemble mean · {result.lead_label(lead_idx)}</b><br>"
        f"<span style='font-size:12px;color:{MUTED}'>Spatial correlation "
        f"{corr:.2f} · values capped at 40 mm/h</span>",
    )
    fig.update_xaxes(title="Observed rain rate (mm/h)", range=[0, 40], gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(title="Ensemble mean (mm/h)", range=[0, 40], gridcolor="rgba(148,163,184,0.25)")
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=0))
    return fig


def motion_map(result: NowcastResult, show_labels: bool = True) -> go.Figure:
    field = np.where(result.init_rain >= 0.1, result.init_rain, np.nan)
    fig = go.Figure(
        _heatmap(
            field,
            result.coords_km,
            RAIN_SCALE,
            0,
            40,
            "mm/h",
            "<b>%{z:.1f}</b> mm/h<extra></extra>",
        )
    )
    _add_district_layers(fig, result, show_labels, True)

    coords = result.coords_km
    spacing = max(1, int(round(20.0 / result.res_km)))
    gx, gy = np.meshgrid(coords, coords)
    u = np.asarray(result.velocity[0], dtype=float) * result.res_km / result.timestep_min * 60.0
    v = np.asarray(result.velocity[1], dtype=float) * result.res_km / result.timestep_min * 60.0
    xs: list = []
    ys: list = []
    tips_x: list = []
    tips_y: list = []
    for j in range(spacing // 2, len(coords), spacing):
        for i in range(spacing // 2, len(coords), spacing):
            if not result.valid_mask[i, j]:
                continue
            speed = float(np.hypot(u[i, j], v[i, j]))
            if not np.isfinite(speed) or speed < 0.8:
                continue
            length = min(18.0, 0.45 * speed)
            ux = u[i, j] / speed
            vy = v[i, j] / speed
            x0, y0 = float(gx[i, j]), float(gy[i, j])
            x1 = x0 + ux * length
            y1 = y0 + vy * length
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            tips_x.append(x1)
            tips_y.append(y1)
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="rgba(15,23,42,0.75)", width=1.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=tips_x,
                y=tips_y,
                mode="markers",
                marker=dict(size=4, color="#0f172a"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    _base_layout(
        fig,
        600,
        f"<b>Optical-flow motion field at initialization</b><br>"
        f"<span style='font-size:12px;color:{MUTED}'>Line length ∝ speed · "
        f"{result.init_label}</span>",
    )
    _km_axes(fig, result.extent_km)
    fig.update_layout(uirevision=f"{result.run_id}-motion")
    return fig


def district_bars(rows: list[dict], threshold: float, lead_label: str) -> go.Figure | None:
    if not rows:
        return None
    rows = rows[:10][::-1]
    names = [r["District"] for r in rows]
    values = [r["Mean probability (%)"] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=values,
                colorscale="YlOrRd",
                cmin=0,
                cmax=100,
                line=dict(width=0),
            ),
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y} · mean %{x:.1f}%<extra></extra>",
        )
    )
    _base_layout(
        fig,
        360,
        f"<b>District exposure at {lead_label}</b><br>"
        f"<span style='font-size:12px;color:{MUTED}'>Mean probability of ≥ {threshold:g} mm/h</span>",
    )
    fig.update_xaxes(title="Mean exceedance probability (%)", range=[0, 108], gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(title="")
    return fig
