"""Matplotlib exports for reports, posters and offline sharing."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from .models import NowcastResult
from .viz import RAIN_SCALE

MAP_CMAP = LinearSegmentedColormap.from_list(
    "rain_light", [(stop, color) for stop, color in RAIN_SCALE]
)
MAP_CMAP.set_bad(alpha=0.0)
PROB_CMAP = "YlOrRd"


def _segments(result: NowcastResult, key: str):
    return [np.asarray(seg, dtype=float) for seg in result.districts.get(key, [])]


def add_district_overlay(
    ax,
    result: NowcastResult,
    show_labels: bool = True,
    show_rings: bool = True,
    font_scale: float = 1.0,
) -> None:
    other = _segments(result, "other_segments")
    surabaya = _segments(result, "surabaya_segments")
    if other:
        ax.add_collection(
            LineCollection(other, colors="#334155", linewidths=0.7, alpha=0.7, zorder=4)
        )
    if surabaya:
        ax.add_collection(
            LineCollection(surabaya, colors="#000000", linewidths=1.6, alpha=0.95, zorder=5)
        )

    if show_rings:
        theta = np.linspace(0, 2 * np.pi, 360)
        for radius in (50, 100, result.extent_km):
            if radius > result.extent_km * 1.01:
                continue
            ax.plot(
                radius * np.cos(theta),
                radius * np.sin(theta),
                color="#334155",
                linestyle=":",
                linewidth=0.6,
                alpha=0.55,
                zorder=3,
            )

    if show_labels:
        for cx, cy, label in result.districts.get("labels", []):
            if np.hypot(cx, cy) > result.extent_km * 0.95:
                continue
            is_sby = "surabaya" in str(label).lower()
            ax.text(
                cx,
                cy,
                label,
                fontsize=(7.5 if is_sby else 6.4) * font_scale,
                fontweight="bold" if is_sby else "normal",
                color="#111111",
                ha="center",
                va="center",
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)],
            )

    ax.plot(0, 0, "k^", markersize=6, zorder=7)
    ax.set_xlim(-result.extent_km, result.extent_km)
    ax.set_ylim(-result.extent_km, result.extent_km)
    ax.set_xlabel("East–West distance from radar (km)")
    ax.set_ylabel("North–South distance from radar (km)")


def _extent(result: NowcastResult) -> list[float]:
    return [-result.extent_km, result.extent_km, -result.extent_km, result.extent_km]


def _title(result: NowcastResult, suffix: str) -> str:
    tag = " · SYNTHETIC DEMO" if result.is_demo else ""
    return (
        f"BMKG Surabaya Baron GEN3-350CM · {result.config.target_alt_m / 1000:.1f} km "
        f"pseudo-CAPPI — {suffix} (Init: {result.init_label}){tag}"
    )


def figure_probability_maps(
    result: NowcastResult,
    lead_idx: int,
    thresholds: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0),
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    lead = result.lead_label(lead_idx)
    valid = result.valid_time_label(lead_idx)
    for ax, threshold in zip(axes.ravel(), thresholds):
        prob = result.probability_field(lead_idx, threshold)
        plot = np.where(np.isfinite(prob) & (prob >= 5.0), prob, np.nan)
        im = ax.imshow(
            plot,
            origin="lower",
            extent=_extent(result),
            cmap=PROB_CMAP,
            vmin=0,
            vmax=100,
            zorder=2,
        )
        add_district_overlay(ax, result)
        ax.set_title(
            f"P(Rain ≥ {threshold:g} mm/h) at {lead} · {valid}",
            fontsize=11,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, label="Exceedance probability (%)", shrink=0.82)
    fig.suptitle(
        _title(result, f"STEPS exceedance probabilities at {lead}"),
        fontsize=13,
        fontweight="bold",
    )
    return fig


def figure_probability_evolution(
    result: NowcastResult,
    threshold: float = 5.0,
    lead_indices: tuple[int, ...] | None = None,
) -> plt.Figure:
    if lead_indices is None:
        lead_indices = tuple(
            int(round(v)) for v in np.linspace(0, result.n_steps - 1, 4)
        )
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    for ax, idx in zip(axes.ravel(), lead_indices):
        idx = int(np.clip(idx, 0, result.n_steps - 1))
        prob = result.probability_field(idx, threshold)
        plot = np.where(np.isfinite(prob) & (prob >= 5.0), prob, np.nan)
        im = ax.imshow(
            plot,
            origin="lower",
            extent=_extent(result),
            cmap=PROB_CMAP,
            vmin=0,
            vmax=100,
            zorder=2,
        )
        add_district_overlay(ax, result)
        ax.set_title(
            f"P(Rain ≥ {threshold:g} mm/h) at {result.lead_label(idx)} · "
            f"{result.valid_time_label(idx)}",
            fontsize=11,
            fontweight="bold",
        )
        fig.colorbar(im, ax=ax, label="Exceedance probability (%)", shrink=0.82)
    fig.suptitle(
        _title(result, f"2-hour probability evolution · ≥ {threshold:g} mm/h"),
        fontsize=13,
        fontweight="bold",
    )
    return fig


def figure_obs_vs_mean(
    result: NowcastResult,
    lead_indices: tuple[int, ...] | None = None,
) -> plt.Figure | None:
    if result.obs is None:
        return None
    if lead_indices is None:
        lead_indices = tuple(
            int(round(v)) for v in np.linspace(0, result.n_steps - 1, 4)
        )
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), constrained_layout=True)
    for col, idx in enumerate(lead_indices):
        idx = int(np.clip(idx, 0, result.n_steps - 1))
        obs = result.observed_field(idx)
        mean = result.mean_field(idx)
        obs_plot = np.where(obs >= 0.1, obs, np.nan)
        mean_plot = np.where(mean >= 0.1, mean, np.nan)
        im_obs = axes[0, col].imshow(
            obs_plot,
            origin="lower",
            extent=_extent(result),
            cmap=MAP_CMAP,
            vmin=0,
            vmax=35,
            zorder=2,
        )
        add_district_overlay(
            axes[0, col], result, show_labels=(col == 0), show_rings=False, font_scale=0.72
        )
        axes[0, col].set_title(
            f"Observed {result.lead_label(idx)} ({result.valid_time_label(idx)})",
            fontsize=10,
            fontweight="bold",
        )
        im_mean = axes[1, col].imshow(
            mean_plot,
            origin="lower",
            extent=_extent(result),
            cmap=MAP_CMAP,
            vmin=0,
            vmax=35,
            zorder=2,
        )
        add_district_overlay(
            axes[1, col], result, show_labels=(col == 0), show_rings=False, font_scale=0.72
        )
        axes[1, col].set_title(
            f"STEPS ensemble mean {result.lead_label(idx)}",
            fontsize=10,
            fontweight="bold",
        )
    fig.colorbar(im_obs, ax=axes[0, :], label="Observed rain rate (mm/h)", shrink=0.78, pad=0.01)
    fig.colorbar(im_mean, ax=axes[1, :], label="Ensemble mean rain rate (mm/h)", shrink=0.78, pad=0.01)
    fig.suptitle(
        _title(result, "Observed versus STEPS ensemble mean"),
        fontsize=13,
        fontweight="bold",
    )
    return fig


def figure_verification(result: NowcastResult) -> plt.Figure | None:
    metrics = result.metrics or {}
    if not metrics:
        return None
    leads = np.asarray(result.lead_times_min, dtype=float)
    thresholds = [float(t) for t in metrics.get("thresholds", result.config.thresholds)]
    scales = [int(s) for s in metrics.get("fss_scales_km", result.config.fss_scales_km)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    axes[0, 0].plot(leads, metrics["crps"], "o-", color="#1b365d", linewidth=2, markersize=4)
    axes[0, 0].set_title("(a) Continuous Ranked Probability Score (lower = better)", fontweight="bold")
    axes[0, 0].set_xlabel("Lead time (min)")
    axes[0, 0].set_ylabel("CRPS (mm/h)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    for threshold in thresholds:
        key = f"{threshold:g}"
        axes[0, 1].plot(
            leads,
            metrics["brier"].get(key, []),
            "o-",
            linewidth=2,
            markersize=4,
            label=f"≥ {threshold:g} mm/h",
        )
    axes[0, 1].set_title("(b) Brier score by rain threshold (lower = better)", fontweight="bold")
    axes[0, 1].set_xlabel("Lead time (min)")
    axes[0, 1].set_ylabel("Brier score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    for threshold in thresholds:
        key = f"{threshold:g}"
        axes[1, 0].plot(
            leads,
            metrics["roc_auc"].get(key, []),
            "o-",
            linewidth=2,
            markersize=4,
            label=f"≥ {threshold:g} mm/h",
        )
    axes[1, 0].axhline(0.7, color="#b22222", linestyle="--", linewidth=1, label="Useful skill (0.70)")
    axes[1, 0].axhline(0.5, color="gray", linestyle=":", linewidth=1, label="No skill (0.50)")
    axes[1, 0].set_title("(c) ROC area under curve (higher = better)", fontweight="bold")
    axes[1, 0].set_xlabel("Lead time (min)")
    axes[1, 0].set_ylabel("ROC AUC")
    axes[1, 0].set_ylim(0.45, 1.0)
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    for scale in scales:
        key = f"{scale:g}"
        axes[1, 1].plot(
            leads,
            metrics["fss"].get(key, []),
            "o-",
            linewidth=2,
            markersize=4,
            label=f"{scale} km × {scale} km",
        )
    axes[1, 1].axhline(0.5, color="#b22222", linestyle="--", linewidth=1, label="Skillful threshold (0.50)")
    axes[1, 1].set_title(
        f"(d) Fractions Skill Score at ≥ {float(metrics.get('fss_thr', 5)):g} mm/h "
        "(higher = better)",
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Lead time (min)")
    axes[1, 1].set_ylabel("FSS")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        _title(result, "2-hour STEPS verification"),
        fontsize=12.5,
        fontweight="bold",
    )
    return fig


def figure_initial(result: NowcastResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    plot = np.where(result.init_rain >= 0.1, result.init_rain, np.nan)
    im = ax.imshow(
        plot,
        origin="lower",
        extent=_extent(result),
        cmap=MAP_CMAP,
        vmin=0,
        vmax=40,
        zorder=2,
    )
    add_district_overlay(ax, result)
    fig.colorbar(im, ax=ax, label="Rain rate (mm/h)", shrink=0.85)
    fig.suptitle(_title(result, "Observed initialization"), fontsize=12, fontweight="bold")
    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 220) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()
