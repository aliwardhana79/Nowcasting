"""Optical flow, STEPS ensemble generation and probabilistic verification."""
from __future__ import annotations

import contextlib
import io
import warnings
from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter

from .config import NowcastConfig

warnings.filterwarnings("ignore", category=UserWarning)

try:  # pragma: no cover - import guard
    from pysteps import motion as ps_motion
    from pysteps import nowcasts as ps_nowcasts
    from pysteps.utils import transformation as ps_transformation
    from pysteps.verification import probscores as ps_probscores

    HAS_PYSTEPS = True
except Exception:  # noqa: BLE001
    ps_motion = ps_nowcasts = ps_transformation = ps_probscores = None
    HAS_PYSTEPS = False

ProgressFn = Callable[[float, str], None]


class NowcastError(RuntimeError):
    """Raised when the optical flow or STEPS calculation fails."""


def _emit(progress: ProgressFn | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(np.clip(fraction, 0.0, 1.0)), message)


def _clean_precip(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    field = np.where(np.isfinite(field), field, 0.0)
    return np.maximum(field, 0.0)


# ------------------------------------------------------------------- motion
def estimate_velocity(
    rain_stack: np.ndarray,
    metadata: dict,
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate a dense advection field with Lucas-Kanade optical flow."""
    if not HAS_PYSTEPS:
        raise NowcastError(
            "pysteps is required for optical flow and STEPS. "
            "Install it with `pip install pysteps`."
        )
    if rain_stack.ndim != 3 or rain_stack.shape[0] < 2:
        raise NowcastError("At least two rain-rate frames are needed for optical flow.")

    _emit(progress, 0.32, "Transforming rain rates to dB and estimating echo motion")
    rain_db, meta_db = ps_transformation.dB_transform(
        np.asarray(rain_stack, dtype=np.float32),
        metadata,
        threshold=0.1,
        zerovalue=-15.0,
    )
    rain_db = np.nan_to_num(np.asarray(rain_db, dtype=np.float32), nan=-15.0)

    buffer = io.StringIO()
    try:
        method = ps_motion.get_method("LK")
        with contextlib.redirect_stdout(buffer):
            velocity = method(rain_db)
    except Exception as exc:  # noqa: BLE001
        tail = buffer.getvalue()[-1200:]
        raise NowcastError(f"Optical-flow estimation failed: {exc}\n{tail}") from exc

    if not np.isfinite(velocity).any():
        raise NowcastError("Optical flow returned no valid motion vectors.")
    return (
        np.asarray(velocity, dtype=np.float32),
        np.asarray(rain_db, dtype=np.float32),
        meta_db,
    )


# -------------------------------------------------------------------- STEPS
def run_steps_ensemble(
    rain_db: np.ndarray,
    velocity: np.ndarray,
    meta_db: dict,
    config: NowcastConfig,
    timestep_min: float,
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, int]:
    """Run the STEPS stochastic nowcast and return rain rates (M, T, H, W)."""
    if not HAS_PYSTEPS:
        raise NowcastError(
            "pysteps is required for STEPS nowcasting. "
            "Install it with `pip install pysteps`."
        )
    n_steps = config.n_steps(timestep_min)
    _emit(
        progress,
        0.38,
        f"Running STEPS · {n_steps} steps · {config.n_members} members · "
        f"{config.n_cascade_levels} cascade levels",
    )

    buffer = io.StringIO()
    try:
        steps_nowcast = ps_nowcasts.get_method("steps")
        with contextlib.redirect_stdout(buffer):
            ensemble_db = steps_nowcast(
                precip=np.asarray(rain_db, dtype=np.float32),
                velocity=np.asarray(velocity, dtype=np.float32),
                timesteps=n_steps,
                n_ens_members=int(config.n_members),
                n_cascade_levels=int(config.n_cascade_levels),
                precip_thr=-10.0,
                kmperpixel=float(config.res_km),
                timestep=float(timestep_min),
                noise_method="nonparametric",
                vel_pert_method="bps",
                mask_method="incremental",
                probmatching_method="cdf",
                num_workers=1,
                seed=int(config.seed),
            )
    except Exception as exc:  # noqa: BLE001
        tail = buffer.getvalue()[-2000:]
        raise NowcastError(f"STEPS ensemble generation failed: {exc}\n{tail}") from exc

    _emit(progress, 0.48, "Converting STEPS ensemble back to rain rate")
    ensemble_db = np.ma.filled(ensemble_db, fill_value=-15.0)
    try:
        with contextlib.redirect_stdout(buffer):
            ensemble_mmh, _ = ps_transformation.dB_transform(
                np.asarray(ensemble_db, dtype=np.float32),
                meta_db,
                inverse=True,
            )
    except Exception as exc:  # noqa: BLE001
        raise NowcastError(f"Could not convert the STEPS ensemble to mm/h: {exc}") from exc

    ensemble_mmh = _clean_precip(ensemble_mmh)
    ensemble_mmh[ensemble_mmh < 0.1] = 0.0
    return ensemble_mmh.astype(np.float32), n_steps


# ------------------------------------------------------------- verification
def _crps_ensemble(ens: np.ndarray, obs: np.ndarray) -> float:
    """Ensemble CRPS using the sorted-member identity (numpy fallback)."""
    ens = np.asarray(ens, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    m = ens.shape[0]
    if m == 0:
        return float("nan")
    term1 = np.mean(np.abs(ens - obs[None, :]), axis=0)
    ordered = np.sort(ens, axis=0)
    weights = (2.0 * np.arange(1, m + 1) - m - 1)[:, None]
    term2 = np.sum(weights * ordered, axis=0) / (m * m)
    return float(np.mean(term1 - term2))


def _roc_auc_rank(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based ROC AUC with tie handling."""
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(label.sum())
    n_neg = int(label.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    i = 0
    while i < score.size:
        j = i + 1
        while j < score.size and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _roc_auc(prob: np.ndarray, obs: np.ndarray, threshold: float) -> float:
    label = np.asarray(obs) >= threshold
    if label.all() or (~label).all():
        return float("nan")
    if ps_probscores is not None:
        try:
            _, _, area = ps_probscores.ROC_curve(
                np.asarray(prob, dtype=np.float64),
                np.asarray(obs, dtype=np.float64),
                float(threshold),
                n_prob_thrs=20,
                compute_area=True,
            )
            return float(area)
        except Exception:  # noqa: BLE001
            pass
    return _roc_auc_rank(prob, label)


def compute_fss_ensemble(
    ensemble: np.ndarray,
    obs: np.ndarray,
    threshold: float,
    scale_px: int,
    valid_mask: np.ndarray,
) -> float:
    """Fractions Skill Score averaged over ensemble members."""
    size = max(1, int(scale_px))
    ens = np.where(np.isfinite(ensemble), ensemble, 0.0)
    obs = np.where(np.isfinite(obs), obs, 0.0)
    f_bin = (ens >= threshold).astype(np.float32)
    o_bin = (obs >= threshold).astype(np.float32)
    f_frac = uniform_filter(f_bin, size=(1, size, size), mode="constant", cval=0.0)
    o_frac = uniform_filter(o_bin, size=size, mode="constant", cval=0.0)

    weights = valid_mask.astype(bool)
    f_valid = f_frac[:, weights]
    o_valid = o_frac[weights]
    if f_valid.size == 0:
        return float("nan")
    numerator = np.sum((f_valid - o_valid[None, :]) ** 2, axis=1)
    denominator = np.sum(f_valid**2, axis=1) + float(np.sum(o_valid**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, 1.0 - numerator / denominator, np.nan)
    if not np.isfinite(scores).any():
        return float("nan")
    return float(np.nanmean(scores))


def verification_metrics(
    ensemble: np.ndarray,
    obs: np.ndarray,
    valid_mask: np.ndarray,
    config: NowcastConfig,
    timestep_min: float,
    progress: ProgressFn | None = None,
    progress_start: float = 0.78,
    progress_span: float = 0.14,
) -> dict:
    """Compute CRPS, Brier, ROC AUC, FSS and bulk error statistics per lead time."""
    n_steps = int(ensemble.shape[1])
    timestep_min = float(timestep_min)
    thresholds = [float(t) for t in config.thresholds]
    scales = [int(s) for s in config.fss_scales_km]
    metrics: dict = {
        "lead_times": [],
        "crps": [],
        "brier": {f"{thr:g}": [] for thr in thresholds},
        "roc_auc": {f"{thr:g}": [] for thr in thresholds},
        "fss": {f"{scale:g}": [] for scale in scales},
        "mae": [],
        "bias": [],
        "corr": [],
        "spread": [],
        "obs_fraction": [],
        "fcst_fraction": [],
        "thresholds": thresholds,
        "fss_scales_km": scales,
        "fss_thr": float(config.fss_thr_mmh),
        "metric_library": "pysteps" if ps_probscores is not None else "numpy fallback",
    }

    weights = np.asarray(valid_mask, dtype=bool)
    for step in range(n_steps):
        ens_step = np.asarray(ensemble[:, step], dtype=np.float32)
        obs_step = np.asarray(obs[step], dtype=np.float32)
        obs_step = np.where(np.isfinite(obs_step), obs_step, 0.0)
        ens_flat = ens_step[:, weights]
        obs_flat = obs_step[weights]

        if ps_probscores is not None:
            try:
                crps = float(ps_probscores.CRPS(ens_flat, obs_flat))
            except Exception:  # noqa: BLE001
                crps = _crps_ensemble(ens_flat, obs_flat)
        else:
            crps = _crps_ensemble(ens_flat, obs_flat)

        mean_field = np.mean(ens_step, axis=0, dtype=np.float32)
        mean_flat = mean_field[weights]
        metrics["lead_times"].append((step + 1) * timestep_min)
        metrics["crps"].append(crps)

        for thr in thresholds:
            key = f"{thr:g}"
            prob_flat = np.mean(ens_flat >= thr, axis=0, dtype=np.float32)
            obs_bin = (obs_flat >= thr).astype(np.float32)
            metrics["brier"][key].append(float(np.mean((prob_flat - obs_bin) ** 2)))
            metrics["roc_auc"][key].append(_roc_auc(prob_flat, obs_flat, thr))

        for scale in scales:
            scale_px = max(1, int(round(float(scale) / float(config.res_km))))
            metrics["fss"][f"{scale:g}"].append(
                compute_fss_ensemble(
                    ens_step, obs_step, config.fss_thr_mmh, scale_px, weights
                )
            )

        diff = mean_flat - obs_flat
        metrics["mae"].append(float(np.mean(np.abs(diff))))
        metrics["bias"].append(float(np.mean(diff)))
        metrics["spread"].append(float(np.mean(np.std(ens_flat, axis=0))))
        if np.std(mean_flat) > 0 and np.std(obs_flat) > 0:
            metrics["corr"].append(float(np.corrcoef(mean_flat, obs_flat)[0, 1]))
        else:
            metrics["corr"].append(float("nan"))
        metrics["obs_fraction"].append(float(np.mean(obs_flat >= config.fss_thr_mmh)))
        metrics["fcst_fraction"].append(float(np.mean(mean_flat >= config.fss_thr_mmh)))

        brier_preview = metrics["brier"].get("5", [float("nan")])
        brier_value = brier_preview[step] if step < len(brier_preview) else float("nan")
        _emit(
            progress,
            progress_start + progress_span * (step + 1) / n_steps,
            f"Verifying lead time {step + 1}/{n_steps} · "
            f"CRPS {crps:.3f} · Brier@5 {brier_value:.3f}",
        )
    return metrics
