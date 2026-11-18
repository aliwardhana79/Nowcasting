"""End-to-end nowcast run orchestration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np

from . import demo, geodata, nowcasting, radar
from .config import NowcastConfig
from .models import NowcastResult

ProgressFn = Callable[[float, str], None]


def _emit(progress: ProgressFn | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(np.clip(fraction, 0.0, 1.0)), message)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _init_label(dt: datetime) -> str:
    return dt.strftime("%d %b %Y · %H:%M UTC")


def _result_id(mode: str, init_dt: datetime) -> str:
    return f"{mode}_{init_dt:%Y%m%d_%H%M}_{uuid4().hex[:6]}"


def _load_districts(
    config: NowcastConfig,
    radar_lat: float,
    radar_lon: float,
    allow_download: bool,
) -> dict:
    return geodata.load_district_data(
        radar_lat,
        radar_lon,
        max_range_km=float(config.grid_extent_km),
        local_geojson=config.geojson_path,
        allow_download=allow_download,
    )


def _assemble_result(
    *,
    config: NowcastConfig,
    mode: str,
    rain_stack: np.ndarray,
    ensemble: np.ndarray,
    obs: np.ndarray | None,
    velocity: np.ndarray,
    timestamps: list[str],
    obs_timestamps: list[str],
    init_dt: datetime,
    timestep_min: float,
    radar_lat: float,
    radar_lon: float,
    meta: dict,
    progress: ProgressFn | None = None,
    allow_district_download: bool = True,
) -> NowcastResult:
    n_steps = int(ensemble.shape[1])
    lead_times = np.arange(1, n_steps + 1, dtype=float) * float(timestep_min)
    extent = float(config.grid_extent_km)
    res = float(config.res_km)
    n = int(round(2.0 * extent / res)) + 1
    coords = np.linspace(-extent, extent, n)
    gx, gy = np.meshgrid(coords, coords)
    valid_mask = np.hypot(gx, gy) <= extent + 1e-6

    metrics: dict = {}
    if obs is not None and config.verify:
        _emit(progress, 0.82, "Computing CRPS, Brier, ROC AUC and FSS")
        metrics = nowcasting.verification_metrics(
            ensemble,
            obs,
            valid_mask,
            config,
            timestep_min,
            progress=progress,
            progress_start=0.82,
            progress_span=0.10,
        )

    _emit(progress, 0.93, "Loading East Java district boundaries")
    districts = _load_districts(
        config, radar_lat, radar_lon, allow_download=allow_district_download
    )
    surabaya_xy = geodata.latlon_to_xy(-7.2575, 112.7521, radar_lat, radar_lon)

    meta = dict(meta)
    meta.update(
        {
            "product": "BMKG Surabaya Baron GEN3-350CM · pseudo-CAPPI · STEPS",
            "grid_shape": [n, n],
            "valid_pixels": int(valid_mask.sum()),
            "software": {
                "pysteps": _package_version("pysteps"),
                "wradlib": _package_version("wradlib"),
                "numpy": _package_version("numpy"),
            },
            "verification": metrics.get("metric_library", "not requested"),
            "district_source": districts.get("source", ""),
        }
    )

    result = NowcastResult(
        run_id=_result_id(mode, init_dt),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        mode=mode,
        config=config,
        init_timestamp=timestamps[-1] if timestamps else init_dt.strftime("%Y%m%d%H%M%S"),
        init_dt_utc=init_dt.isoformat(timespec="minutes"),
        init_label=_init_label(init_dt),
        timestamps=list(timestamps),
        obs_timestamps=list(obs_timestamps),
        timestep_min=float(timestep_min),
        lead_times_min=lead_times,
        rain_stack=np.asarray(rain_stack, dtype=np.float32),
        ensemble=np.asarray(ensemble, dtype=np.float32),
        obs=None if obs is None else np.asarray(obs, dtype=np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        metrics=metrics,
        districts=districts,
        radar_lat=float(radar_lat),
        radar_lon=float(radar_lon),
        extent_km=extent,
        res_km=res,
        meta=meta,
        surabaya_xy_km=(float(surabaya_xy[0]), float(surabaya_xy[1])),
    )
    _emit(progress, 1.0, "Run complete")
    return result


# ------------------------------------------------------------------ radar run
def run_radar_nowcast(
    config: NowcastConfig, progress: ProgressFn | None = None
) -> NowcastResult:
    data_dir = Path(config.data_dir).expanduser()
    _emit(progress, 0.005, "Inspecting Radar archive")
    inventory = radar.inventory_archive(data_dir)
    timestamps = inventory["timestamps"]
    timestep_min = radar.infer_timestep_min(timestamps)
    n_steps = config.n_steps(timestep_min)

    # Activity profile is cached, so repeat runs start almost instantly.
    _emit(progress, 0.015, "Building storm-activity profile")
    activity = radar.activity_profile(data_dir, timestamps, progress=progress)
    init_index, window_indices = radar.choose_init_index(activity, config, timestep_min)

    init_raw = timestamps[init_index]
    init_dt = radar.parse_timestamp(init_raw)
    init_window = [timestamps[i] for i in window_indices]
    _emit(
        progress,
        0.12,
        f"Storm initialization {_init_label(init_dt)} · "
        f"{n_steps} steps at {timestep_min:.0f} min",
    )

    rain_stack = []
    cappi_info = []
    for j, ts in enumerate(init_window):
        paths = radar.volume_paths(data_dir, ts)
        rain, info = radar.build_cappi(
            paths["dBZ"], paths["RhoHV"], paths["ZDR"], paths["PhiDP"], config
        )
        rain_stack.append(rain)
        cappi_info.append({"timestamp": ts, **info})
        _emit(
            progress,
            0.13 + 0.16 * (j + 1) / len(init_window),
            f"1.5 km pseudo-CAPPI {j + 1}/{len(init_window)} · "
            f"{radar.time_of_day(ts)} UTC · {info['n_sweeps']} sweeps",
        )
    rain_stack = np.stack(rain_stack)

    metadata = radar.cappi_metadata(config)
    velocity, rain_db, meta_db = nowcasting.estimate_velocity(
        rain_stack, metadata, progress
    )
    ensemble, n_steps = nowcasting.run_steps_ensemble(
        rain_db, velocity, meta_db, config, timestep_min, progress
    )

    obs = None
    obs_timestamps: list[str] = []
    if config.verify:
        obs_frames = []
        scan_dts = [radar.parse_timestamp(ts) for ts in timestamps]
        for step in range(1, n_steps + 1):
            target_dt = init_dt + timedelta(minutes=step * timestep_min)
            future = range(init_index + 1, len(timestamps))
            closest = min(future, key=lambda i: abs((scan_dts[i] - target_dt).total_seconds()))
            ts_match = timestamps[closest]
            obs_timestamps.append(ts_match)
            paths = radar.volume_paths(data_dir, ts_match)
            obs_field, _ = radar.build_cappi(
                paths["dBZ"], paths["RhoHV"], paths["ZDR"], paths["PhiDP"], config
            )
            obs_frames.append(obs_field)
            _emit(
                progress,
                0.55 + 0.22 * step / n_steps,
                f"Verification CAPPI {step}/{n_steps} · "
                f"{radar.time_of_day(ts_match)} UTC",
            )
        obs = np.stack(obs_frames)

    radar_lat, radar_lon = radar.get_radar_latlon(
        radar.volume_paths(data_dir, init_raw)["dBZ"]
    )
    meta = {
        "inventory": {
            "data_dir": inventory["data_dir"],
            "n_scans": inventory["n_scans"],
            "n_incomplete": inventory["n_incomplete"],
            "first_label": inventory["first_label"],
            "last_label": inventory["last_label"],
            "size_gb": inventory["size_gb"],
        },
        "activity": activity,
        "init_index": init_index,
        "init_window_indices": window_indices,
        "cappi": cappi_info,
        "n_steps": n_steps,
        "timestep_min": timestep_min,
        "activity_peak_pixels": int(activity[init_index]["convective_pixels"]),
        "radar": {
            "name": "BMKG Surabaya Baron GEN3-350CM",
            "lat": radar_lat,
            "lon": radar_lon,
            "band": "C-band dual-pol",
        },
        "warnings": [],
    }
    return _assemble_result(
        config=config,
        mode="radar",
        rain_stack=rain_stack,
        ensemble=ensemble,
        obs=obs,
        velocity=velocity,
        timestamps=init_window,
        obs_timestamps=obs_timestamps,
        init_dt=init_dt,
        timestep_min=timestep_min,
        radar_lat=radar_lat,
        radar_lon=radar_lon,
        meta=meta,
        progress=progress,
        allow_district_download=True,
    )


# ------------------------------------------------------------------- demo run
def run_demo_nowcast(
    config: NowcastConfig, progress: ProgressFn | None = None
) -> NowcastResult:
    config = NowcastConfig.from_dict(
        {**config.to_dict(), "mode": "demo"}
    )
    demo_data = demo.generate_demo_arrays(config, progress=progress)
    _emit(progress, 0.74, "Scoring the synthetic ensemble")
    return _assemble_result(
        config=config,
        mode="demo",
        rain_stack=demo_data["rain_stack"],
        ensemble=demo_data["ensemble"],
        obs=demo_data["obs"],
        velocity=demo_data["velocity"],
        timestamps=demo_data["timestamps"],
        obs_timestamps=demo_data["obs_timestamps"],
        init_dt=demo_data["init_dt"],
        timestep_min=demo_data["timestep_min"],
        radar_lat=demo.DEMO_RADAR_LAT,
        radar_lon=demo.DEMO_RADAR_LON,
        meta=demo_data["meta"],
        progress=progress,
        allow_district_download=False,
    )


# ------------------------------------------------------------------- workload
def workload_estimate(config: NowcastConfig, n_steps: int) -> dict:
    """A deliberately rough compute estimate for the UI."""
    cells = config.grid_shape[0] ** 2
    n_cappi = (3 + n_steps) if config.verify else 3
    cappi_seconds = n_cappi * 1.7 * (cells / (301**2))
    steps_seconds = n_steps * config.n_members * 0.10 * (cells / (301**2))
    total = cappi_seconds + steps_seconds + 12.0
    if total < 90:
        level = "short"
        text = f"~{max(1, round(total / 60))} min"
    elif total < 480:
        level = "medium"
        text = f"~{max(2, round(total / 60))} min"
    else:
        level = "long"
        text = f"~{max(8, round(total / 60))} min"
    return {
        "seconds": total,
        "level": level,
        "text": text,
        "n_cappi": n_cappi,
        "n_steps": n_steps,
    }
