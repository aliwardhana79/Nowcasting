"""Save, list and reload nowcast runs for the dashboard history."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .config import NowcastConfig, RunInfo, RUNS_DIR
from .models import NowcastResult, districts_from_json, districts_to_json, jsonable


def _safe_run_dir(base_dir: Path, run_id: str) -> Path:
    path = base_dir / run_id
    if path.exists():
        counter = 2
        while (base_dir / f"{run_id}_{counter}").exists():
            counter += 1
        path = base_dir / f"{run_id}_{counter}"
    return path


def _dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total / 1024**2


def metrics_csv_rows(result: NowcastResult) -> tuple[list[str], list[list]]:
    """Return header and rows for the verification CSV."""
    metrics = result.metrics or {}
    thresholds = [float(t) for t in metrics.get("thresholds", result.config.thresholds)]
    scales = [int(s) for s in metrics.get("fss_scales_km", result.config.fss_scales_km)]
    header = [
        "step",
        "lead_time_min",
        "valid_timestamp_utc",
        "product_altitude_m",
        "crps_mmh",
        "mae_mmh",
        "bias_mmh",
        "corr",
        "ensemble_spread_mmh",
    ]
    header += [f"brier_{thr:g}mmh" for thr in thresholds]
    header += [f"roc_auc_{thr:g}mmh" for thr in thresholds]
    header += [f"fss_{scale:g}km_{metrics.get('fss_thr', result.config.fss_thr_mmh):g}mmh" for scale in scales]
    header += ["obs_fraction", "fcst_fraction"]

    rows: list[list] = []
    if not metrics:
        return header, rows
    n = len(metrics.get("crps", []))
    for step in range(n):
        ts = result.obs_timestamps[step] if step < len(result.obs_timestamps) else ""
        row = [
            step + 1,
            int(round(float(result.lead_times_min[step]))),
            ts,
            int(round(float(result.config.target_alt_m))),
            round(float(metrics["crps"][step]), 5),
            round(float(metrics["mae"][step]), 5),
            round(float(metrics["bias"][step]), 5),
            round(float(metrics["corr"][step]), 5),
            round(float(metrics["spread"][step]), 5),
        ]
        for thr in thresholds:
            row.append(round(float(metrics["brier"][f"{thr:g}"][step]), 5))
        for thr in thresholds:
            row.append(round(float(metrics["roc_auc"][f"{thr:g}"][step]), 5))
        for scale in scales:
            row.append(round(float(metrics["fss"][f"{scale:g}"][step]), 5))
        row.extend(
            [
                round(float(metrics["obs_fraction"][step]), 5),
                round(float(metrics["fcst_fraction"][step]), 5),
            ]
        )
        rows.append(row)
    return header, rows


def save_run(result: NowcastResult, base_dir: Path | str = RUNS_DIR) -> Path:
    """Persist arrays, metadata and CSV to a run directory."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _safe_run_dir(base_dir, result.run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    ensemble = np.asarray(result.ensemble, dtype=np.float16)
    obs = (
        np.asarray(result.obs, dtype=np.float16)
        if result.obs is not None
        else np.zeros((0,), dtype=np.float16)
    )
    np.savez_compressed(
        run_dir / "arrays.npz",
        ensemble=ensemble,
        obs=obs,
        rain_stack=np.asarray(result.rain_stack, dtype=np.float32),
        velocity=np.asarray(result.velocity, dtype=np.float32),
    )

    (run_dir / "districts.json").write_text(
        json.dumps(districts_to_json(result.districts), separators=(",", ":")),
        encoding="utf-8",
    )

    header, rows = metrics_csv_rows(result)
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    meta = {
        "run_id": result.run_id,
        "created_at": result.created_at,
        "mode": result.mode,
        "init_label": result.init_label,
        "init_timestamp": result.init_timestamp,
        "init_dt_utc": result.init_dt_utc,
        "timestamps": result.timestamps,
        "obs_timestamps": result.obs_timestamps,
        "timestep_min": result.timestep_min,
        "lead_times_min": result.lead_times_min.tolist(),
        "radar_lat": result.radar_lat,
        "radar_lon": result.radar_lon,
        "extent_km": result.extent_km,
        "res_km": result.res_km,
        "surabaya_xy_km": list(result.surabaya_xy_km),
        "config": result.config.to_dict(),
        "metrics": jsonable(result.metrics),
        "meta": jsonable(result.meta),
        "district_source": result.districts.get("source", ""),
        "storage": {
            "ensemble_dtype": "float16",
            "verified": result.verified,
        },
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=True), encoding="utf-8"
    )
    result.run_dir = str(run_dir)
    return run_dir


def load_run(run_dir: Path | str) -> NowcastResult:
    """Load a previously saved run into memory."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = np.load(run_dir / "arrays.npz", allow_pickle=False)
    ensemble = arrays["ensemble"]
    obs_array = arrays["obs"]
    obs = None if obs_array.size == 0 else obs_array
    districts = districts_from_json(
        json.loads((run_dir / "districts.json").read_text(encoding="utf-8"))
        if (run_dir / "districts.json").exists()
        else None
    )
    config = NowcastConfig.from_dict(meta.get("config", {}))
    result = NowcastResult(
        run_id=meta.get("run_id", run_dir.name),
        created_at=meta.get("created_at", ""),
        mode=meta.get("mode", "radar"),
        config=config,
        init_timestamp=meta.get("init_timestamp", ""),
        init_dt_utc=meta.get("init_dt_utc", ""),
        init_label=meta.get("init_label", ""),
        timestamps=list(meta.get("timestamps", [])),
        obs_timestamps=list(meta.get("obs_timestamps", [])),
        timestep_min=float(meta.get("timestep_min", 6.0)),
        lead_times_min=np.asarray(meta.get("lead_times_min", []), dtype=float),
        rain_stack=np.asarray(arrays["rain_stack"], dtype=np.float32),
        ensemble=ensemble,
        obs=obs,
        velocity=np.asarray(arrays["velocity"], dtype=np.float32),
        metrics=meta.get("metrics", {}),
        districts=districts,
        radar_lat=float(meta.get("radar_lat", -7.4103)),
        radar_lon=float(meta.get("radar_lon", 112.7604)),
        extent_km=float(meta.get("extent_km", 150.0)),
        res_km=float(meta.get("res_km", 1.0)),
        meta=meta.get("meta", {}),
        surabaya_xy_km=tuple(meta.get("surabaya_xy_km", (0.0, 0.0))),
        run_dir=str(run_dir),
    )
    return result


def list_runs(base_dir: Path | str = RUNS_DIR) -> list[RunInfo]:
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []
    items: list[RunInfo] = []
    for path in base_dir.iterdir():
        if not path.is_dir() or not (path / "meta.json").exists():
            continue
        try:
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            config = meta.get("config", {})
            items.append(
                RunInfo(
                    run_id=meta.get("run_id", path.name),
                    path=path,
                    created_at=meta.get("created_at", ""),
                    mode=meta.get("mode", "radar"),
                    init_timestamp=meta.get("init_label", ""),
                    members=int(meta.get("config", {}).get("n_members", 0)),
                    steps=len(meta.get("lead_times_min", [])),
                    altitude_m=float(config.get("target_alt_m", 1500.0)),
                    verified=bool(meta.get("metrics")),
                    size_mb=_dir_size_mb(path),
                    label=config.get("label", ""),
                )
            )
        except Exception:  # noqa: BLE001
            continue
    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


def delete_run(run_dir: Path | str) -> None:
    path = Path(run_dir)
    if path.is_dir() and path.parent == RUNS_DIR:
        shutil.rmtree(path)
