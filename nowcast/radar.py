"""Rainbow5 volume ingest, dual-pol QC and pseudo-CAPPI construction."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from .config import CACHE_DIR, NowcastConfig

try:  # pragma: no cover - import guard for demo-only installs
    import wradlib as wrl
except Exception:  # noqa: BLE001
    wrl = None

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("wradlib").setLevel(logging.ERROR)

ProgressFn = Callable[[float, str], None]

_KD_INDEX_CACHE: dict[tuple, np.ndarray] = {}
_CACHE_LIMIT = 160
_CAPPI_CACHE: dict[tuple, tuple[np.ndarray, dict]] = {}
_CAPPI_CACHE_LIMIT = 80

VOL_VARIABLES = ("dBZ", "RhoHV", "ZDR", "PhiDP")


class RadarError(RuntimeError):
    """Raised when the radar archive cannot satisfy a request."""


def require_wradlib() -> None:
    if wrl is None:
        raise RadarError(
            "wradlib is required to read Rainbow5 volumes. "
            "Install it with `pip install wradlib`."
        )


# --------------------------------------------------------------------- helpers
def parse_timestamp(raw: str) -> datetime:
    return datetime.strptime(str(raw)[:14], "%Y%m%d%H%M%S")


def timestamp_label(raw: str) -> str:
    try:
        return parse_timestamp(raw).strftime("%d %b %Y · %H:%M UTC")
    except Exception:  # noqa: BLE001
        return str(raw)


def time_of_day(raw: str) -> str:
    try:
        return parse_timestamp(raw).strftime("%H:%M")
    except Exception:  # noqa: BLE001
        return str(raw)


def _emit(progress: ProgressFn | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(np.clip(fraction, 0.0, 1.0)), message)


def _read_rainbow(path: Path, loaddata: bool = True):
    require_wradlib()
    return wrl.io.read_rainbow(str(path), loaddata=loaddata)


def _slice_list(rb) -> list:
    slices = rb["volume"]["scan"]["slice"]
    if isinstance(slices, dict):
        return [slices]
    return list(slices)


def resolve_phidp_path(data_dir: Path, timestamp: str) -> Path:
    primary = data_dir / f"{timestamp}PhiDP.vol"
    if primary.exists():
        return primary
    return data_dir / f"{timestamp}uPhiDP.vol"


def volume_paths(data_dir: Path, timestamp: str) -> dict[str, Path]:
    return {
        "dBZ": data_dir / f"{timestamp}dBZ.vol",
        "RhoHV": data_dir / f"{timestamp}RhoHV.vol",
        "ZDR": data_dir / f"{timestamp}ZDR.vol",
        "PhiDP": resolve_phidp_path(data_dir, timestamp),
    }


def infer_timestep_min(timestamps: Iterable[str]) -> float:
    stamps = list(timestamps)
    if len(stamps) < 2:
        return 6.0
    diffs = []
    for a, b in zip(stamps[:-1], stamps[1:]):
        try:
            delta = (parse_timestamp(b) - parse_timestamp(a)).total_seconds() / 60.0
        except Exception:  # noqa: BLE001
            continue
        if 0 < delta < 30:
            diffs.append(delta)
    if not diffs:
        return 6.0
    return float(np.median(diffs[:30]))


# ------------------------------------------------------------------- inventory
def inventory_archive(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir).expanduser()
    if not data_dir.is_dir():
        raise RadarError(f"Radar directory not found: {data_dir}")

    dbz_files = sorted(data_dir.glob("*dBZ.vol"))
    if not dbz_files:
        raise RadarError(f"No *dBZ.vol Rainbow5 files in {data_dir}")

    complete: list[str] = []
    incomplete: list[dict] = []
    for path in dbz_files:
        ts = path.name[:16]
        missing = []
        for var in ("RhoHV", "ZDR"):
            if not (data_dir / f"{ts}{var}.vol").exists():
                missing.append(var)
        if not resolve_phidp_path(data_dir, ts).exists():
            missing.append("PhiDP")
        if missing:
            incomplete.append({"timestamp": ts, "missing": missing})
        else:
            complete.append(ts)

    if not complete:
        raise RadarError(
            "No complete dual-pol volumes found. Each scan needs "
            "dBZ, RhoHV, ZDR and PhiDP (or uPhiDP)."
        )

    return {
        "data_dir": str(data_dir),
        "timestamps": complete,
        "n_scans": len(complete),
        "n_incomplete": len(incomplete),
        "incomplete": incomplete[:50],
        "first": complete[0],
        "last": complete[-1],
        "first_label": timestamp_label(complete[0]),
        "last_label": timestamp_label(complete[-1]),
        "timestep_min": infer_timestep_min(complete),
        "size_gb": sum(p.stat().st_size for p in data_dir.glob("*.*vol")) / 1e9,
    }


# ------------------------------------------------------------- sweep decoding
def _decode_raw(rawdata: dict) -> np.ndarray:
    raw = np.asarray(rawdata["data"]).astype(np.float32)
    raw_min = float(rawdata["@min"])
    raw_max = float(rawdata["@max"])
    depth = int(rawdata["@depth"])
    scale = (raw_max - raw_min) / (2**depth - 2)
    data = np.where(raw == 0, np.nan, raw_min + (raw - 1.0) * scale)
    return data.astype(np.float32)


def _ray_azimuths(rayinfo) -> np.ndarray:
    """Decode Rainbow5 ray start angles to degrees.

    Some Rainbow5 writers store azimuths as 16-bit fractions of 360 degrees
    (e.g. 65354 -> 359.0). Older exports store plain degrees. The scaling is
    detected from the value range so both conventions work.
    """
    if isinstance(rayinfo, list):
        entry = next(
            (ri for ri in rayinfo if str(ri.get("@refid", "")).lower() == "startangle"),
            rayinfo[0],
        )
    else:
        entry = rayinfo
    az = np.asarray(entry["data"], dtype=np.float32).ravel()
    depth = int(entry.get("@depth", 16) or 16)
    if az.size and float(np.nanmax(az)) > 360.5:
        az = az * (360.0 / float(2**depth))
    if az.size and float(np.nanmax(az)) >= 360.0:
        az = np.mod(az, 360.0)
    return az.astype(np.float32)


def _first_float(*values, default: float = 0.0) -> float:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _parse_sweep(sl: dict, radar_alt_m: float) -> tuple[np.ndarray, float, dict]:
    """Decode one sweep into physical units and projected polar coordinates."""
    rawdata = sl["slicedata"]["rawdata"]
    data = _decode_raw(rawdata)
    nrays, nbins = data.shape

    az = _ray_azimuths(sl["slicedata"]["rayinfo"])
    if az.size != nrays:
        if az.size < nrays:
            az = np.pad(az, (0, nrays - az.size), mode="edge")
        else:
            az = az[:nrays]

    el_deg = _first_float(sl.get("posangle"), sl.get("pargroup", {}).get("posangle"), default=0.5)
    r_step_km = _first_float(
        sl.get("rangestep"),
        sl.get("pargroup", {}).get("rangestep"),
        default=0.0,
    )
    r_start_km = _first_float(
        sl.get("start_range"),
        sl.get("startrange"),
        sl.get("pargroup", {}).get("start_range"),
        default=0.0,
    )
    if r_step_km <= 0:
        stop = _first_float(sl.get("stoprange"), sl.get("pargroup", {}).get("stoprange"))
        r_step_km = (stop - r_start_km) / max(nbins, 1) if stop > r_start_km else 0.25

    r_meters = (r_start_km + (np.arange(nbins, dtype=np.float32) + 0.5) * r_step_km) * 1000.0
    el_rad = np.deg2rad(el_deg)
    az_rad = np.deg2rad(az)[:, None]
    r_2d = r_meters[None, :]

    r_eff = (4.0 / 3.0) * 6371000.0
    beam_alt_m = (
        radar_alt_m
        + np.sqrt(np.maximum(0.0, r_2d**2 + r_eff**2 + 2.0 * r_2d * r_eff * np.sin(el_rad)))
        - r_eff
    )
    ground_range = r_eff * np.arcsin(
        np.clip(
            (r_2d * np.cos(el_rad)) / (r_eff + np.maximum(beam_alt_m - radar_alt_m, 0.0)),
            -1.0,
            1.0,
        )
    )
    x_polar = ground_range * np.sin(az_rad)
    y_polar = ground_range * np.cos(az_rad)

    az_hash = hashlib.blake2b(az.tobytes(), digest_size=8).hexdigest()
    geom_key = (
        nrays,
        nbins,
        round(el_deg, 3),
        round(r_step_km, 5),
        round(r_start_km, 5),
        az_hash,
    )
    info = {
        "elevation_deg": el_deg,
        "range_step_km": r_step_km,
        "start_range_km": r_start_km,
        "geometry_key": geom_key,
    }
    return data, el_deg, {
        "x": x_polar,
        "y": y_polar,
        "alt": np.broadcast_to(beam_alt_m, data.shape),
        "info": info,
    }


def read_sweep_with_height(
    vol_path: str | Path, sweep_idx: int, radar_alt_m: float = 0.0
):
    """Public helper retained for compatibility with the original script."""
    rb = _read_rainbow(Path(vol_path), loaddata=True)
    slices = _slice_list(rb)
    data, el_deg, geom = _parse_sweep(slices[sweep_idx], radar_alt_m)
    return (
        data,
        el_deg,
        geom["x"],
        geom["y"],
        geom["alt"],
        geom["info"]["geometry_key"],
    )


def get_radar_latlon(vol_path: str | Path) -> tuple[float, float]:
    rb = _read_rainbow(Path(vol_path), loaddata=False)
    sensor = rb.get("volume", {}).get("sensorinfo", {})
    lat = _first_float(sensor.get("lat"), sensor.get("@lat"), default=-7.4103)
    lon = _first_float(sensor.get("lon"), sensor.get("@lon"), default=112.7604)
    return lat, lon


# ------------------------------------------------------------------- activity
def _dir_fingerprint(data_dir: Path, timestamps: list[str]) -> str:
    total = 0
    latest = 0.0
    for ts in timestamps:
        path = data_dir / f"{ts}dBZ.vol"
        try:
            stat = path.stat()
        except OSError:
            continue
        total += stat.st_size
        latest = max(latest, stat.st_mtime)
    token = f"{data_dir.resolve()}|{len(timestamps)}|{total}|{latest:.3f}"
    return hashlib.sha1(token.encode()).hexdigest()


def activity_profile(
    data_dir: str | Path,
    timestamps: list[str],
    progress: ProgressFn | None = None,
    force: bool = False,
) -> list[dict]:
    """Convective-pixel count per scan (first sweep), cached on disk."""
    data_dir = Path(data_dir).expanduser()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / (
        "activity_" + hashlib.sha1(str(data_dir.resolve()).encode()).hexdigest()[:12] + ".json"
    )
    fingerprint = _dir_fingerprint(data_dir, timestamps)
    if cache_file.exists() and not force:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                return cached["activity"]
        except Exception:  # noqa: BLE001
            pass

    rows: list[dict] = []
    n = max(len(timestamps), 1)
    for i, ts in enumerate(timestamps):
        try:
            rb = _read_rainbow(data_dir / f"{ts}dBZ.vol", loaddata=True)
            sl = _slice_list(rb)[0]
            dbz = _decode_raw(sl["slicedata"]["rawdata"])
            finite = np.isfinite(dbz)
            rows.append(
                {
                    "timestamp": ts,
                    "label": timestamp_label(ts),
                    "time": time_of_day(ts),
                    "convective_pixels": int(np.sum(dbz >= 25.0)),
                    "strong_pixels": int(np.sum(dbz >= 35.0)),
                    "max_dbz": float(np.nanmax(dbz)) if finite.any() else float("nan"),
                    "valid_fraction": float(finite.mean()),
                }
            )
        except Exception:  # noqa: BLE001
            rows.append(
                {
                    "timestamp": ts,
                    "label": timestamp_label(ts),
                    "time": time_of_day(ts),
                    "convective_pixels": 0,
                    "strong_pixels": 0,
                    "max_dbz": float("nan"),
                    "valid_fraction": 0.0,
                }
            )
        if i % 5 == 0 or i == len(timestamps) - 1:
            _emit(
                progress,
                0.02 + 0.08 * (i + 1) / n,
                f"Scanning convective activity · {i + 1}/{len(timestamps)} volumes",
            )

    payload = {"fingerprint": fingerprint, "activity": rows}
    try:
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass
    return rows


def choose_init_index(
    activity: list[dict],
    config: NowcastConfig,
    timestep_min: float,
) -> tuple[int, list[int]]:
    """Pick the initialization scan and the three-scan optical-flow window."""
    n = len(activity)
    n_steps = config.n_steps(timestep_min)
    lookback = max(1, int(config.lookback_scans))
    min_idx = lookback - 1
    max_idx = n - n_steps - 1
    if max_idx < min_idx:
        raise RadarError(
            f"Archive has {n} scans, but a {config.horizon_min:.0f}-minute forecast "
            f"at {timestep_min:.0f}-min intervals needs {n_steps + lookback} scans."
        )

    if config.init_mode == "manual" and config.init_timestamp:
        wanted = str(config.init_timestamp)
        matches = [
            i
            for i, row in enumerate(activity)
            if row["timestamp"] == wanted or row["timestamp"].startswith(wanted[:14])
        ]
        if not matches:
            raise RadarError(f"Initialization scan {wanted} is not in the complete archive.")
        idx = matches[0]
        if idx < min_idx or idx > max_idx:
            raise RadarError(
                "The selected initialization scan does not have enough scans before "
                "or after it for optical flow and verification."
            )
    else:
        candidates = list(range(min_idx, max_idx + 1, 4))
        if not candidates:
            candidates = [min_idx]
        idx = max(candidates, key=lambda i: activity[i]["convective_pixels"])
        # Refine to the local maximum so the peak storm scan is not missed
        lo, hi = max(min_idx, idx - 3), min(max_idx, idx + 3)
        idx = max(range(lo, hi + 1), key=lambda i: activity[i]["convective_pixels"])

    window = list(range(idx - lookback + 1, idx + 1))
    return idx, window


# --------------------------------------------------------------------- CAPPI
def _nearest_indices(
    x_polar: np.ndarray,
    y_polar: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    config: NowcastConfig,
    geom_key: tuple,
) -> np.ndarray:
    key = (geom_key, round(config.grid_extent_km, 3), round(config.res_km, 4))
    cached = _KD_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    tree = cKDTree(np.column_stack([x_polar.ravel(), y_polar.ravel()]))
    _, idx = tree.query(points, k=1)
    if len(_KD_INDEX_CACHE) > _CACHE_LIMIT:
        _KD_INDEX_CACHE.clear()
    _KD_INDEX_CACHE[key] = idx
    return idx


def _range_mask(grid_x: np.ndarray, grid_y: np.ndarray, extent_km: float) -> np.ndarray:
    return np.hypot(grid_x, grid_y) > extent_km * 1000.0


def build_cappi(
    dbz_path: str | Path,
    rhohv_path: str | Path,
    zdr_path: str | Path,
    phidp_path: str | Path,
    config: NowcastConfig,
) -> tuple[np.ndarray, dict]:
    """Build a constant-altitude pseudo-CAPPI rain-rate field (mm/h)."""
    require_wradlib()
    started = time.time()
    paths = {
        "dBZ": Path(dbz_path),
        "RhoHV": Path(rhohv_path),
        "ZDR": Path(zdr_path),
        "PhiDP": Path(phidp_path),
    }
    for name, path in paths.items():
        if not path.exists():
            raise RadarError(f"Missing {name} volume: {path}")

    cache_key = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths.values()
    ) + (
        round(float(config.grid_extent_km), 4),
        round(float(config.res_km), 4),
        round(float(config.target_alt_m), 3),
        round(float(config.max_elev_deg), 3),
        round(float(config.zr_a), 5),
        round(float(config.zr_b), 5),
    )
    cached = _CAPPI_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], dict(cached[1])

    volumes = {name: _read_rainbow(path, loaddata=True) for name, path in paths.items()}
    slices = {name: _slice_list(rb) for name, rb in volumes.items()}
    sensor = volumes["dBZ"].get("volume", {}).get("sensorinfo", {})
    radar_alt_m = _first_float(sensor.get("alt"), sensor.get("@alt"), default=0.0)
    n_slices = min(len(slices[name]) for name in slices)

    extent_km = float(config.grid_extent_km)
    res_km = float(config.res_km)
    coords = np.arange(-extent_km * 1000.0, (extent_km + res_km) * 1000.0, res_km * 1000.0)
    grid_x, grid_y = np.meshgrid(coords, coords)
    range_mask = _range_mask(grid_x, grid_y, extent_km)

    sweep_records: list[tuple[float, np.ndarray, np.ndarray]] = []
    elevations: list[float] = []
    met_fraction: list[float] = []

    for s_idx in range(n_slices):
        dbz, el_deg, geom = _parse_sweep(slices["dBZ"][s_idx], radar_alt_m)
        if el_deg > float(config.max_elev_deg):
            continue

        rho, _, _ = _parse_sweep(slices["RhoHV"][s_idx], radar_alt_m)
        zdr, _, _ = _parse_sweep(slices["ZDR"][s_idx], radar_alt_m)
        phidp, _, _ = _parse_sweep(slices["PhiDP"][s_idx], radar_alt_m)
        if not (dbz.shape == rho.shape == zdr.shape == phidp.shape):
            continue

        # 1. Dual-pol meteorological echo mask
        met_mask = (
            np.isfinite(dbz)
            & np.isfinite(rho)
            & np.isfinite(zdr)
            & (dbz >= 5.0)
            & (rho >= 0.85)
            & (np.abs(zdr) <= 5.0)
        )
        if met_mask.sum() == 0:
            continue
        dbz_clean = np.where(met_mask, dbz, np.nan)
        met_fraction.append(float(met_mask.mean()))

        # 2. PhiDP unwrap / smoothing and C-band path-integrated attenuation
        phidp_met = np.where(met_mask & np.isfinite(phidp), phidp, np.nan)
        phidp_filled = np.nan_to_num(phidp_met, nan=0.0)
        phidp_unwrapped = (
            np.rad2deg(np.unwrap(np.deg2rad(2.0 * phidp_filled), axis=1)) / 2.0
        )
        kernel = 7 if phidp_unwrapped.shape[1] >= 7 else 1
        phidp_smooth = median_filter(phidp_unwrapped, size=(1, kernel))

        valid_near = np.where(met_mask[:, : min(40, met_mask.shape[1])],
                              phidp_smooth[:, : min(40, phidp_smooth.shape[1])],
                              np.nan)
        sys_phase = (
            float(np.nanpercentile(valid_near, 5))
            if np.any(np.isfinite(valid_near))
            else 0.0
        )
        delta_phidp = np.maximum.accumulate(
            np.maximum(0.0, phidp_smooth - sys_phase), axis=1
        )
        pia_db = np.clip(0.08 * delta_phidp, 0.0, 10.0)
        dbz_corr = np.where(met_mask, dbz_clean + pia_db, -32.0)

        # 3. Z-R conversion
        rain_rate = ((10.0 ** (dbz_corr / 10.0)) / float(config.zr_a)) ** (
            1.0 / float(config.zr_b)
        )
        rain_rate = np.where(met_mask, rain_rate, 0.0)
        # Guard against isolated calibration/noise spikes; 150 mm/h is already
        # far beyond the range where the Z-R relation is meaningful.
        rain_rate = np.clip(rain_rate, 0.0, 150.0)
        rain_rate = np.nan_to_num(rain_rate, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Nearest-neighbour polar-to-Cartesian projection
        idx = _nearest_indices(
            geom["x"], geom["y"], grid_x, grid_y, config, geom["info"]["geometry_key"]
        )
        rain_cart = rain_rate.ravel()[idx].reshape(grid_x.shape)
        alt_cart = geom["alt"].ravel()[idx].reshape(grid_x.shape)
        sweep_records.append((float(el_deg), alt_cart, rain_cart))
        elevations.append(float(el_deg))

    if not sweep_records:
        raise RadarError(
            "No radar sweeps below the maximum elevation could be processed."
        )

    # 5. Vertical interpolation to the target altitude
    sweep_records.sort(key=lambda item: item[0])
    alt_stack = np.array([rec[1] for rec in sweep_records], dtype=np.float32)
    rain_stack = np.array([rec[2] for rec in sweep_records], dtype=np.float32)
    k_sweeps = alt_stack.shape[0]

    below = alt_stack <= float(config.target_alt_m)
    idx_low = np.clip(np.sum(below, axis=0) - 1, 0, k_sweeps - 1)
    idx_high = np.clip(idx_low + 1, 0, k_sweeps - 1)

    h_low = np.take_along_axis(alt_stack, idx_low[None, :, :], axis=0)[0]
    h_high = np.take_along_axis(alt_stack, idx_high[None, :, :], axis=0)[0]
    r_low = np.take_along_axis(rain_stack, idx_low[None, :, :], axis=0)[0]
    r_high = np.take_along_axis(rain_stack, idx_high[None, :, :], axis=0)[0]

    denom = np.where(h_high > h_low, h_high - h_low, 1.0)
    weight = np.where(
        h_high > h_low,
        np.clip((float(config.target_alt_m) - h_low) / denom, 0.0, 1.0),
        0.0,
    )
    cappi_rain = (1.0 - weight) * r_low + weight * r_high
    cappi_rain[range_mask] = np.nan
    cappi_rain = cappi_rain.astype(np.float32)

    info = {
        "n_sweeps": k_sweeps,
        "elevations_deg": elevations,
        "radar_alt_m": radar_alt_m,
        "met_fraction": float(np.mean(met_fraction)) if met_fraction else 0.0,
        "max_rain_mmh": float(np.nanmax(cappi_rain)) if np.isfinite(cappi_rain).any() else 0.0,
        "coverage_fraction": float(np.isfinite(cappi_rain).mean()),
        "elapsed_s": time.time() - started,
    }
    if len(_CAPPI_CACHE) >= _CAPPI_CACHE_LIMIT:
        _CAPPI_CACHE.clear()
    _CAPPI_CACHE[cache_key] = (cappi_rain, dict(info))
    return cappi_rain, info


def cappi_metadata(config: NowcastConfig) -> dict:
    return {
        "xpixelsize": float(config.res_km) * 1000.0,
        "ypixelsize": float(config.res_km) * 1000.0,
        "unit": "mm/h",
        "transform": None,
        "zerovalue": 0.0,
        "threshold": 0.1,
    }
