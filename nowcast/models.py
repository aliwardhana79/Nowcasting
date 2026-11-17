"""Result model shared by the pipeline, plotting and storage layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from .config import NowcastConfig


def jsonable(value: Any) -> Any:
    """Recursively convert numpy containers/scalars into JSON-safe objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def districts_to_json(district_data: dict) -> dict:
    """Convert district arrays to nested lists for disk storage."""
    out = {
        "source": district_data.get("source", ""),
        "labels": [list(item) for item in district_data.get("labels", [])],
        "other_segments": [],
        "surabaya_segments": [],
        "polygons": [],
    }
    for key in ("other_segments", "surabaya_segments"):
        for seg in district_data.get(key, []):
            arr = np.asarray(seg, dtype=float)
            if arr.size:
                out[key].append(arr.tolist())
    for poly in district_data.get("polygons", []):
        rings = []
        for ring in poly.get("rings", []):
            arr = np.asarray(ring, dtype=float)
            if arr.size:
                rings.append(arr.tolist())
        out["polygons"].append(
            {
                "name": poly.get("name", ""),
                "display_name": poly.get("display_name", poly.get("name", "")),
                "rings": rings,
            }
        )
    return out


def districts_from_json(data: dict | None) -> dict:
    """Rehydrate district geometry arrays after loading a run."""
    if not data:
        return {
            "source": "",
            "labels": [],
            "other_segments": [],
            "surabaya_segments": [],
            "polygons": [],
        }
    out = {
        "source": data.get("source", ""),
        "labels": [
            (float(item[0]), float(item[1]), str(item[2]))
            for item in data.get("labels", [])
            if len(item) >= 3
        ],
        "other_segments": [
            np.asarray(seg, dtype=float) for seg in data.get("other_segments", []) if seg
        ],
        "surabaya_segments": [
            np.asarray(seg, dtype=float)
            for seg in data.get("surabaya_segments", [])
            if seg
        ],
        "polygons": [],
    }
    for poly in data.get("polygons", []):
        rings = [np.asarray(r, dtype=float) for r in poly.get("rings", []) if r]
        if not rings:
            continue
        out["polygons"].append(
            {
                "name": poly.get("name", ""),
                "display_name": poly.get("display_name", poly.get("name", "")),
                "rings": rings,
            }
        )
    return out


@dataclass
class NowcastResult:
    """Everything a dashboard view needs from one nowcast run."""

    run_id: str
    created_at: str
    mode: str
    config: NowcastConfig
    init_timestamp: str
    init_dt_utc: str
    init_label: str
    timestamps: list[str]
    obs_timestamps: list[str]
    timestep_min: float
    lead_times_min: np.ndarray
    rain_stack: np.ndarray
    ensemble: np.ndarray
    obs: np.ndarray | None
    velocity: np.ndarray
    metrics: dict
    districts: dict
    radar_lat: float
    radar_lon: float
    extent_km: float
    res_km: float
    meta: dict = field(default_factory=dict)
    surabaya_xy_km: tuple[float, float] = (0.0, 0.0)
    run_dir: str = ""

    # ------------------------------------------------------------- properties
    @property
    def n_members(self) -> int:
        return int(self.ensemble.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.ensemble.shape[1])

    @property
    def verified(self) -> bool:
        return self.obs is not None and bool(self.metrics.get("crps"))

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def coords_km(self) -> np.ndarray:
        n = int(round(2.0 * self.extent_km / self.res_km)) + 1
        return np.linspace(-self.extent_km, self.extent_km, n)

    @cached_property
    def valid_mask(self) -> np.ndarray:
        coords = self.coords_km
        gx, gy = np.meshgrid(coords, coords)
        return np.hypot(gx, gy) <= self.extent_km + 1e-6

    @property
    def init_rain(self) -> np.ndarray:
        return self.rain_stack[-1]

    def lead_label(self, idx: int) -> str:
        idx = int(np.clip(idx, 0, self.n_steps - 1))
        return f"+{int(round(float(self.lead_times_min[idx])))} min"

    def valid_time_label(self, idx: int) -> str:
        idx = int(np.clip(idx, 0, len(self.obs_timestamps) - 1))
        raw = self.obs_timestamps[idx]
        if not raw:
            return self.lead_label(idx)
        try:
            from datetime import datetime

            dt = datetime.strptime(raw[:14], "%Y%m%d%H%M%S")
            return dt.strftime("%H:%M UTC")
        except Exception:
            return raw

    # -------------------------------------------------------------- fields
    def probability_field(self, lead_idx: int, threshold: float) -> np.ndarray:
        """Exceedance probability (%) for one lead time and threshold."""
        ens = self.ensemble[:, int(lead_idx)]
        prob = np.mean(ens >= float(threshold), axis=0, dtype=np.float32) * 100.0
        prob[~self.valid_mask] = np.nan
        return prob

    def mean_field(self, lead_idx: int) -> np.ndarray:
        field = np.mean(self.ensemble[:, int(lead_idx)], axis=0, dtype=np.float32)
        field[~self.valid_mask] = np.nan
        return field

    def spread_field(self, lead_idx: int) -> np.ndarray:
        field = np.std(self.ensemble[:, int(lead_idx)], axis=0, dtype=np.float32)
        field[~self.valid_mask] = np.nan
        return field

    def observed_field(self, lead_idx: int) -> np.ndarray | None:
        if self.obs is None:
            return None
        field = np.asarray(self.obs[int(lead_idx)], dtype=np.float32).copy()
        field[~self.valid_mask] = np.nan
        return field

    def domain_peak_probability(self, lead_idx: int, threshold: float) -> float:
        prob = self.probability_field(lead_idx, threshold)
        if not np.isfinite(prob).any():
            return float("nan")
        return float(np.nanmax(prob))

    def probability_around_surabaya(
        self, lead_idx: int, threshold: float, radius_km: float = 18.0
    ) -> float:
        """Mean probability in a disc over the Surabaya urban area."""
        prob = self.probability_field(lead_idx, threshold)
        coords = self.coords_km
        gx, gy = np.meshgrid(coords, coords)
        sx, sy = self.surabaya_xy_km
        disc = np.hypot(gx - sx, gy - sy) <= radius_km
        if not disc.any():
            return float("nan")
        return float(np.nanmean(prob[disc]))

    def motion_summary(self) -> dict[str, float]:
        """Domain-median storm motion in km/h and meteorological direction."""
        if self.velocity is None or self.velocity.size == 0:
            return {}
        u = np.asarray(self.velocity[0], dtype=float)
        v = np.asarray(self.velocity[1], dtype=float)
        mask = self.valid_mask & np.isfinite(u) & np.isfinite(v)
        if not mask.any():
            return {}
        u_med = float(np.median(u[mask]))
        v_med = float(np.median(v[mask]))
        speed = float(np.hypot(u_med, v_med) * self.res_km / self.timestep_min * 60.0)
        # Array rows increase northwards; v[1] is the northward component.
        direction = float((np.degrees(np.arctan2(u_med, v_med)) + 360.0) % 360.0)
        return {
            "u_kmh": u_med * self.res_km / self.timestep_min * 60.0,
            "v_kmh": v_med * self.res_km / self.timestep_min * 60.0,
            "speed_kmh": speed,
            "direction_deg": direction,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "mode": self.mode,
            "init": self.init_label,
            "members": self.n_members,
            "steps": self.n_steps,
            "altitude_m": self.config.target_alt_m,
            "verified": self.verified,
            "demo": self.is_demo,
        }
