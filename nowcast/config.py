"""Configuration objects and paths for the Surabaya radar nowcast dashboard."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
RUNS_DIR = APP_DIR / "runs"
CACHE_DIR = Path.home() / ".cache" / "surabaya_nowcast"

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "SBY_RADAR_DIR",
        "/Users/aliwardhana/Downloads/Training_TL/00_data/Radar/2025-03-06",
    )
)

DEFAULT_THRESHOLDS = (1.0, 5.0, 10.0)
DEFAULT_FSS_SCALES = (5, 15, 30)
MAP_THRESHOLDS = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0)


@dataclass
class NowcastConfig:
    """All tunable parameters for one nowcast run."""

    data_dir: str = str(DEFAULT_DATA_DIR)
    mode: str = "radar"  # "radar" | "demo"

    # Radar / CAPPI
    grid_extent_km: float = 150.0
    res_km: float = 1.0
    target_alt_m: float = 1500.0
    max_elev_deg: float = 12.0

    # Z-R relation (Z = a R^b)
    zr_a: float = 250.0
    zr_b: float = 1.2

    # STEPS ensemble
    horizon_min: float = 120.0
    n_members: int = 20
    n_cascade_levels: int = 6
    seed: int = 42

    # Verification
    verify: bool = True
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    fss_scales_km: tuple[int, ...] = DEFAULT_FSS_SCALES
    fss_thr_mmh: float = 5.0

    # Storm window
    init_mode: str = "auto"  # "auto" | "manual"
    init_timestamp: str | None = None
    lookback_scans: int = 3

    # Optional local ADM2 boundaries
    geojson_path: str | None = None

    # Housekeeping
    save_run: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        self.thresholds = tuple(float(t) for t in self.thresholds)
        self.fss_scales_km = tuple(int(s) for s in self.fss_scales_km)
        if self.n_members < 2:
            raise ValueError("At least two ensemble members are required.")
        if self.zr_b <= 0:
            raise ValueError("Z-R exponent b must be positive.")
        if self.horizon_min < 6:
            raise ValueError("Forecast horizon must be at least 6 minutes.")
        if self.res_km <= 0 or self.grid_extent_km <= 0:
            raise ValueError("Grid extent and resolution must be positive.")

    # ------------------------------------------------------------------ utils
    @property
    def grid_shape(self) -> tuple[int, int]:
        n = int(round(2.0 * self.grid_extent_km / self.res_km)) + 1
        return n, n

    @property
    def short_label(self) -> str:
        if self.label:
            return self.label
        return "demo" if self.mode == "demo" else Path(self.data_dir).name

    def n_steps(self, timestep_min: float) -> int:
        return max(1, int(round(self.horizon_min / float(timestep_min))))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["thresholds"] = list(self.thresholds)
        data["fss_scales_km"] = list(self.fss_scales_km)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NowcastConfig":
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in allowed}
        if "thresholds" in clean:
            clean["thresholds"] = tuple(clean["thresholds"])
        if "fss_scales_km" in clean:
            clean["fss_scales_km"] = tuple(clean["fss_scales_km"])
        return cls(**clean)


@dataclass
class RunInfo:
    """Lightweight description of a stored run."""

    run_id: str
    path: Path
    created_at: str
    mode: str
    init_timestamp: str
    members: int
    steps: int
    altitude_m: float
    verified: bool
    size_mb: float = 0.0
    label: str = field(default="")
