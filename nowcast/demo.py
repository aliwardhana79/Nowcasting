"""Fast synthetic storm generator used for the dashboard demo mode.

The demo is deliberately labelled as synthetic everywhere in the UI. It is a
physically plausible advecting-cell model, not a substitute for a radar run.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import NowcastConfig

DEMO_DATE = datetime(2025, 3, 6, 14, 38, 0)
DEMO_TIMESTEP_MIN = 6.0
DEMO_RADAR_LAT = -7.410306
DEMO_RADAR_LON = 112.760422


@dataclass
class DemoCell:
    x: float
    y: float
    u: float
    v: float
    amp: float
    sigma: float
    peak_min: float
    life_min: float


BASE_CELLS = [
    DemoCell(-38, 4, 27.0, 12.0, 42.0, 8.5, 35.0, 170.0),
    DemoCell(-78, -26, 24.0, 15.0, 30.0, 11.0, 65.0, 200.0),
    DemoCell(-8, -62, 29.0, 8.0, 27.0, 9.5, 80.0, 170.0),
    DemoCell(48, -18, 25.0, 14.0, 20.0, 8.0, 25.0, 150.0),
    DemoCell(-105, 38, 22.0, 11.0, 22.0, 13.0, 105.0, 210.0),
    DemoCell(-20, 30, 23.0, 9.0, 3.2, 55.0, 95.0, 320.0),  # stratiform envelope
]


def _coords(config: NowcastConfig) -> np.ndarray:
    n = int(round(2.0 * config.grid_extent_km / config.res_km)) + 1
    return np.linspace(-config.grid_extent_km, config.grid_extent_km, n)


def _add_cell(
    field: np.ndarray,
    coords: np.ndarray,
    cx: float,
    cy: float,
    amplitude: float,
    sigma: float,
) -> None:
    if amplitude <= 0.02 or sigma <= 0.2:
        return
    half = 3.6 * sigma
    j0 = int(np.searchsorted(coords, cx - half, side="left"))
    j1 = int(np.searchsorted(coords, cx + half, side="right"))
    i0 = int(np.searchsorted(coords, cy - half, side="left"))
    i1 = int(np.searchsorted(coords, cy + half, side="right"))
    if i0 >= i1 or j0 >= j1:
        return
    yy, xx = np.meshgrid(coords[i0:i1], coords[j0:j1], indexing="ij")
    field[i0:i1, j0:j1] += amplitude * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)
    )


def _cell_amplitude(cell: DemoCell, t_min: float) -> float:
    return cell.amp * float(np.exp(-((t_min - cell.peak_min) / cell.life_min) ** 2))


def _field(
    cells: list[DemoCell],
    t_min: float,
    coords: np.ndarray,
    rng: np.random.Generator,
    texture_amplitude: float = 0.22,
    bias: float = 1.0,
) -> np.ndarray:
    field = np.zeros((coords.size, coords.size), dtype=np.float32)
    for cell in cells:
        cx = cell.x + cell.u * t_min / 60.0
        cy = cell.y + cell.v * t_min / 60.0
        amp = _cell_amplitude(cell, t_min)
        _add_cell(field, coords, cx, cy, amp, cell.sigma)

    texture = gaussian_filter(rng.normal(size=field.shape), sigma=2.2)
    texture /= max(float(np.std(texture)), 1e-6)
    field *= np.clip(1.0 + texture_amplitude * texture, 0.15, None).astype(np.float32)
    field *= bias
    field[field < 0.12] = 0.0
    return field


def _perturb_cells(
    cells: list[DemoCell],
    rng: np.random.Generator,
    position_sigma: float = 4.5,
) -> list[DemoCell]:
    out = []
    for cell in cells:
        out.append(
            replace(
                cell,
                x=cell.x + rng.normal(0.0, position_sigma),
                y=cell.y + rng.normal(0.0, position_sigma),
                u=cell.u + rng.normal(0.0, 2.4),
                v=cell.v + rng.normal(0.0, 2.4),
                amp=cell.amp * float(rng.lognormal(0.0, 0.12)),
                sigma=cell.sigma * float(rng.uniform(0.9, 1.12)),
                peak_min=cell.peak_min + rng.normal(0.0, 8.0),
            )
        )
    return out


def _stamp(when: datetime) -> str:
    return when.strftime("%Y%m%d%H%M%S") + "00"


def generate_demo_arrays(config: NowcastConfig, progress=None) -> dict:
    """Create analysis, truth and a stochastic ensemble on the run grid."""

    def emit(fraction: float, message: str) -> None:
        if progress is not None:
            progress(float(np.clip(fraction, 0.0, 1.0)), message)

    timestep = DEMO_TIMESTEP_MIN
    n_steps = config.n_steps(timestep)
    coords = _coords(config)
    n = coords.size
    rng = np.random.default_rng(int(config.seed))

    emit(0.04, "Synthesising a convective storm cell complex")
    truth_cells = _perturb_cells(BASE_CELLS, np.random.default_rng(int(config.seed) + 7), 2.0)

    rain_stack = []
    for k in (-2, -1, 0):
        t_min = k * timestep
        frame_rng = np.random.default_rng(int(config.seed) + 100 + k)
        rain_stack.append(_field(truth_cells, t_min, coords, frame_rng, 0.18, 1.0))
    rain_stack = np.stack(rain_stack).astype(np.float32)

    obs_frames = []
    for step in range(1, n_steps + 1):
        t_min = step * timestep
        frame_rng = np.random.default_rng(int(config.seed) + 300 + step)
        obs = _field(truth_cells, t_min, coords, frame_rng, 0.24, 1.04)
        # A small advection/amplitude mismatch makes verification non-trivial.
        obs = np.roll(obs, shift=(0, 1), axis=(0, 1))
        obs_frames.append(obs)
    obs = np.stack(obs_frames).astype(np.float32)

    ensemble = np.zeros(
        (int(config.n_members), n_steps, n, n), dtype=np.float32
    )
    for member in range(int(config.n_members)):
        member_rng = np.random.default_rng(int(config.seed) + 1000 + member)
        cells = _perturb_cells(BASE_CELLS, member_rng, 5.0)
        intensity_bias = float(member_rng.lognormal(0.0, 0.10))
        for step in range(1, n_steps + 1):
            t_min = step * timestep
            noise_rng = np.random.default_rng(
                int(config.seed) + 10000 + member * 1000 + step
            )
            ensemble[member, step - 1] = _field(
                cells,
                t_min,
                coords,
                noise_rng,
                texture_amplitude=0.30,
                bias=intensity_bias,
            )
        if member % 4 == 0 or member == int(config.n_members) - 1:
            emit(
                0.08 + 0.62 * (member + 1) / int(config.n_members),
                f"Generating ensemble member {member + 1}/{int(config.n_members)}",
            )

    init_dt = DEMO_DATE
    timestamps = [
        _stamp(init_dt + timedelta(minutes=k * timestep)) for k in (-2, -1, 0)
    ]
    obs_timestamps = [
        _stamp(init_dt + timedelta(minutes=(step + 1) * timestep))
        for step in range(n_steps)
    ]

    # Dense, gently varying motion field for display and consistency.
    u_px = 25.0 / config.res_km * timestep / 60.0
    v_px = 11.0 / config.res_km * timestep / 60.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    velocity = np.stack(
        [
            u_px + 0.03 * u_px * np.sin(xx / 55.0 + yy / 90.0),
            v_px + 0.03 * v_px * np.cos(yy / 60.0 - xx / 110.0),
        ]
    ).astype(np.float32)

    return {
        "rain_stack": rain_stack,
        "ensemble": ensemble,
        "obs": obs,
        "velocity": velocity,
        "timestamps": timestamps,
        "obs_timestamps": obs_timestamps,
        "init_dt": init_dt,
        "timestep_min": timestep,
        "meta": {
            "demo": True,
            "demo_note": "Synthetic advecting-cell storm; not observed rainfall.",
            "n_steps": n_steps,
        },
    }
