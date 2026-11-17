# Surabaya Radar Nowcast Lab

An interactive Streamlit dashboard for the BMKG Surabaya **Baron** C-band dual-pol radar.
It turns Rainbow5 volumes into a quality-controlled 1.5 km pseudo-CAPPI, runs a
STEPS ensemble nowcast up to three hours ahead, and verifies the forecast against
future radar scans.

> Research and training prototype. This is **not** an operational warning service,
> and radar-derived rain rates are indirect estimates.

## What the dashboard does

| Stage | Processing |
| --- | --- |
| **Ingest** | Reads Rainbow5 `dBZ`, `RhoHV`, `ZDR` and `PhiDP`/`uPhiDP` volumes and decodes 16-bit azimuth angles to degrees. |
| **QC** | Keeps meteorological echoes with `dBZ ≥ 5`, `RhoHV ≥ 0.85` and `|ZDR| ≤ 5 dB`. |
| **Attenuation** | Smooths and unwraps ΦDP, estimates system phase from the near range, and applies a capped C-band path-integrated attenuation correction (up to 10 dB). |
| **CAPPI** | Projects each sweep to a 1.5 km pseudo-CAPPI, chooses the bracketing elevations at every grid cell and interpolates vertically. |
| **Rain rate** | Converts corrected reflectivity with `Z = 250 R^1.2`, capped at 150 mm/h. |
| **Nowcast** | Lucas–Kanade optical-flow motion plus a stochastic STEPS cascade (6 levels, 20 members by default), up to 120 minutes at radar cadence. |
| **Verification** | Scores each lead time against the nearest future pseudo-CAPPI: CRPS, Brier score, ROC AUC, FSS, MAE, bias, spatial correlation and ensemble spread. |
| **Geography** | Downloads and caches geoBoundaries ADM2 boundaries, rasterizes district masks and reports exposure around Surabaya. |

## Interactive views

- **Overview** — initial pseudo-CAPPI, +60 min exceedance probability, storm motion, peak/Surabaya probabilities and headline skill.
- **Probability** — threshold and lead-time explorer with district exposure bars and table.
- **Observed vs forecast** — observed, ensemble-mean and difference maps, scatter density and local error metrics.
- **Verification** — CRPS, Brier, ROC AUC, FSS, MAE, bias, correlation and spread curves, plus a score table.
- **Ensemble** — member and ensemble-mean maps, member statistics and a member-mean distribution.
- **Diagnostics** — archive activity profile, motion vectors, CAPPI processing metadata and method notes.
- **Exports** — verification CSV, run configuration JSON and publication-quality PNG panels.

Every result stores its configuration and can be reloaded from the **Run history**
panel. Runs are written under `runs/` with float16-compressed ensembles.

## Quick start

The project uses a local virtual environment; the commands below assume it lives
in the repository's parent directory.

```bash
# From the directory containing this README
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m streamlit run app.py
```

Then open <http://localhost:8501>.

### Data

Point the sidebar at a folder of Rainbow5 volumes, for example:

```
/Users/aliwardhana/Downloads/Training_TL/00_data/Radar/2025-03-06
```

Each scan needs four files sharing a timestamp prefix:

```
20250306143800dBZ.vol
20250306143800RhoHV.vol
20250306143800ZDR.vol
20250306143800PhiDP.vol   # or uPhiDP.vol
```

The default path can be changed with the `SBY_RADAR_DIR` environment variable.

### Synthetic demo

Switch **Data source** to *Synthetic demo* to generate an advecting storm complex
with a 20-member ensemble. Demo output is labelled **SYNTHETIC DEMO** everywhere
and never mixes with radar observations. It is useful for exploring the workflow
without the archive.

## Repository layout

```
surabaya_nowcast_dashboard/
├── app.py                 # Streamlit UI
├── requirements.txt
├── .streamlit/config.toml
├── nowcast/
│   ├── config.py          # NowcastConfig, paths, defaults
│   ├── radar.py           # Rainbow5 decode, QC, ΦDP correction, CAPPI
│   ├── nowcasting.py      # motion, STEPS ensemble, verification metrics
│   ├── pipeline.py        # run orchestration and workload estimate
│   ├── demo.py            # synthetic storm generator
│   ├── geodata.py         # ADM2 boundaries, masks, district exposure
│   ├── models.py          # NowcastResult
│   ├── store.py           # run save/load/history
│   ├── viz.py             # Plotly figures
│   └── plotting.py        # matplotlib publication figures
└── runs/                  # saved runs (created on first save)
```

Caches live under `~/.cache/surabaya_nowcast/`:

- `idn_adm2_geoboundaries.geojson` — downloaded district boundaries.
- `activity_<hash>.json` — per-archive convective activity profiles.

## Configuration notes

- Default domain: 300 km across (150 km radius) at 1 km grid spacing.
- Default CAPPI height: 1500 m.
- Default Z–R relation: `a = 250`, `b = 1.2`.
- Default STEPS settings: 20 members, 6 cascade levels, seed 42.
- Verification thresholds: 1, 5 and 10 mm/h; FSS neighbourhoods: 5, 15 and 30 km.
- Manual storm initialization is available; auto mode selects the strongest scan
  in the archive that still has enough future scans for verification.

A full 20-member, 120-minute run on the 301 × 301 grid is substantially heavier
than the demo. Use fewer members, a shorter horizon or a coarser grid when testing.
The sidebar shows an estimate before launching a radar run.

## Method references

- Seed (2003) — STEPS stochastic nowcasting.
- Lucas & Kanade (1981) — optical-flow motion estimation.
- Roberts & Lean (2008) — Fractions Skill Score.
- geoBoundaries — ADM2 administrative boundaries.

Radar-based verification is not the same as verification against rain gauges.
Low-level radar coverage, beam broadening, attenuation residue and the Z–R
assumption all affect the scores. Treat the dashboard as a research tool for
exploring sensitivity and skill, not as an authoritative forecast.
# Nowcasting
