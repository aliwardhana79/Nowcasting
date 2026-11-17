"""East Java district boundaries and radar-centred coordinate helpers."""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .config import CACHE_DIR

GEOBOUNDARIES_META = "https://www.geoboundaries.org/api/current/gbOpen/IDN/ADM2/"
CACHE_FILE = CACHE_DIR / "idn_adm2_geoboundaries.geojson"

# Last-resort labels so the dashboard remains useful without a GeoJSON file.
FALLBACK_CITIES = [
    ("Surabaya", -7.2575, 112.7521),
    ("Gresik", -7.1560, 112.6550),
    ("Sidoarjo", -7.4474, 112.7183),
    ("Bangkalan", -7.0455, 112.7350),
    ("Mojokerto", -7.4700, 112.4380),
    ("Lamongan", -7.1190, 112.4170),
    ("Tuban", -6.8970, 112.0650),
    ("Bojonegoro", -7.1500, 111.8820),
    ("Jombang", -7.5460, 112.2340),
    ("Pasuruan", -7.6470, 112.9070),
    ("Probolinggo", -7.7540, 113.2160),
    ("Malang", -7.9660, 112.6330),
    ("Batu", -7.8670, 112.5230),
    ("Kediri", -7.8160, 112.0110),
    ("Madiun", -7.6290, 111.5230),
    ("Blitar", -8.0950, 112.1610),
    ("Banyuwangi", -8.2190, 114.3690),
    ("Jember", -8.1680, 113.7020),
    ("Sumenep", -7.0050, 113.8490),
    ("Pamekasan", -7.1570, 113.4750),
    ("Sampang", -7.1870, 113.2390),
    ("Trenggalek", -8.0500, 111.7100),
    ("Tulungagung", -8.0650, 111.9020),
    ("Nganjuk", -7.6050, 111.9030),
    ("Ngawi", -7.4100, 111.4460),
    ("Pacitan", -8.1940, 111.1010),
    ("Ponorogo", -7.8650, 111.4620),
    ("Situbondo", -7.7060, 114.0090),
    ("Bondowoso", -7.9130, 113.8210),
    ("Lumajang", -8.1340, 113.2250),
    ("Magetan", -7.6500, 111.3300),
]


# --------------------------------------------------------------- coordinates
def _wgs84_factors(radar_lat: float) -> tuple[float, float]:
    lat_rad = np.deg2rad(radar_lat)
    a_km = 6378.137
    e2 = 0.00669437999014
    n_rad = a_km / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    m_rad = a_km * (1.0 - e2) / (1.0 - e2 * np.sin(lat_rad) ** 2) ** 1.5
    km_per_deg_lon = (np.pi / 180.0) * n_rad * np.cos(lat_rad)
    km_per_deg_lat = (np.pi / 180.0) * m_rad
    return float(km_per_deg_lon), float(km_per_deg_lat)


def lonlat_to_xy(
    lon, lat, radar_lat: float, radar_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    km_per_deg_lon, km_per_deg_lat = _wgs84_factors(radar_lat)
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    return (lon_arr - radar_lon) * km_per_deg_lon, (lat_arr - radar_lat) * km_per_deg_lat


def latlon_to_xy(
    lat: float, lon: float, radar_lat: float, radar_lon: float
) -> tuple[float, float]:
    x, y = lonlat_to_xy(lon, lat, radar_lat, radar_lon)
    return float(x), float(y)


# ------------------------------------------------------------------- download
def _read_geojson(path: Path) -> dict | None:
    try:
        if path.exists() and path.stat().st_size > 1000:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return None


def _download_geojson(destination: Path = CACHE_FILE) -> dict | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    contexts = [ssl.create_default_context(), ssl._create_unverified_context()]
    headers = {"User-Agent": "SurabayaNowcastDashboard/1.0"}
    for context in contexts:
        try:
            request = urllib.request.Request(GEOBOUNDARIES_META, headers=headers)
            with urllib.request.urlopen(request, timeout=20, context=context) as resp:
                meta = json.loads(resp.read().decode("utf-8"))
            url = meta.get("gjDownloadURL") or meta.get("simplifiedGeometryGeoJSON")
            if not url:
                continue
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45, context=context) as resp:
                payload = resp.read()
            destination.write_bytes(payload)
            return json.loads(payload.decode("utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return None


# -------------------------------------------------------------------- parsing
def _polys_of(geometry: dict) -> list:
    gtype = geometry.get("type")
    if gtype == "Polygon":
        return [geometry.get("coordinates", [])]
    if gtype == "MultiPolygon":
        return geometry.get("coordinates", [])
    return []


def _base_name(raw: str) -> tuple[str, str]:
    text = " ".join(str(raw).split())
    lower = text.lower()
    if lower.startswith("kota "):
        base = text[5:]
        return base, f"{base} (Kota)"
    if lower.startswith("kabupaten "):
        base = text[10:]
        return base, f"{base} (Kab.)"
    if lower.startswith("kab. "):
        base = text[5:]
        return base, f"{base} (Kab.)"
    return text, text


def _fallback(radar_lat: float, radar_lon: float, max_range_km: float) -> dict:
    labels = []
    for name, lat, lon in FALLBACK_CITIES:
        x, y = latlon_to_xy(lat, lon, radar_lat, radar_lon)
        if np.hypot(x, y) <= max_range_km * 0.95:
            labels.append((x, y, name))
    return {
        "source": "built-in city markers (no district boundaries)",
        "labels": labels,
        "other_segments": [],
        "surabaya_segments": [],
        "polygons": [],
        "radar_lat": radar_lat,
        "radar_lon": radar_lon,
    }


def load_district_data(
    radar_lat: float,
    radar_lon: float,
    max_range_km: float = 150.0,
    local_geojson: str | None = None,
    allow_download: bool = True,
) -> dict:
    """Load ADM2 boundaries and project them to radar-centred kilometres."""
    geo: dict | None = None
    source = ""
    if local_geojson and Path(local_geojson).expanduser().exists():
        geo = _read_geojson(Path(local_geojson).expanduser())
        source = f"local file · {Path(local_geojson).name}"
    if geo is None:
        geo = _read_geojson(CACHE_FILE)
        if geo is not None:
            source = "cached geoBoundaries ADM2"
    if geo is None and allow_download:
        geo = _download_geojson()
        if geo is not None:
            source = "geoBoundaries ADM2 (downloaded)"
    if geo is None:
        return _fallback(radar_lat, radar_lon, max_range_km)

    other_segments: list[np.ndarray] = []
    surabaya_segments: list[np.ndarray] = []
    polygons: list[dict] = []
    raw_labels: list[tuple[float, float, str]] = []
    reach = max_range_km * 1.35

    for feat in geo.get("features", []):
        props = feat.get("properties", {}) or {}
        name = str(
            props.get("shapeName")
            or props.get("WADMKK")
            or props.get("NAME_2")
            or props.get("name")
            or ""
        ).strip()
        if not name:
            continue
        geometry = feat.get("geometry") or {}
        polys = _polys_of(geometry)
        if not polys:
            continue

        # Quick geographic pre-filter around the radar site
        all_lon: list[float] = []
        all_lat: list[float] = []
        for poly in polys:
            for ring in poly:
                pts = np.asarray(ring, dtype=float)
                if pts.ndim == 2 and pts.shape[0] >= 3:
                    all_lon.extend(pts[:, 0].tolist())
                    all_lat.extend(pts[:, 1].tolist())
        if not all_lon:
            continue
        cx_lon, cy_lat = float(np.mean(all_lon)), float(np.mean(all_lat))
        cx, cy = latlon_to_xy(cy_lat, cx_lon, radar_lat, radar_lon)
        if np.hypot(cx, cy) > reach:
            continue

        base, display = _base_name(name)
        rings_km: list[np.ndarray] = []
        centroid_x: list[np.ndarray] = []
        centroid_y: list[np.ndarray] = []
        for poly in polys:
            for ring in poly:
                pts = np.asarray(ring, dtype=float)
                if pts.ndim != 2 or pts.shape[0] < 4:
                    continue
                x_km, y_km = lonlat_to_xy(pts[:, 0], pts[:, 1], radar_lat, radar_lon)
                segment = np.column_stack([x_km, y_km])
                rings_km.append(segment)
                centroid_x.append(x_km)
                centroid_y.append(y_km)
                if "surabaya" in display.lower():
                    surabaya_segments.append(segment)
                else:
                    other_segments.append(segment)
        if not rings_km:
            continue

        polygons.append(
            {"name": name, "display_name": display, "rings": rings_km}
        )
        mx = float(np.mean(np.concatenate(centroid_x)))
        my = float(np.mean(np.concatenate(centroid_y)))
        if np.hypot(mx, my) <= max_range_km * 0.92:
            raw_labels.append((mx, my, base))

    # Keep labels readable: prioritise Surabaya and nearby districts, and
    # enforce a minimum separation so labels do not pile up on small screens.
    ordered = sorted(
        raw_labels,
        key=lambda item: (
            0 if "surabaya" in item[2].lower() else 1,
            np.hypot(item[0], item[1]),
        ),
    )
    labels: list[tuple[float, float, str]] = []
    for cx, cy, name in ordered:
        if any(
            name == existing_name
            and np.hypot(cx - existing_x, cy - existing_y) < 30.0
            for existing_x, existing_y, existing_name in labels
        ):
            continue
        if any(
            np.hypot(cx - existing_x, cy - existing_y) < 42.0
            for existing_x, existing_y, _ in labels
        ):
            continue
        labels.append((cx, cy, name))
        if len(labels) >= 12:
            break

    if not polygons:
        return _fallback(radar_lat, radar_lon, max_range_km)

    return {
        "source": source or "ADM2 boundaries",
        "labels": labels,
        "other_segments": other_segments,
        "surabaya_segments": surabaya_segments,
        "polygons": polygons,
        "radar_lat": radar_lat,
        "radar_lon": radar_lon,
    }


def build_district_masks(
    district_data: dict,
    extent_km: float,
    res_km: float,
    min_area_km2: float = 4.0,
) -> list[tuple[str, np.ndarray, float]]:
    """Rasterize district polygons to boolean masks on the nowcast grid."""
    from matplotlib.path import Path

    n = int(round(2.0 * extent_km / res_km)) + 1
    coords = np.linspace(-extent_km, extent_km, n)
    gx, gy = np.meshgrid(coords, coords)
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    masks: list[tuple[str, np.ndarray, float]] = []

    for poly in district_data.get("polygons", []):
        rings = poly.get("rings", [])
        if not rings:
            continue
        mask = np.zeros(flat.shape[0], dtype=bool)
        min_x = min(float(np.min(r[:, 0])) for r in rings)
        max_x = max(float(np.max(r[:, 0])) for r in rings)
        min_y = min(float(np.min(r[:, 1])) for r in rings)
        max_y = max(float(np.max(r[:, 1])) for r in rings)
        if max_x < -extent_km or min_x > extent_km:
            continue
        if max_y < -extent_km or min_y > extent_km:
            continue
        bbox = (
            (flat[:, 0] >= min_x - res_km)
            & (flat[:, 0] <= max_x + res_km)
            & (flat[:, 1] >= min_y - res_km)
            & (flat[:, 1] <= max_y + res_km)
        )
        if not bbox.any():
            continue
        points = flat[bbox]
        for ring in rings:
            if ring.shape[0] < 4:
                continue
            try:
                mask[bbox] |= Path(ring, closed=True).contains_points(points)
            except Exception:  # noqa: BLE001
                continue
        count = int(mask.sum())
        area = count * res_km * res_km
        if area < min_area_km2:
            continue
        masks.append((poly.get("display_name") or poly.get("name", "District"), mask.reshape(n, n), area))

    masks.sort(key=lambda item: item[0])
    return masks


def district_probability_table(
    result,
    lead_idx: int,
    threshold: float,
    masks: list[tuple[str, np.ndarray, float]],
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Mean/max exceedance probability and high-probability area per district."""
    prob = result.probability_field(lead_idx, threshold)
    rows: list[dict[str, Any]] = []
    for name, mask, area in masks:
        values = prob[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "District": name,
                "Mean probability (%)": float(np.mean(values)),
                "Max probability (%)": float(np.max(values)),
                "Area ≥ 50% (km²)": float(np.sum(values >= 50.0) * result.res_km**2),
                "Area (km²)": float(area),
            }
        )
    rows.sort(key=lambda row: (row["Mean probability (%)"], row["Max probability (%)"]), reverse=True)
    return rows[:top_n]
