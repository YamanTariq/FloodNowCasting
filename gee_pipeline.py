import ee
import geemap
import numpy as np


def fetch_regional_data(bbox, project: str):
    ee.Initialize(project=project)
    min_lon, min_lat, max_lon, max_lat = bbox
    region = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

    # --- DEM ---
    dem = ee.Image('USGS/SRTMGL1_003').clip(region)
    # scale=30 matches SRTM native resolution
    dem_array = geemap.ee_to_numpy(dem, region=region, scale=30)
    if len(dem_array.shape) == 3:
        dem_array = dem_array[:, :, 0]

    # --- Land Cover ---
    landcover = ee.ImageCollection("ESA/WorldCover/v100").first().clip(region)
    landcover_array = geemap.ee_to_numpy(landcover, region=region, scale=10)
    if len(landcover_array.shape) == 3:
        landcover_array = landcover_array[:, :, 0]

    # --- Sentinel-2 RGB (10m) ---
    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region)
        .filterDate('2023-01-01', '2024-01-01')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))   # prefer cloud-free
        .select(['B4', 'B3', 'B2'])
        .median()
        .clip(region)
    )

    # Fetch raw DN values at native 10m resolution
    # IMPORTANT: pass scale=10 explicitly so geemap doesn't use a coarse default
    rgb_raw = geemap.ee_to_numpy(s2, region=region, scale=10)  # shape (H, W, 3), uint16 range ~0-10000

    # Robust normalisation: stretch to full 0-255 range per-channel
    rgb_float = rgb_raw.astype(np.float32)
    # Typical S2 SR surface reflectance is ~0-3000 for urban/vegetated, cap at 3500
    rgb_clipped = np.clip(rgb_float, 0, 3500)
    rgb_norm = (rgb_clipped / 3500.0 * 255).astype(np.uint8)

    # Safety: if image came back already in byte range (some geemap versions rescale)
    if rgb_raw.max() <= 255:
        rgb_norm = np.clip(rgb_raw, 0, 255).astype(np.uint8)

    return dem_array, landcover_array, rgb_norm