import ee
import geemap
import numpy as np

def fetch_regional_data(bbox, project: str):
    ee.Initialize(project=project)
    min_lon, min_lat, max_lon, max_lat = bbox
    region = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

    dem = ee.Image('USGS/SRTMGL1_003').clip(region)
    dem_array = geemap.ee_to_numpy(dem, region=region)
    if len(dem_array.shape) == 3: dem_array = dem_array[:, :, 0]

    landcover = ee.ImageCollection("ESA/WorldCover/v100").first().clip(region)
    landcover_array = geemap.ee_to_numpy(landcover, region=region)
    if len(landcover_array.shape) == 3: landcover_array = landcover_array[:, :, 0]

    # Forcing scale=10 prevents the "zoomed out" single-pixel bug
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate('2023-01-01', '2024-01-01')
          .select(['B4', 'B3', 'B2']) 
          .median()                   
          .setDefaultProjection(crs='EPSG:4326', scale=10) 
          .clip(region))
    
    # Brighten for visualization
    rgb_array = geemap.ee_to_numpy(s2.divide(2500).clamp(0, 1).multiply(255).uint8(), region=region)
    return dem_array, landcover_array, rgb_array