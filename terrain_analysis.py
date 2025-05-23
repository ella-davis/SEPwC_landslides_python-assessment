# Import Libraries
import argparse
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.features import geometry_mask

# A function which opens up the raster files and reads in the data, location and metadata.
def convert_to_rasterio(raster_data_path, template_raster_path = None):
    with rasterio.open(raster_data_path) as src:
        data = src.read(1)
        profile = src.profile
        transform = src.transform
    return data, transform, profile