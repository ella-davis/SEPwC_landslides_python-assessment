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

# A function which extracts part of the raster using a shape mask.
def extract_values_from_raster(raster_path, shape_object):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, shape_object.geometry, crop = True)
        return out_image[0] 