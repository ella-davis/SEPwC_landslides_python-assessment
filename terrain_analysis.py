# Import Libraries
import argparse
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.features import geometry_mask
