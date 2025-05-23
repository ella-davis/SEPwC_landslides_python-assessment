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

#Reference - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# A function to train a learning model using the raster data read in.
def make_classifier(x, y, verbose=False):
    from skLearn.ensemble import RandomForestClassifier
    from skLearn.modal_selection import train
    fromskLearn.metrics import accuracy_score

    x_train, x_test, y_train, y=test = train(x,y,test_size=0.2, random_state=50)

    clf = RandomForestClassifier(n_estimators=100, random_state=50)
    clf.fit(x_train, y_train)

    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):

    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    return

# The core function
def main():

    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument('--topography',
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("landslides",
                    help="the landslide location shapefile")
    parser.add_argument("output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()


if __name__ == '__main__':
    main()
