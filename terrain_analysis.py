# Import Libraries
import argparse
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.features import geometry_mask

import pandas as pd 
from shapely.geometry import Point
import random

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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=50)

    clf = RandomForestClassifier(n_estimators=100, random_state=50)
    clf.fit(x_train, y_train)

    if verbose:
        preds = clf.predict(x_test)
        print("Classifier Accuracy", accuracy_score(y_test, preds))

    return clf

# A function to use the trained learning model data to check the probability of a landslide occuring.
def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):
    rows, cols = topo.shape
    X_all = np.column_stack([
        topo.ravel(),
        geo.ravel(),
        lc.ravel(),
        dist_fault.ravel(),
        slope.ravel()
        ])

    probability = classifier.predict_proba(X_all)[:, 1]
    return probability.reshape(rows, cols)

# A function to plot X/Y values of where the landslides did and didn't occur.
def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):
    slide_values = []
    for idx, row in landslides.iterrows():
        pt = row.geometry
        col, row = ~shape[0].transform * (pt.x, pt.y)
        col, row = int(col), int(row)
        values = [a[row, col] for a in [topo, geo, lc, dist_fault, slope]]
        slide_values.append(values)

    non_slide_values = []
    rows, cols = topo.shape
    while len(non_slide_values) < len(slide_values):
            r, c = random.randint(0, rows-1), random.randint(0, cols-1)
            pt = Point(shape[0].transform * (c, r))
            if not landslides.contains(pt).any():
                values = [a[r,c] for a in [topo, geo, lc, dist_fault, slope]]
                non_slide_values.append(values)

    x = pd.DataFrame(slide_values + non_slide_values, columns=["topo", "geo", "lc", "dist_fault", "slope"])
    y = pd.Series([1]*len(slide_values) + [0]*len(non_slide_values))

    return x, y

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

    if args.verbose:
        print("*** Reading in Raster Data ***")

    topo, transform, profile = convert_to_rasterio(args.topography)
    geo, _, _ = convert_to_rasterio(args.geology)
    lc, _, _ = convert_to_rasterio(args.landcover)

    if args.verbose:
        print("*** Reading in Shape Files ***")

    landslides = gpd.read_file(args.landslides)
    faults = gpd.read_file(args.faults)

    if args.verbose:
        print("*** Generating Raster ***")

    fault_mask = geometry_mask(faults.geometry, transform=transform, invert=True, out_shape=topo.shape)
    dist_fault = np.where(fault_mask, 0, 9999)

    if args.verbose:
        print("*** Calculating Slope ***")

    gy, gx = np.gradient(topo.astype(float))
    slope = np.sqrt(gx**2 + gy**2)

    if args.verbose:
        print("*** Creating Training Model Data ***")

    x, y = create_dataframe(topo, geo, lc, dist_fault, slope, [rasterio.open(args.topography)], landslides)

    if args.verbose:
        print("*** Classifier Training ***")

    clf = make_classifier(x, y, verbose=args.verbose)

    if args.verbose:
        print("*** Generating Probability Raster ***")

    prob_map = make_prob_raster_data(topo, geo, lc, dist_fault, slope, clf)

    if args.verbose:
        print("*** Logging Output ***")

    profile.update(dtype='float32', count=1)
    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    if args.verbose:
        print("Exported Output:", args.output)


if __name__ == '__main__':
    main()
