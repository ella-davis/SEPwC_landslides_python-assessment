import os
os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL'] = 'True'

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# A function which opens up the raster files and reads in the data, location and metadata.
def convert_to_rasterio(raster_data_path, template_raster_path = None):
    with rasterio.open(raster_data_path) as src:
        data = src.read(1)
        profile = src.profile
        transform = src.transform
    return data, transform, profile

def calculate_slope(topo_array):
    gy, gx = np.gradient(topo_array.astype(float))
    return np.sqrt(gx**2 + gy**2)

def rasterize_faults_as_distance(faults_gdf, shape, transform):
    mask_array = geometry_mask(faults_gdf.geometry, transform=transform, invert=True, out_shape=shape)
    return np.where(mask_array, 0, 9999)

# A function to plot X/Y values of where the landslides did and didn't occur.
def create_training_dataframe(topo, geo, lc, dist_fault, slope, transform, landslides):
    slide_values = []
    for idx, row in landslides.iterrows():
        pt = row.geometry.centroid
        col, row_idx = ~transform * (pt.x, pt.y)
        col, row_idx = int(col), int(row_idx)
        if 0 <= row_idx < topo.shape[0] and 0 <= col < topo.shape[1]:
            values = [a[row_idx, col] for a in [topo, geo, lc, dist_fault, slope]]
            slide_values.append(values)

    non_slide_values = []
    rows, cols = topo.shape
    while len(non_slide_values) < len(slide_values):
        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
        pt = Point(transform * (c, r))
        if not landslides.contains(pt).any():
            values = [a[r, c] for a in [topo, geo, lc, dist_fault, slope]]
            non_slide_values.append(values)

    x = pd.DataFrame(slide_values + non_slide_values, columns=["topo", "geo", "lc", "dist_fault", "slope"])
    y = pd.Series([1]*len(slide_values) + [0]*len(non_slide_values))
    return x, y

#Reference - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# A function to train a learning model using the raster data read in.
def make_classifier(x, y, verbose=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    clf = RandomForestClassifier(n_estimators=100, random_state=50)
    clf.fit(x_train, y_train)

    if verbose:
        preds = clf.predict(x_test)
        print("Classifier Accuracy:", accuracy_score(y_test, preds))

    return clf

# A function to use the trained learning model data to check the probability of a landslide occuring.
def make_prob_raster(topo, geo, lc, dist_fault, slope, classifier):
    rows, cols = topo.shape
    X_all = pd.DataFrame(np.column_stack([
        topo.ravel(),
        geo.ravel(),
        lc.ravel(),
        dist_fault.ravel(),
        slope.ravel()
    ]), columns=["topo", "geo", "lc", "dist_fault", "slope"])

    proba = classifier.predict_proba(X_all)

    if 1 in classifier.classes_:
        index = list(classifier.classes_).index(1)
        prob = proba[:, index]
    else:
        print("Warning: Class 1 (landslide) was not present in training data. Returning 0 probabilities.")
        prob = np.zeros(X_all.shape[0])

    return prob.reshape((rows, cols))

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
    parser.add_argument('--slope-output',
                    help="outputs the slope output into .tif format")
    parser.add_argument('--dist-fault-output',
                    help="outputs the distance from fault into .tif format")

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

    dist_fault = rasterize_faults_as_distance(faults, topo.shape, transform)

    if args.verbose:
        print("*** Calculating Slope ***")

    slope = calculate_slope(topo)

    if args.verbose:
        print("*** Creating Training Model Data ***")

    x, y = create_training_dataframe(topo, geo, lc, dist_fault, slope, transform, landslides)

    if args.verbose:
        print("*** Classifier Training ***")

    clf = make_classifier(x, y, verbose=args.verbose)

    if args.verbose:
        print("*** Generating Probability Raster ***")

    prob_map = make_prob_raster(topo, geo, lc, dist_fault, slope, clf)

    if args.verbose:
        print("*** Logging Output ***")

    profile.update(dtype='float32', count=1)
    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    if args.verbose:
        print("Exported Output:", args.output)

    if args.dist_fault_output:
        dist_profile = profile.copy()
        dist_profile.update(dtype='float32', count=1)
        with rasterio.open(args.dist_fault_output, 'w', **dist_profile) as dst:
            dst.write(dist_fault.astype(np.float32), 1)
        if args.verbose:
            print("Saved distance from fault tif raster:", args.dist_fault_output)

    if args.slope_output:
        slope_profile = profile.copy()
        slope_profile.update(dtype='float32', count=1)
        with rasterio.open(args.slope_output, 'w', **slope_profile) as dst:
            dst.write(slope.astype(np.float32), 1)
        if args.verbose:
            print("Exported slope tif raster:", args.slope_output)

if __name__ == '__main__':
    main()
