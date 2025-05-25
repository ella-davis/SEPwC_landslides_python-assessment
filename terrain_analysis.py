# To run the program in verbose mode then make sure to use the flag -v
# python3 terrain_analysis.py --topography data/AW3D30.tif --geology data/geology_raster.tif --landcover data/Landcover.tif --faults data/Confirmed_faults.shp data/landslides.shp probability.tif
# Also make sure that verbose=True is enabled on the functions instead of verbose=False

# Import Libraries
import os
import random
import argparse

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Environment Variables for allowing deprecated SKLearn Package
os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL'] = 'True'

# A function which opens up the raster files and reads in the data, location and metadata
def convert_to_rasterio(raster_data_path, verbose=False):
    with rasterio.open(raster_data_path) as src:
        data = src.read(1)
        profile = src.profile
        transform = src.transform

        # Verbose Logging
        if verbose:
            print("\n[Load in Raster] {raster_data_path}")
            print(f"    Shape {data.shape}, Type: {data.dtype}")
            print(f"    Min: {np.min(data)}, Max: {np.max(data)}, NaNs: {np.isnan(data)}")
    return data, transform, profile

def calculate_slope(topo_array, verbose=False):
    gy, gx = np.gradient(topo_array.astype(float))
    slope = np.sqrt(gx**2 + gy**2)

    # Verbose Logging
    if verbose:
        print("\n[Calculate Slope]")
        print(f"    Shape {slope.shape}")
        print(f"    Min: {slope.min()}, Max: {slope.max()}")
    return slope

def rasterize_faults_as_distance(faults_gdf, shape, transform):
    mask_array = geometry_mask(faults_gdf.geometry, transform=transform, invert=True, out_shape=shape)
    return np.where(mask_array, 0, 9999)

# A function to plot X/Y values of where the landslides did and didn't occur
def create_training_dataframe(topo, geo, lc, dist_fault, slope, transform, landslides, verbose=False):
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

    columns=["topo", "geo", "lc", "dist_fault", "slope"]
    x = pd.DataFrame(slide_values + non_slide_values, columns=columns)
    y = pd.Series([1]*len(slide_values) + [0]*len(non_slide_values))

    # Verbose Logging
    if verbose:
        print("\n[Training Data Frame]")
        print(f"    Positive Samples: {len(slide_values)}")
        print(f"    Negative Samples: {len(non_slide_values)}")
        print("    Output First 20 Samples\n", x.head(20))
    return x, y

#Reference - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# A function to train a learning model using the raster data read in
def make_classifier(x, y, verbose=False):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    clf = RandomForestClassifier(n_estimators=100, random_state=50)
    clf.fit(x_train, y_train)

    if verbose:
        preds = clf.predict(x_test)
        print("\n[Classifier]")
        print("Classifier Accuracy:", accuracy_score(y_test, preds))
        print(f"    Important Features: {clf.feature_importances_}")
        print(f"     Sample Predictions: {preds[:5]}")

    return clf

# A function to use the trained learning model data to check the probability of a landslide occurring
def make_prob_raster(topo, geo, lc, dist_fault, slope, classifier, verbose=False):
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

        if verbose:
            print("\n[Probability Mapping]")
            print(f"    Shape: {rows} x {cols}")
            print(f"    Minimum Probability: {prob.min():.4f}, Maximum Probability: {prob.max():.4f}")

    return prob.reshape((rows, cols))

# The core function originating from the skeleton script with additional logic for outputting
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

    # Calculating the total amount of steps for non-verbose logging output
    total_steps = 7
    current_step = 0

    def log(msg):
        if not args.verbose:
            print(msg)

    # Progress bar printer
    def progress(step, label):
        if not args.verbose:
            pct = int((step / total_steps) * 100)
            bar_len = 20
            filled_len = int(bar_len * pct // 100)
            bar = "#" * filled_len + "-" * (bar_len - filled_len)
            print(f"[{bar}] {pct}% - {label}")

    # First step, reading in Raster Data
    current_step += 1
    progress(current_step, "Reading in raster data")
    topo, transform, profile = convert_to_rasterio(args.topography, verbose=args.verbose)
    geo, _, _ = convert_to_rasterio(args.geology, verbose=args.verbose)
    lc, _, _ = convert_to_rasterio(args.landcover, verbose=args.verbose)

    # Second step, reading in Shapefiles
    current_step += 1
    progress(current_step, "Reading in shapefiles")
    landslides = gpd.read_file(args.landslides)
    faults = gpd.read_file(args.faults)

    # Third step, processing input data
    current_step += 1
    progress(current_step, "Processing input data")
    dist_fault = rasterize_faults_as_distance(faults, topo.shape, transform)
    slope = calculate_slope(topo, verbose=args.verbose)

    # Fourth step, preparing training data
    current_step += 1
    progress(current_step, "Preparing training data")
    x, y = create_training_dataframe(topo, geo, lc, dist_fault, slope, transform, landslides, verbose=args.verbose)

    # Fifth Step, training classifier
    current_step += 1
    progress(current_step, "Training classifier")
    clf = make_classifier(x, y, verbose=args.verbose)

    # Sixth Step, generating probability map
    current_step += 1
    progress(current_step, "Generating probability map")
    prob_map = make_prob_raster(topo, geo, lc, dist_fault, slope, clf, verbose=args.verbose)

    # Seventh Step, exporting output files
    current_step += 1
    progress(current_step, "Exporting outputs")
    profile.update(dtype='float32', count=1)
    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    if args.dist_fault_output:
        dist_profile = profile.copy()
        dist_profile.update(dtype='float32', count=1)
        with rasterio.open(args.dist_fault_output, 'w', **dist_profile) as dst:
            dst.write(dist_fault.astype(np.float32), 1)

    if args.slope_output:
        slope_profile = profile.copy()
        slope_profile.update(dtype='float32', count=1)
        with rasterio.open(args.slope_output, 'w', **slope_profile) as dst:
            dst.write(slope.astype(np.float32), 1)

    log(f"Probability Map outputted to {args.output}")

    if args.verbose:
        print("Exported Output:", args.output)

    # Verbose Logging for Distance from Fault Sub-Function.
    if args.dist_fault_output:
        dist_profile = profile.copy()
        dist_profile.update(dtype='float32', count=1)
        with rasterio.open(args.dist_fault_output, 'w', **dist_profile) as dst:
            dst.write(dist_fault.astype(np.float32), 1)
        if args.verbose:
            print("Saved distance from fault tif raster:", args.dist_fault_output)

    # Verbose Logging for Slope Raster Sub-Function.
    if args.slope_output:
        slope_profile = profile.copy()
        slope_profile.update(dtype='float32', count=1)
        with rasterio.open(args.slope_output, 'w', **slope_profile) as dst:
            dst.write(slope.astype(np.float32), 1)
        if args.verbose:
            print("Exported slope tif raster:", args.slope_output)

# An IF statement to check whether this is 'main', if it is present then run the main() function
if __name__ == '__main__':
    main()
