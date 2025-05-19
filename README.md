# SEPwC Landslide Risk Coursework (Python)

## Introduction

Your task is to write code to read in muliple raster and shapefiles, perform some analysis, 
and then generate a risk (probability) map of landslides. You should output this as a 
raster (with values from 0 to 1).

## The tests

The test suite uses a number of test data sets. The tests will check these data
functions work. 

You can run the tests by running `pytest` or `pytest test/test_terrain.py`
from the main directory. Try it now, before you make any changes!

You can run a single test using:

```bash
pytest test/test_terrian.py::TestTerrainAnalysis::test_convert_rasterio
```

You can run the unit tests only:

```bash
pytest test/test_terrain.py::TestTerrainAnalysis
```

The regression tests check the whole code:

```bash
pytest test/test_terrain.py::TestRegression
```

You'll notice another test file: `test_distance.py`. This is because I could
not find a decent function to work out the distance from a value in Python. So I
had to write one. This is the test suite for that code.

## The data

There are a number of rasters and shapefiles for this task:

 - `AW3D30.tif` - topography data
 - `Confirmed_faults.shp` - fault location data
 - `Geology.tif` - rock types across the region
 - `Lancover.tif` - landcover codes
 - `landslides.shp` - Landslide occurances

From those you will also need to generate slope raster and a "distance from fault" raster.

Your code should run like:

```bash
python3 terrain_analysis.py --topography data/AW3D30.tif --geology data/geology_raster.tif --landcover data/Landcover.tif --faults data/Confirmed_faults.shp data/landslides.shp probability.tif
```

## Hints and tips

Use `geopandas` to load shapefile data and manage most of the data wrangling. 

The `rasterio` module can handle the raster data and a number of other features you need.

The [`sklearn.ensamble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
can be used to generate the fitting function to generate 
the probability map. This can be done by extracting data from all of your rasters under the landslides shapefile
into a geopandas data frame. You'll also need negative samples, i.e. where landslides do not occur 
(hint: will need to be the same length as the landslide data!). You send these into
The RF classifier and then use `predict` to make a model!

The data in the RF classifier should probably be split into "test" and "train", there is a function
in `sklearn` to do that for you (`test_train_split`). It might be helpful to print the accuracy score
or other metrics if the verbose flag is on, perhaps. You can find more random forests in python on [this
webpage](https://www.datacamp.com/tutorial/random-forests-classifier-python).

## The rules

You cannot alter any of the assert comments in `test/test_terrain.py`

If you alter any function names in the main code, you *can* alter the name
in the test file to match or split the tests into different chunks; however 
the actual test, i.e. the asserts, must remain unchanged. This will be checked.
If you alter the tests, you may get conflicts if this main repository is changed. 
It is your job to fix any. 

If you wish to add more tests, please do, but place them in a separate file
in the `test` directory. Remember to name the file `test_something.py`. You must
also make sure the `class` name(s) are different to those in `test/test_terrain.py`.

You can also add extra functionality, but the command-line interface must pass
the tests set.

