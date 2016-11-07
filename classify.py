#!/usr/bin/env python
import os
import subprocess
import sys

import psycopg2
from osgeo import gdal

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

# In-memory path for manipulating rasters
vsipath = "/vsimem/from_postgis"

# pg cursor
cur = None

def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)

    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])

    return target_ds

def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize the vectors in the given directory in a single image."""
    labeled_pixels = np.ones((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels

def classify(training_labels, training_samples):
    """Create numpy classifier"""
    classifier = RandomForestClassifier(n_jobs=-1)

    is_train = np.nonzero(training_labels)
    training_labels = training_labels[is_train]
    training_samples = training_samples[is_train]

    classifier.fit(training_samples, training_labels)

    return classifier

def prepare_data(lower_bound, num_train):
    """Fetch data from PostGIS database and return a label and sample array"""
    for i in range(lower_bound, lower_bound + num_train):

        # Select the ith raster
        cur.execute("SELECT ST_AsGDALRaster(rast, 'GTiff') FROM eightbands WHERE rid = %s;", [i]);
        res = cur.fetchone()
        gdal.FileFromMemBuffer(vsipath, bytes(res[0]))
        raster_dataset = gdal.Open(vsipath)

        # Get transforms
        geo_transform = raster_dataset.GetGeoTransform()
        proj = raster_dataset.GetProjectionRef()

        # Read data into array
        raster_data = []
        for b in range(1, raster_dataset.RasterCount+1):
            band = raster_dataset.GetRasterBand(b)
            raster_data.append(band.ReadAsArray())

        raster_data = np.dstack(raster_data)

        # Add to sample array
        try:
            training_samples
        except NameError:
            training_samples = raster_data
        else:
            training_samples = np.concatenate((training_samples, raster_data))

        rows, cols, n_bands = raster_data.shape

        # Next, find all buildings within this raster's boundary and output
        # a shapefile
        devnull = open(os.devnull, 'w') # Supress output
        print("Reading raster %d..." % i)
        subprocess.call(["pgsql2shp", "-f", "tmp/test.shp", "-h", "localhost",
            "-u", "joefarned", "spacenet", "SELECT * FROM buildings WHERE \
            ST_ContainsProperly (ST_Polygon((SELECT rast FROM eightbands  \
            WHERE rid = {i}), 1), geom)".format(i=i)], stdout=devnull)

        # Get the training data
        train_data_path = os.path.dirname(os.path.realpath(__file__)) + "/tmp"
        files = [f for f in os.listdir(train_data_path) if f.endswith('.shp')]
        classes = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(train_data_path, f)
                      for f in files if f.endswith('.shp')]
        train_data = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)

        # Add to label array
        try:
            training_labels
        except NameError:
            training_labels = train_data
        else:
            training_labels = np.concatenate((training_labels, train_data))

    # Return tuple with data for classification
    return (training_labels, training_samples)

def test_accuracy(classifier, lower_bound, num_test):
    """Test the model's accuracy"""
    print("Testing model accuracy...")
    (test_labels, test_samples) = prepare_data(lower_bound, num_test)

    rows, cols, n_bands = test_samples.shape

    n_samples = rows*cols
    flat_pixels = test_samples.reshape((n_samples, n_bands))
    result = classifier.predict(flat_pixels)
    classification = result.reshape((rows, cols))

    # Calculate MCE
    misclassification = np.sum(np.absolute(classification - test_labels))
    num_predictions = rows * cols
    mce = (num_predictions - misclassification) / num_predictions
    print("Model accuracy was %f" % mce)

    f = plt.figure()
    f.add_subplot(1, 2, 2)
    r = test_samples[:,:,3]
    g = test_samples[:,:,2]
    b = test_samples[:,:,1]
    rgb = np.dstack([r,g,b])
    f.add_subplot(1, 2, 1)
    plt.imshow(rgb/255)
    f.add_subplot(1, 2, 2)
    plt.imshow(classification)
    plt.show()

def main():
    """Training prodedure"""
    global cur

    # Create connection to database
    conn = psycopg2.connect(dbname="spacenet", port="5432", user="joefarned",
                            host="localhost")
    cur = conn.cursor()

    # Prepare the data, run classifier
    prepared_data = prepare_data(int(sys.argv[1]), int(sys.argv[2]))
    classifier = classify(prepared_data[0], prepared_data[1])
    test_accuracy(classifier, int(sys.argv[3]), int(sys.argv[4]))

    # Close out connection
    cur.close()
    conn.close()

if __name__ == "__main__": main()
