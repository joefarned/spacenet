#!/usr/bin/env python

import os
import psycopg2
from osgeo import gdal
import numpy as np

vsipath = "/vsimem/from_postgis"

# Create connection to database
conn = psycopg2.connect(dbname="spacenet", port="5432", user="joefarned",
                        host="localhost")
cur = conn.cursor()

for i in range(3, 4):
    cur.execute("SELECT ST_AsGDALRaster(rast, 'GTiff') FROM eightbands WHERE rid = %s;", [i]);
    tile = cur.fetchone()[0]
    gdal.FileFromMemBuffer(vsipath, bytes(tile))
    raster_dataset = gdal.Open(vsipath)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data)
    print(bands_data[0])
    rows, cols, n_bands = bands_data.shape
    print(rows, cols, n_bands)


    cur.execute("SELECT ST_AsGDALRaster(ST_AsRaster(geom, 10, 10, '16BUI'), \
        'GTiff') FROM buildings WHERE ST_ContainsProperly \
        (ST_Polygon((SELECT rast FROM eightbands WHERE rid = %s), 1), \
        geom)", [i]);
    tile = cur.fetchone()[0]
    gdal.FileFromMemBuffer(vsipath, bytes(tile))
    ds = gdal.Open(vsipath, gdal.GA_Update)
    ds.SetGeoTransform(geo_transform)
    ds.SetProjection(proj)

    array = ds.GetRasterBand(1).ReadAsArray()
    print(array)

gdal.Unlink(vsipath)

cur.close()
conn.close()
