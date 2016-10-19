#!/usr/bin/env python

import os
import psycopg2
from osgeo import gdal

vsipath = '/vsimem/test.tiff'


conn = psycopg2.connect(dbname="spacenet", port="5432", user="joefarned",
                        host="localhost")
cur = conn.cursor()
cur.execute("SELECT ST_AsGDALRaster(rast, 'GTiff') FROM eightband;")

tile = cur.fetchone()[0]
vsipath = '/vsimem/from_postgis'
gdal.FileFromMemBuffer(vsipath, bytes(tile))
ds = gdal.Open(vsipath)
array = ds.GetRasterBand(1).ReadAsArray()

print(array)

gdal.Unlink(vsipath)



cur.close()
conn.close()
