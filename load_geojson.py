#!/usr/bin/env python

import os, sys
import psycopg2
import json

# TODO: user-ize db + table

""" Handle GeoJSON from a single file single file """
def add_file(f):
    with open(sys.argv[1] + '/' + f) as data_file:
        data = json.load(data_file)

    for i in xrange(len(data["features"])):
        for j in xrange(len(data["features"][i]["geometry"]["coordinates"][0])):
            data["features"][i]["geometry"]["coordinates"][0][j].pop(2)

        # Add into table
        data["features"][i]["geometry"]["crs"] = {"type":"name","properties":{"name":"EPSG:4326"}}
        if (data["features"][i]["geometry"]["type"] == "Polygon"):

            try:
                cur.execute("INSERT INTO buildings (geom) VALUES (ST_GeomFromGeoJSON(%s))",
                            [json.dumps(data["features"][i]["geometry"])])

            # Exception occuring here
            except psycopg2.DataError:
                print(f)
                print("error")


    conn.commit()

""" Create table and loop over files """
conn = psycopg2.connect(dbname="spacenet", port="5432", user="joefarned",
                        host="localhost")
cur = conn.cursor()

# Create the table for the building locations
cur.execute("CREATE TABLE buildings (id serial PRIMARY KEY);")
conn.commit()

# Add geom column
cur.execute("SELECT AddGeometryColumn ('buildings', 'geom', 4326, 'POLYGON', 2);")
conn.commit()

# Loop over all GeoJSON files
for f in os.listdir(sys.argv[1]):
    add_file(f)

cur.close()
conn.close()
