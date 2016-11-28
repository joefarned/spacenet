## Spacenet Classifier
![](img.png)

## Summary
This application processes Spacenet images of Rio de Janiero, and classifies
whether or not there are buildings present in image segments. Average
classification success rate is 82%.

## Description
GeoTiffs (images) and GeoJson (labels) are loaded into a PostGIS database.
To build the model, the data is read out of the database and transformed into
numpy arrays using GDAL. A Random Forest classifier is run on the data. Finally,
the testing set data is evaluated using the model to obtain accuracy.

## Usage
'''
./load_geotiffs.sh
python load_geojson.py
python classify.py [train_index] [num_train] [test_index] [num_test]
'''
