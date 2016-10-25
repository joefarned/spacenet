#!/bin/bash

DBNAME=$1
TABLENAME=$2
IMAGEPATH=$3

mkdir -p $IMAGEPATH/sql

# TODO: parallelize
for f in $IMAGEPATH/*.tif
do
    raster2pgsql -a -s 4326 $f public.$TABLENAME > "$IMAGEPATH/sql/$(basename $f)".sql
done

psql -d $DBNAME -c "CREATE TABLE "public"."$TABLENAME" ("rid" serial PRIMARY KEY,"rast" raster);"
psql -d $DBNAME -c "SELECT AddRasterConstraints('public','eightbands','rast',TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE);"

# TODO: parallelize
for f in $IMAGEPATH/sql/*.sql
do
    psql -d $DBNAME -f $f
done
