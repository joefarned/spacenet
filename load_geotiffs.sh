#!/bin/bash

IMAGEPATH=$1

mkdir -p $IMAGEPATH/sql

for f in $IMAGEPATH/*.tif
do
    raster2pgsql -I -s 4236 $f > "$IMAGEPATH/sql/$(basename $f)".sql
done

for f in $IMAGEPATH/sql/*.sql
do
    psql -d spacenet -f $f
done
