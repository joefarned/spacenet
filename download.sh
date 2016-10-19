#!/bin/bash

aws s3api get-object --bucket spacenet-dataset --key manifest.txt \
  --request-payer requester manifest.txt

for f in $(less manifest.txt | grep 'processedData.*\..*') ; do \
  mkdir -p $(dirname ${f/\.\//}) && \
  echo Downloading ${f/\.\//} && \
  aws s3api get-object --bucket spacenet-dataset --key ${f/\.\//} \
    --request-payer requester ${f/\.\//};\
done
