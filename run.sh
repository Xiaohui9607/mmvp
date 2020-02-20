#!/bin/bash

for VARIABLE in 1 2 3 4 5 .. 20
do
  rm -r ../data/CY101NPY
  python ./data/make_data.py
  python ./train.py --output_dir weight_$VARIABLE > log_$VARIABLE
done