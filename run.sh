#!/bin/bash

rm -r ../data/CY101NPY
python ./data/make_data.py
for VARIABLE in 1 2 3 4 5
do
  python ./train.py --output_dir weight_use_haptic_$VARIABLE --use_haptic --use_behavior
  python ./train.py --output_dir weight_baseline_$VARIABLE
  rm -r ../data/CY101NPY
  python ./data/make_data.py
done

