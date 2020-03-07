#!/bin/bash

declare -a BEHAVIOR_ARRAY=('crush' 'poke' 'push')
for BEHAVIOR in ${BEHAVIOR_ARRAY[@]};do
  python ./data/make_data.py --behavior $BEHAVIOR
  for VARIABLE in 1
  do
    python ./train.py --output_dir weight_use_haptic_$VARIABLE --use_haptic
    python ./train.py --output_dir weight_baseline_$VARIABLE
    rm -r ../data/CY101NPY
  done
done


