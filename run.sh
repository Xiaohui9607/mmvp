#!/bin/bash
#
#rm -r ../data/CY101NPY
#python ./data/make_data.py
#for VARIABLE in 1 2 3 4 5
#do
#  python ./train.py --output_dir weight_use_haptic_$VARIABLE --use_haptic --use_behavior --use_audio
#  python ./train.py --output_dir weight_baseline_$VARIABLE
#  rm -r ../data/CY101NPY
#  python ./data/make_data.py
declare -a BEHAVIOR_ARRAY=('crush' 'poke' 'push')
for BEHAVIOR in ${BEHAVIOR_ARRAY[@]};do
  python ./data/make_data.py --behavior $BEHAVIOR
  for VARIABLE in 1
  do
    python ./train.py --output_dir weight_use_haptic_$VARIABLE_$BEHAVIOR --use_haptic --use_behavior --use_audio
    python ./train.py --output_dir weight_baseline_$VARIABLE_$BEHAVIOR
    rm -r ../data/CY101NPY
  done
done


