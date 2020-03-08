#!/bin/bash
#
rm -r ../data/CY101NPY
python ./data/make_data.py
for VARIABLE in 1 2 3 4 5
do
  python ./train.py --output_dir weight_use_haptic_$VARIABLE --use_haptic --use_behavior --use_audio
  python ./train.py --output_dir weight_baseline_$VARIABLE
  rm -r ../data/CY101NPY
  python ./data/make_data.py
done

#declare -a BEHAVIOR_ARRAY=('crush' 'poke' 'push' 'lift_slow' 'shake' 'push')
##for BEHAVIOR in ${BEHAVIOR_ARRAY[@]};do
##  python ./data/make_data.py --behavior $BEHAVIOR
#for VARIABLE in 1 2 3 4 5
#do
#    python ./train.py --output_dir weight_use_haptic_${VARIABLE}_${BEHAVIOR} --use_haptic --use_behavior --use_audio
#    python ./train.py --output_dir weight_baseline_${VARIABLE}_${BEHAVIOR}
#    rm -r ../data/CY101NPY
#done
##done
