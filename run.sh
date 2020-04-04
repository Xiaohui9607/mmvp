#!/bin/bash

# --------------- verison 1.0 ---------------
#for VARIABLE in 1 2 3 4 5
#do
#  python ./train.py --output_dir weight_use_haptic_$VARIABLE --use_haptic --use_behavior --use_audio
#  python ./train.py --output_dir weight_baseline_$VARIABLE
#done

# --------------- version 2.0 ---------------
#declare -a BEHAVIOR_ARRAY=('crush' 'poke' 'hold' 'lift_slow' 'shake' 'push' 'low_drop' 'tap' 'grasp')
#for BEHAVIOR in ${BEHAVIOR_ARRAY[@]};do
##  python ./data/make_data.py --behavior $BEHAVIOR
#for VARIABLE in 1 2 3 4 5
#do
#    rm -r ../data/CY101NPY
#    python ./data/make_data.py --behavior ${BEHAVIOR}
#    python ./train.py --output_dir weight_use_haptic_audio_${VARIABLE}_${BEHAVIOR} --behavior_layer 1 --use_haptic --use_behavior --use_audio  --aux
#    python ./train.py --output_dir weight_baseline_${VARIABLE}_${BEHAVIOR} --behavior_layer 0 --baseline
#done
#done

# --------------- version 3.0 ---------------
#rm -r ../data/CY101NPY
#python ./data/make_data.py
#VARIABLE = 1
#for VARIABLE in 1 2 3 4 5
#do
#python ./train.py --output_dir weight_use_haptic_1 --use_haptic --use_behavior --aux
#
#python ./train.py --output_dir weight_use_audio_1 --use_behavior --use_audio  --aux
python ./train.py --output_dir weight_use_haptic_audio_1 --use_haptic --use_behavior --use_audio  --aux
python ./train.py --output_dir weight_use_haptic_audio_1 --use_haptic --use_behavior --use_audio  --use_vibro --aux


#python ./train.py --output_dir weight_baseline_1 --baseline

#done
