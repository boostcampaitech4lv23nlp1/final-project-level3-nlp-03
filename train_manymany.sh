#!/bin/bash
CONFIGS=("baseline_boundary02" "baseline_boundary03" "baseline_boundary04" "baseline_boundary05" "baseline_boundary06" "baseline_boundary07" "baseline_boundary08")

for (( i=0; i<7; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done