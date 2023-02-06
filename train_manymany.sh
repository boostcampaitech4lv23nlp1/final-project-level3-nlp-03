#!/bin/bash
CONFIGS=("baseline" "baseline_tmp")

for (( i=0; i<2; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done