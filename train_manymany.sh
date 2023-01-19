#!/bin/bash
CONFIGS=("DeBERTa_8_lr" "DeBERTa_9_lr")

for (( i=0; i<2; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done