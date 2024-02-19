#!/bin/bash

FOLD=$1
GPU=$2
# MASKED=$3

# Run the supervised script
# python MARTA_Supervised.py --fold $FOLD --gpu $GPU

# Run the classifier script
python MARTA-S_classifier.py --fold $FOLD --gpu $GPU #--masked $MASKED