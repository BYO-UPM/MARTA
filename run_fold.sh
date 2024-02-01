#!/bin/bash

FOLD=$1
GPU=$2

# Run the supervised script
# python SThVAE_supervised.py --fold $FOLD --gpu $GPU

# Run the classifier script
python SThVAE_classifier.py --fold $FOLD --gpu $GPU