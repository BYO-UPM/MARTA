#!/bin/bash

FOLD=$1
GPU=$2

# Run the supervised script
python cnns_spectrograms_GMVAE_speech_therapist_supervised.py --fold $FOLD --gpu $GPU

# Run the classifier script
python cnns_spectrograms_GMVAE_speech_therapist_classifier.py --fold $FOLD --gpu $GPU