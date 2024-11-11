#!/bin/bash
conda activate vqvae

# # Loop through 10 different folds
for FOLD in {0..9}
do
echo "Running scripts for fold $FOLD"

python MARTA-S_classifier_whisper.py --fold $FOLD --gpu 1 --latent_dim 64 --domain_adversarial 1 --cross_lingual multilingual
PID_GPU1=$! 
wait $PID_GPU1

echo "Finished scripts for fold $FOLD AND LATENT DIM 64"


done

echo "All tasks for all folds are done!"
