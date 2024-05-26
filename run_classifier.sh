#!/bin/bash

# Create directory to save tagged models
mkdir ../models_sup

# Define methods to test
methods=("mgd" "cagrad" "graddrop")

for m in ${methods[@]}; do
    for f in $(seq 1 10); do
        echo "method: $m - fold: $f"
        model=../models_sup/GMVAE_cnn_best_model_2d_${m}_${f}.pt
        python MARTA_Supervised.py --gpu 0 --fold "$f" --method "$m" --model "$model"
    done
done
