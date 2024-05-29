#!/bin/bash

# Create directory to save tagged models
mkdir ../models_sup

# Define methods to test
methods=("weightsum")

for m in ${methods[@]}; do
    for f in $(seq 0 10); do
        echo "method: $m - fold: $f"
        python MARTA_Supervised.py --gpu 0 --fold "$f" --method "$m"
        mv local_results/spectrograms/manner_gmvae_alb_neurovoz_32supervised90-10-fold${f}/GMVAE_cnn_best_model_2d.pt ../models_sup/GMVAE_cnn_best_model_2d_${m}_${f}.pt
        mv local_results/trace_cos.csv ../models_sup/trace_cos_${m}_${f}.csv
        mv local_results/trace_mag.csv ../models_sup/trace_mag_${m}_${f}.csv
        mv local_results/trace_loss.csv ../models_sup/trace_loss_${m}_${f}.csv
    done
done
