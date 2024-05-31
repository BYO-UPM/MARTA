#!/bin/bash

# Create directory to save tagged models
mkdir ../res_acc

# Define methods to test
methods=("cagrad" "graddrop" "mgd" "pcgrad" "sumloss" "weightsum")

for m in ${methods[@]}; do
    for f in $(seq 0 10); do
        echo "method: $m - fold: $f"
        model=../models_sup/GMVAE_cnn_best_model_2d_${m}_${f}.pt
        python MARTA-S_classifier.py --gpu 0 --fold "$f" --method "$m" --model "$model"
        mv local_results/spectrograms/manner_gmvae_alb_neurovoz_32final_model_classifier_LATENTSPACE+manner_MLP_fold${f}/log.txt ../res_acc/log_classifier_${m}_${f}.txt
    done
done
