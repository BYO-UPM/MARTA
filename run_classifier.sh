#!/bin/bash

# Create directory to save tagged models
mkdir ../res_acc

# Define methods to test
methods=("mgd" "pcgrad" "sumloss")

for m in ${methods[@]}; do
    for f in $(seq 0 10); do
        echo "method: $m - fold: $f"
        python MARTA-S_classifier.py --gpu 0 --fold "$f" --method "$m"
        mv local_results/spectrograms/manner_gmvae_alb_neurovoz_32final_model_classifier_LATENTSPACE+manner_MLP_fold${f}/log.txt ../res_acc/log_classifier_${m}_${f}.txt
        mv local_results/spectrograms/manner_gmvae_alb_neurovoz_32final_model_classifier_LATENTSPACE+manner_MLP_fold${f}/best_threshold.txt ../res_acc/best_threshold_${m}_${f}.txt
        mv local_results/spectrograms/manner_gmvae_alb_neurovoz_32final_model_classifier_LATENTSPACE+manner_MLP_fold${f}/GMVAE_cnn_best_model_2d.pt ../res_acc/GMVAE_cnn_best_model_2d_${m}_${f}.pt
    done
done
