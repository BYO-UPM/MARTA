#!/bin/bash
conda activate vqvae

# # Loop through 10 different folds
for FOLD in {0..9}
do
echo "Running scripts for fold $FOLD"

# Run the first pair of scripts simultaneously on GPU 0 and GPU 1
python MARTA_Supervised.py --fold $FOLD --gpu 0 --latent_dim 64 --domain_adversarial 0 --cross_lingual testing_neurovoz &
PID_GPU1=$! 
python MARTA_Supervised.py --fold $FOLD --gpu 1 --latent_dim 64 --domain_adversarial 0 --cross_lingual testing_gita &
PID_GPU2=$!
wait $PID_GPU1
wait $PID_GPU2



python MARTA-S_classifier.py --fold $FOLD --gpu 0 --latent_dim 64 --domain_adversarial 0 --cross_lingual testing_neurovoz &
PID_GPU1=$!  
python MARTA-S_classifier.py --fold $FOLD --gpu 1 --latent_dim 64 --domain_adversarial 0 --cross_lingual testing_gita &
PID_GPU2=$!
wait $PID_GPU1
wait $PID_GPU2

echo "Finished scripts for fold $FOLD AND LATENT DIM 3"


done

echo "All tasks for all folds are done!"
