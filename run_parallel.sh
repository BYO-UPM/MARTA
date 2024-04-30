# Adjusted loop for distributing folds 0 to 9 across GPUs 0 and 1
for FOLD in `seq 0 10`; do
    # Assign GPU 0 for even folds and GPU 1 for odd folds
    GPU_ID=$((FOLD % 2))

    python MARTA_Supervised.py --fold $FOLD --gpu $GPU --latent_dim 3 --domain_adversarial 1 &
    python MARTA_Supervised.py --fold $FOLD --gpu $GPU --latent_dim 3 --domain_adversarial 0

    # Wait until both processed have finished
    wait 
done

# Wait for the last set of processes to finish
wait
