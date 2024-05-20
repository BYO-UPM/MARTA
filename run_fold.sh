#!/bin/bash
conda activate vqvae

# # Loop through 10 different folds
# for FOLD in {0..9}
# do
# echo "Running scripts for fold $FOLD"

# echo "Marta supervised with latent dim 3"
# # Run the first pair of scripts simultaneously on GPU 0 and GPU 1
# python MARTA_Supervised.py --fold $FOLD --gpu 0 --latent_dim 3 --domain_adversarial 1 &
# PID_GPU0=$!  # Capture the process ID of the last background process

# echo "Marta classifier with latent dim 3 and domain adversarial"
# # Run the third pair of scripts simultaneously on GPU 0 and GPU 1

# python MARTA-S_classifier.py --fold $FOLD --gpu 1 --latent_dim 3 --domain_adversarial 0 &
# PID_GPU1=$!
# wait $PID_GPU0 $PID_GPU1

echo "Marta supervised with latent dim 32"
# # Run the second pair of scripts simultaneously on GPU 0 and GPU 1
# python MARTA_Supervised.py  --gpu 0 --latent_dim 32 --domain_adversarial 1 --cross_lingual nv_gita  --fold 2&
# PID_GPU0=$!
# python MARTA_Supervised.py  --gpu 1 --latent_dim 32 --domain_adversarial 0 --cross_lingual nv_gita --fold 2&
# PID_GPU1=$!
# wait $PID_GPU0 $PID_GPU1

echo "Marta classifier with latent dim 32 and domain adversarial"
#Run the fourth pair of scripts simultaneously on GPU 0 and GPU 1
python MARTA-S_classifier.py  --gpu 0 --latent_dim 32 --domain_adversarial 1 --cross_lingual nv_gita --fold 2&
PID_GPU0=$!
wait $PID_GPU0
# python MARTA-S_classifier.py --gpu 1 --latent_dim 32 --domain_adversarial 0 --cross_lingual nv_gita --fold 2&
# PID_GPU1=$!
# wait $PID_GPU0 $PID_GPU1

# echo "Marta supervised with latent dim 32"
# # Run the second pair of scripts simultaneously on GPU 0 and GPU 1
# python MARTA_Supervised.py  --gpu 0 --latent_dim 32 --domain_adversarial 1 --cross_lingual gita_nv --fold 2&
# PID_GPU0=$!
# python MARTA_Supervised.py  --gpu 1 --latent_dim 32 --domain_adversarial 0 --cross_lingual gita_nv --fold 2&
# PID_GPU1=$!
# wait $PID_GPU0 $PID_GPU1

# echo "Marta classifier with latent dim 32 and domain adversarial"
# # Run the fourth pair of scripts simultaneously on GPU 0 and GPU 1
# python MARTA-S_classifier.py  --gpu 0 --latent_dim 32 --domain_adversarial 1 --cross_lingual gita_nv --fold 2&
# PID_GPU0=$!
# python MARTA-S_classifier.py  --gpu 1 --latent_dim 32 --domain_adversarial 0 --cross_lingual gita_nv --fold 2&
# PID_GPU1=$!
# wait $PID_GPU0 #$PID_GPU1


# echo "Finished scripts for fold $FOLD"
# done

echo "All tasks for all folds are done!"
