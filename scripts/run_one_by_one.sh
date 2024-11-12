#!/bin/bash
# echo "Marta classifier with latent dim 32 and domain adversarial"
# #Run the fourth pair of scripts simultaneously on GPU 0 and GPU 1
# # python MARTA-S_classifier.py  --gpu 0 --latent_dim 32 --domain_adversarial 1 --cross_lingual nv_gita --fold 2&
# # PID_GPU0=$!
# # wait $PID_GPU0

# echo "Marta supervised with latent dim 32"
# # # Run the second pair of scripts simultaneously on GPU 0 and GPU 1

# python MARTA_Supervised.py  --gpu 0 --latent_dim 32 --domain_adversarial 0 --cross_lingual nv_gita --fold 2&
# PID_GPU0=$!
# wait $PID_GPU0

# #Run the fourth pair of scripts simultaneously on GPU 0 and GPU 1
# python MARTA-S_classifier.py  --gpu 0 --latent_dim 32 --domain_adversarial 0 --cross_lingual nv_gita --fold 2&
# PID_GPU0=$!
# wait $PID_GPU0


# python MARTA_Supervised.py  --gpu 0 --latent_dim 32 --domain_adversarial 0 --cross_lingual gita_nv --fold 2&
# PID_GPU0=$!
# python MARTA_Supervised.py  --gpu 1 --latent_dim 32 --domain_adversarial 1 --cross_lingual gita_nv --fold 2&
# PID_GPU1=$!
# wait $PID_GPU0 $PID_GPU1


python MARTA_Supervised.py  --gpu 1 --latent_dim 32 --domain_adversarial 0 --cross_lingual testing_neurovoz --fold 2&
PID_GPU0=$!
wait $PID_GPU0 
python MARTA-S_classifier.py  --gpu 1 --latent_dim 32 --domain_adversarial 0 --cross_lingual testing_neurovoz --fold 2&
