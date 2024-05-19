# Run MARTA-unsupervised secuentially with all gradients manipulation methods.
python MARTA_unsupervised.py --method pcgrad
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/GMVAE_cnn_best_model_2d.pt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/GMVAE_cnn_best_model_2d_pcgrad.pt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/logs.txt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/logs_pcgrad.txt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_cos.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_cos_pcgrad.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_loss.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_loss_pcgrad.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_mag.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_mag_pcgrad.csv
python MARTA_unsupervised.py --method mgd
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/GMVAE_cnn_best_model_2d.pt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/GMVAE_cnn_best_model_2d_mgd.pt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/logs.txt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/logs_mgd.txt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_cos.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_cos_mgd.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_loss.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_loss_mgd.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_mag.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_mag_mgd.csv
python MARTA_unsupervised.py --method graddrop
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/GMVAE_cnn_best_model_2d.pt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/GMVAE_cnn_best_model_2d_graddrop.pt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/logs.txt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/logs_graddrop.txt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_cos.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_cos_graddrop.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_loss.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_loss_graddrop.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_mag.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_mag_graddrop.csv
python MARTA_unsupervised.py --method cagrad
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/GMVAE_cnn_best_model_2d.pt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/GMVAE_cnn_best_model_2d_cagrad.pt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/logs.txt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/logs_cagrad.txt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_cos.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_cos_cagrad.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_loss.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_loss_cagrad.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_mag.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_mag_cagrad.csv
python MARTA_unsupervised.py --method sumloss
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/GMVAE_cnn_best_model_2d.pt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/GMVAE_cnn_best_model_2d_sumloss.pt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/spectrograms/manner_gmvae_alb_neurovoz_32unsupervised32d_final_model2/logs.txt /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/logs_sumloss.txt
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_cos.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_cos_sumloss.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_loss.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_loss_sumloss.csv
mv /home/iprieto@gaps_domain.ssr.upm.es/MTL/MARTA/local_results/trace_mag.csv /home/iprieto@gaps_domain.ssr.upm.es/MTL/models/trace_mag_sumloss.csv