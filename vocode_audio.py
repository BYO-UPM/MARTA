"""
This script will load the data used for the training of the classifier GMVAE. And we will test how well the reconstruction is done. To do so, we will use the HiFIGAN.
First, we will vocode the audio using the WaveGlow vocoder. Then, we will use the GMVAE to reconstruct the audio. Finally, we will compare the original audio with the reconstructed audio."""


from models.pt_models import SpeechTherapist
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
from hifi_gan.models import Generator
from hifi_gan.env import AttrDict
from hifi_gan import meldataset
import json
import torch

# Select the free GPU if there is one available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device being used:", device)


hyperparams = {
    "frame_size_ms": 0.400,  # 400ms
    "n_plps": 0,
    "n_mfccs": 0,
    "spectrogram_win_size": 0.023,  # 23ms as it is recommended in the librosa library for speech recognition
    "material": "MANNER",
    "hop_size_percent": 0.5,
    "spectrogram": True,
    "wandb_flag": False,
    "epochs": 500,
    "batch_size": 128,
    "lr": 1e-3,
    "latent_dim": 32,
    "hidden_dims_enc": [64, 1024, 64],
    "hidden_dims_gmvae": [256],
    "weights": [
        1,  # w1 is rec loss,
        1,  # w2 is gaussian kl loss,
        1,  # w3 is categorical kl loss,
        10,  # w5 is metric loss
    ],
    "supervised": True,
    "classifier": "cnn",  # "cnn" or "mlp"
    "n_gaussians": 16,  # 2 per manner class
    "semisupervised": False,
    "train": True,
    "train_albayzin": True,  # If True, only albayzin+neuro is used to train. If False only neuro are used for training
}

print("Reading data...")

print("Reading train, val and test loaders from local_results/...")
train_loader = torch.load(
    "local_results/train_loader0.4spec_winsize_0.023hopsize_0.5.pt"
)
val_loader = torch.load("local_results/val_loader0.4spec_winsize_0.023hopsize_0.5.pt")
test_loader = torch.load("local_results/test_loader0.4spec_winsize_0.023hopsize_0.5.pt")
test_data = torch.load("local_results/test_data0.4spec_winsize_0.023hopsize_0.5.pt")


model = SpeechTherapist(
    x_dim=train_loader.dataset[0][0].shape,
    z_dim=hyperparams["latent_dim"],
    n_gaussians=hyperparams["n_gaussians"],
    n_manner=8,
    hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
    hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
    classifier=hyperparams["classifier"],
    weights=hyperparams["weights"],
    device=device,
)

name = "local_results/spectrograms/manner_gmvae_alb_neurovoz_32final_modeltesting_hifigan_80channels/GMVAE_cnn_best_model_2d.pt"
tmp = torch.load(name)
model.load_state_dict(tmp["model_state_dict"])


# Load the HiFiGAN generator
checkpoint_file = "hifi_gan/generator_v3"
config_file = "hifi_gan/config.json"

with open(config_file) as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)
torch.manual_seed(h.seed)
torch.cuda.manual_seed(h.seed)

generator = Generator(h).to(device)

print("Loading '{}'".format(checkpoint_file))
checkpoint_dict = torch.load(checkpoint_file, map_location=device)
print("Complete.")
generator.load_state_dict(checkpoint_dict["generator"])

generator.eval()
generator.remove_weight_norm()


# Get 10 spectrograms from the test set randomly
import random
import numpy as np

idx_list = random.sample(range(len(test_data)), 10)
spectrograms = np.stack([test_data["spectrogram"].iloc[idx] for idx in idx_list])
# Convert to torch tensor and move to GPU
spectrograms = torch.tensor(spectrograms).to(device)

# Get the corresponding audio by using the HiFiGAN
audio = generator(spectrograms).squeeze(1)
# Get the real audio
real_audio = np.stack([test_data["signal_framed"].iloc[idx] for idx in idx_list])
# Real audio processed full by higan
wav = torch.FloatTensor(real_audio).to(device)
spect = meldataset.mel_spectrogram(
    wav, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax
)

# Play both audios and compare in the noteook
import IPython.display as ipd

ipd.Audio(real_audio[0], rate=16000)
ipd.Audio(audio[0].detach().cpu().numpy(), rate=16000)

# Get the reconstructed spectrogram
