from models.pt_models import MARTA
from training.pt_training import MARTA_trainer, MARTA_tester
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
from utils.definitions import *
import torch
import wandb
import sys
import os
import argparse
from utils.utils import make_balanced_sampler, augment_data, stratify_dataset
import time


def main(args, hyperparams):

    # Select device.
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print('Device being used:', device)

    # Prepare directory to save local results.
    hyperparams["path_to_save"] = \
        f'local_results/spectrograms/grb_gmvae_neurovoz_{hyperparams["latent_dim"]}_multiregressor_fold{hyperparams["fold"]}'
    if not os.path.exists(hyperparams["path_to_save"]):
        os.makedirs(hyperparams["path_to_save"])

    # Prepare logs dumping to file.
    old_stdout = sys.stdout
    log_file = open(f'{hyperparams["path_to_save"]}/log.txt', 'w')
    sys.stdout = log_file
    
    # Generate MARTA data partitions if configured.
    if hyperparams['new_data_partition']:
        print('Reading data...')
        dataset = Dataset_AudioFeatures( 'labeled/NeuroVoz', hyperparams)
        print('Creating train, val and test loaders...')
        _, _, _, _, _, _ = dataset.get_dataloaders(
            train_albayzin=hyperparams["train_albayzin"], supervised=hyperparams["supervised"])

    # Read data from GRB-transformed data partitions.
    print('Reading train, val and test loaders from local_results/...')
    name_core_1 = f'_supervised_True_frame_size_0.4spec_winsize_{hyperparams["spectrogram_win_size"]}'
    name_core_2 = f'hopsize_0.5fold{hyperparams["fold"]}.pt'
    train_loader = torch.load(f'{PARTITIONS_DATA_LOCAL}train_loader{name_core_1}{name_core_2}')
    val_loader   = torch.load(f'{PARTITIONS_DATA_LOCAL}val_loader{name_core_1}{name_core_2}')
    test_loader  = torch.load(f'{PARTITIONS_DATA_LOCAL}test_loader{name_core_1}{name_core_2}')
    test_data    = torch.load(f'{PARTITIONS_DATA_LOCAL}test_data{name_core_1}{name_core_2}')

    # Create GMVAE model.
    print('Defining models...')
    model = MARTA(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams['latent_dim'],
        n_gaussians=hyperparams['n_gaussians'],
        n_manner=16,
        hidden_dims_spectrogram=hyperparams['hidden_dims_enc'],
        hidden_dims_gmvae=hyperparams['hidden_dims_gmvae'],
        classifier=hyperparams['classifier_type'],
        grb_enable=hyperparams['grb_enable'],
        weights=hyperparams['weights'],
        device=device,
    )

    # Tailor MARTA's model for a GRB MTL task, with pre-trained GMVAE 
    # parameters and trainable GRB MTL parameters.
    if hyperparams['train']:

        # Load the trained model with the best MARTA's GMVAE parameters.
        name = (
            'local_results/models_z/GMVAE_cnn_best_model_2d' + '_'
            + 'graddrop' + '_' # FIXME: allow to choose the GMVAE optimization strategy
            + str(hyperparams["fold"])
            + '.pt'
        )
        tmp = torch.load(name, map_location='cuda:0')

        # Append GRB MTL parameters to MARTA's GMVAE parameters.
        model_state_dict = model.state_dict()
        for key in model_state_dict.keys():
            if key in tmp['model_state_dict']:
                model_state_dict[key] = tmp['model_state_dict'][key]
        model.load_state_dict(model_state_dict)

        # Make GRB MTL parameters the only ones trainable.
        for param in model.parameters():
            param.requires_grad = False
        model.multi_task()
        for param in model.mt_grb.parameters():
            param.requires_grad = True
        model.to(device)

        # Train MARTA's model and save the best one.
        print('Training GMVAE...')
        MARTA_trainer(
            model=model,
            trainloader=train_loader,
            validloader=val_loader,
            epochs=hyperparams['epochs'],
            lr=hyperparams['lr'],
            wandb_flag=hyperparams['wandb_flag'],
            path_to_save=hyperparams["path_to_save"],
            supervised=hyperparams["supervised"],
            classifier=hyperparams['classifier'],
            grb_enable=hyperparams['grb_enable'],
            method=hyperparams['method'],
        )

        print('Training finished!')
    else:
        print('Loading model...')

    # Load best MARTA's model.
    name = hyperparams["path_to_save"] + '/GMVAE_cnn_best_model_2d.pt'
    tmp = torch.load(name)
    model.load_state_dict(tmp['model_state_dict'])

    # Test the model
    print('Testing GMVAE...')
    MARTA_tester(
        model=model,
        testloader=test_loader,
        test_data=test_data,
        supervised=False,  # Not implemented yet
        wandb_flag=hyperparams['wandb_flag'],
        path_to_plot=hyperparams["path_to_save"],
        # best_threshold=threshold,
    )
    print('Testing finished!')

    if hyperparams['wandb_flag']:
        wandb.finish()

    sys.stdout = old_stdout
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script configuration')
    parser.add_argument(
        '--fold', type=int, default=1, help='Fold number for the experiment'
    )
    parser.add_argument(
        '--gpu', type=int, default=0, help='GPU number for the experiment'
    )
    parser.add_argument(
        '--method', type=str, default='sumloss', help='Gradients manipulation method'
    )

    args = parser.parse_args()

    ### DEBUG DEGINS
    args.method = 'sumloss'
    args.gpu = 1
    args.fold = 0
    ### DEBUG ENDS

    hyperparams = {
        # ================ Spectrogram parameters ===================
        'spectrogram': True,  # If true, use spectrogram. If false, use plp (In this study we only use spectrograms)
        'frame_size_ms': 0.400,  # Size of each spectrogram frame
        "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
        'hop_size_percent': 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
        # ================ GMVAE parameters ===================
        'epochs': 500,  # Number of epochs to train the model (at maximum, we have early stopping)
        'batch_size': 128,  # Batch size
        'lr': 1e-3,  # Learning rate: we use cosine annealing over ADAM optimizer
        'latent_dim': 32,  # Latent dimension of the z vector (remember it is also the input to the classifier)
        'n_gaussians': 16,  # Number of gaussians in the GMVAE
        'hidden_dims_enc': [
            64,
            1024,
            64,
        ],  # Hidden dimensions of encoder/decoder (from audio framed to spectrogram and viceversa)
        'hidden_dims_gmvae': [256],  # Hidden dimensions of the GMVAE encoder/decoder
        'weights': [  # Weights for the different losses
            1,  # w1 is rec loss,
            1,  # w2 is gaussian kl loss,
            1,  # w3 is categorical kl loss,
            10,  # w5 is metric loss
        ],
        # ================ Classifier parameters ===================
        'classifier_type': 'cnn',  # classifier architecture (cnn or mlp)-.Their dimensions are hard-coded in pt_models.py (we should fix this)
        'classifier': False,  # If true, train the classifier
        "supervised": True,  # It must be true
        # ================ Training parameters ===================
        'method': args.method, 
        'grb_enable': True, # If true, MARTA model is tailored for GRB rating scale prediction.
        'train': True,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
        "train_albayzin": False,  # If true, train with albayzin data. If false, only train with neurovoz data.
        'new_data_partition': False,  # If True, new folds are created. If False, the folds are read from local_results/folds/. IT TAKES A LOT OF TIME TO CREATE THE FOLDS (5-10min aprox).
        "fold": args.fold,  # Which fold to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
        'gpu': args.gpu,  # Which gpu to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
        # ================ UNUSED PARAMETERS (we should fix this) ===================
        # These parameters are not used at all and they are from all versions of the code, we should fix this.
        'material': 'MANNER',  # not used here
        'n_plps': 0,  # Not used here
        'n_mfccs': 0,  # Not used here
        'wandb_flag': False,  # Not used here
        'semisupervised': False,  # Not used here
    }

    main(args, hyperparams)
