# DL_PyTorch_Base_Modules

This repository contains the base modules for experimenting with Deep Learning models according to the applications addressed in the BYO-UPM group. It is organized as follows:

- models
    - Base models to build on top of
    - Fine tuning architectures
- losses
    - Contrastive-Learning losses
        - SlimCL
        - GE2E
- data_loaders
    - data loaders for image and matrices in .txt files
    - data augmentation techniques tailored for spectrograms, FRPs
- training
    - training schemes for pre-training and fine tunining models
- validation
    - structured estimation, saving and displaying of performance measures and figures of merit.
    - validation methodologies: cross-validation, uncertainty estimation.
- Experiments
    - Particular experiment configurations


main: file to launch the experiment

Directories for saving check point models, data and results are keeping local.

# TODO list

- [ ] Supervise by vowels