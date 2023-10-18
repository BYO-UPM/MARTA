#!/bin/bash

# Author: Alejandro Guerrero-López
# This Bash script is used to install the necessary packages and set up the environment for using a Python script.
# It performs the following tasks:
# 1. Installs packages via `apt-get` (gcc-multilib) required for the script.
# 2. Extracts and configures the HTK (Hidden Markov Model Toolkit) software.
# 3. Creates a Conda environment for Python packages required by `faseAlign`.
# 4. Sets environment variables for UTF-8 support.
# 5. Activates the Conda environment for `faseAlign`.

# Step 1: Install gcc-multilib
sudo apt-get install gcc-multilib

# Step 2: Extract and configure HTK
tar -xvzf HTK-3.4.1.tar.gz
cd htk
export CPPFLAGS=-UPHNALG
./configure --disable-hlmtools --disable-hslab --without-x
make all
sudo make install

# Verify HTK installation
HVite -V

# Print that the installation is complete
echo "HTK Installation Complete"

# Step 3: Install Python Packages
echo "Installing Python Packages"

# Download the environment.yml file
wget https://raw.githubusercontent.com/EricWilbanks/faseAlign/master/environment.yml

# Create the Conda environment for faseAlign
conda env create -f environment.yml

# Step 4: Export UTF-8 environment variables
echo export LC_ALL=en_US.UTF-8 >> ~/.bashrc
echo export LC_ALL=en_US.UTF-8 >> ~/.profile
echo export LANG=en_US.UTF-8 >> ~/.bashrc
echo export LANG=en_US.UTF-8 >> ~/.profile
echo export LANGUAGE=en_US.UTF-8 >> ~/.bashrc
echo export LANGUAGE=en_US.UTF-8 >> ~/.profile

# Reload the environment variables
source ~/.bashrc

# Print that the installation is complete
echo "Python Packages Installation Complete"

# Step 5: Activate the Conda environment for faseAlign
conda activate fase
