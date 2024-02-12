# MARTA: a model for the automatic phonemic grouping of the parkinsonian speech

This repository contains a series of Python scripts implementing the Manner of ARTiculation Analysis (MARTA) based in a Gaussian Mixture Variational Autoencoder (GMVAE) model for speech feature analysis, specifically manner class articulation, in Parkinson's Disease (PD) research. These scripts focus on different aspects of speech analysis, ranging from unsupervised learning to supervised approaches emphasizing latent space distinctions.

## Overview

The project includes the following key scripts:

1. **MARTA:** Focuses on analyzing manner class articulations of healthy individuals and evaluating latent space cluster distances between healthy subjects from differents datasets, such as Albayzin and NeuroVoz. This is an unsupervised analysis and the script associated to it is __MARTA_unsupervised.py__
2. **MARTA with Supervision:** A supervised variant of MARTA, maximizing distances between clusters in the latent space for different conditions (parkinsonian and nosomorphic speech). This study involves two experiments. The first is a supervised studio of the cluster distances (__MARTA_Supervised.py__). The second is a discriminative power studio based on the manner classes clusters (__MARTA-S_classifier.py__).

Each script is designed to process speech data, train the MARTA model, and analyze the results, providing insights into the potential indicators of Parkinson's Disease in speech patterns.

## Installation

To set up the project using Conda, follow these steps:

```bash
# Clone the repository
git clone https://github.com/BYO-UPM/MARTA.git
cd MARTA

# Create a Conda environment from the .yml file
conda env create -f environment.yml

# Activate the Conda environment
conda activate your-env-name
```

## Usage

Each script can be run independently, depending on the specific analysis you wish to perform. Here are the general steps to follow:

1. **Prepare Your Data:** Ensure your speech data is correctly formatted and stored in the required directories.
2. **Configure Hyperparameters:** Adjust the hyperparameters in each script according to your data and analysis needs.
3. **Run the Script:** Execute the script using a Python interpreter. For example:

   ```bash
   python MARTA_unsupervised.py --fold 0 --gpu 1
   python MARTA_Supervised.py --fold 0 --gpu 1
   python MARTA-S_classifier.py --fold 0 --gpu 1
   ```

4. **Analyze Results:** Check the output files and visualizations generated by the scripts for analysis.

Detailed usage for each script is documented within the script files themselves.

In case you want to train all folds, you can use __run_parallel.sh__ bash script.

## Contributing

Contributions to this project are welcome. Please follow the standard fork-and-pull request workflow on GitHub. If you have any suggestions or improvements, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Please see the `LICENSE` file for more details.

## Contact

For any queries or further assistance, please reach out to [Dr. Alejandro Guerrero-López](mailto:alejandro.guerrero@upm.es).
