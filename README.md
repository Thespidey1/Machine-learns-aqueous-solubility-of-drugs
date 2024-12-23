# Machine learning for drug aqueous solubility prediction

This repository contains the code and data used in "Machine learns aqueous solubility of drugs". The codes for descriptor generation, hyperparameter optimization, model training, and model testing are included for both regression and classification tasks of solubility prediction. Four base learners (RF, XGB, LightGBM, and SVM) as well as their Stackings, and two DL adcanced models (Transfoprmer-CNN and GNN) are presented.

## Requirements for Environments
To run the scripts provided in this repository, you'll need the following Python libraries:

* Python 3.11+
* scikit-learn >= 1.3.0
* scikit-plot >= 0.3.7
* pandas >= 2.0.3
* numpy >= 1.24.3
* matplotlib >= 3.7.2
* imbalanced_learn >= 0.12.2
* rdkit >= 2023.3.3
* scipy >= 1.11.1
* torch >= 2.2.1+cu121
* lightgbm >= 4.5.0
* tqdm >= 4.65.0
* networkx >=3.1  
* PyYAML >=6.0.1  
* torch_geometric >=2.1.0  
* torch_scatter >=2.0.9

To install required packages, use the command: `pip install <package_name>==<version_number>`. For example, `pip install imbalanced_learn == 0.12.2`

## Usage
First, download the entire repository as zip file and unzip it. The datasets and corresponding descriptors are provided in `.xlsx` format. To train and test the DL models, just run the `CV.py` file. If you want to reduce training time, just specify a trained weight in `model_weight` directory. To train and test statistical ML moldes and their Stakings, run the `.py` file with the corresponding name. The `opt.py` files in each directory are for hyperparameter optimization. `Metrics_compute_class.py` files are used to generate metrics and corresponding figures. Please note that `Metrics_compute_class.py` should run after training process, in which tables will be generated about training and testing results.
