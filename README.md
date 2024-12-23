Machine learning for drug aqueous solubility prediction
=

This repository contains the code and data used in "Machine learns aqueous solubility of drugs". The codes for descriptor generation, hyperparameter optimization, model training, and model testing are included for both regression and classification tasks of solubility prediction. Four base learners (RF, XGB, LightGBM, and SVM) as well as their Stackings, and two DL adcanced models (Transfoprmer-CNN and GNN) are presented.

# Requirements for Environments
To run the scripts provided in this repository, you'll need the following Python libraries:

* Python 3.8+
* scikit-learn >= 1.0.2
* pandas >= 1.4.3
* numpy >= 1.24.4
* matplotlib >= 3.5.2
* imbalanced_learn >= 0.12.2
* rdkit >= 2023.9.6
* sklearn_relief >= 1.0.0b2
* scipy >= 1.10.1
