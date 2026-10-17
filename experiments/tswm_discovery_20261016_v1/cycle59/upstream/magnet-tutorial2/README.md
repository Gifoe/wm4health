# Soft Magnetic Materials Models with ODEs

## Summary

This repository contains the **slides** and **code** related to the following **webinar**:
* **Soft Magnetic Materials and Ordinary Differential Equations**
* **From Linear Circuits to Neural Network Models**
* **IEEE MagNet Challenge Webinar - May 23 2025**
* **Thomas Guillod - Dartmouth College**

This webinar focuses on **ordinary differential equation models for soft-magnetic materials**:
* Using **equation-based models** (linear and nonlinear circuits)
* Using **neural ordinary differential equation** models

The **Python code** has the following **features**:
* Training and inference of ODE models
* Management of the dataset with dataframes
* Various metrics and plotting capabilities
* Using JAX for computations (does not require a GPU)
* Using Diffrax for solving the ODEs (with adjoints)

Various **optimizers** can be used to **train** the models:
* Scipy / latin hypercube sampler
* Scipy / differential evolution
* JAX / Optimistix / minimize solver
* JAX / Optimistix / least-square
* JAX / Optax / gradient descent

## Repository Description

* Main Files
  * [slides.pdf](slides.pdf) - Slides of the webinar (CC BY-ND 4.0)
  * [requirements.txt](requirements.txt) - List of the used Python packages.
* Python Files
  * [run_1_dataset.py](run_1_dataset.py) - Parse the CSV files into a DataFrame.
  * [run_2_eqn_train.py](run_2_eqn_train.py) - Train an equation-based model.
  * [run_3_eqn_infer.py](run_3_eqn_infer.py) - Inference of an equation-based model.
  * [run_4_ann_train.py](run_4_ann_train.py) - Train a neural network-based model.
  * [run_5_ann_infer.py](run_5_ann_infer.py) - Inference of a neural network-based model.
  * [model_eqn.py](model_eqn.py) - Specifications of the equation-based models.
  * [model_ann.py](model_ann.py) - Specifications of the neural network-based models.
* Folders and Python Packages
  * [data](data) - Folder containing the datasets and trained models.
  * [odemodel](odemodel) - Python package with the definition of the models.
  * [odesolver](odesolver) - Python package with the training and inference code.

## Disclaimers

* The goal of this code is to demonstrate basic ODE models.
* The implementation is neither comprehensive nor optimized.
* The dataset provided in the repository is very small.
* The dataset is extracted from the MagNetX dataset.

## Compatibility

* Tested on Linux x86/64.
* Tested with Python 3.12.3.
* Package list in `requirements.txt`.

## Author

* Name: **Thomas Guillod**
* Affiliation: Dartmouth College
* Email: guillod@otvam.ch
* Website: https://otvam.ch

## Credits

This research was done at **Dartmouth College** by the research group of **Prof. Sullivan**:

* Dartmouth College, NH, USA: https://dartmouth.edu
* Dartmouth Engineering: https://engineering.dartmouth.edu

## Copyright

(c) 2024-2025 / Thomas Guillod / Dartmouth College

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
