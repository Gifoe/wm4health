# Disentangling Coarse and Fine Latent Dynamics for Probabilistic Time Series Forecasting

**This code is the official PyTorch implementation of our IJCAL'2026 Paper: Disentangling Coarse and Fine Latent Dynamics for Probabilistic Time Series Forecasting**


## Setup:

### Environment

This code is developed with Python 3.10 and relies on [PyTorch Lightning](https://github.com/Lightning-AI/lightning). To set up the environment:

```bash
# Create a new conda environment
conda create -n COFE python=3.10
conda activate COFE

# Install required packages
pip install .
```

### Datasets
A compressed file containing the dataset is available in the “dataset” directory.
## Quick Start:

```bash 
bash run.sh
```



## Results

The experimental results for 7 short-term and 8 long-term real-world datasets are as follows:
<div align=center> <img src="figs/short-term.png" width = 95%/> </div>
<div align=center> <img src="figs/long-term.png" width = 95%/> </div>






