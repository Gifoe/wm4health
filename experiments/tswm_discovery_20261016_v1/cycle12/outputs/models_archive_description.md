# QG Closure Trained Models Archive

**Archive filename:** models_archive.tar.gz

## Description
This archive contains the full set of trained neural network models produced by the QG Closure pipeline for data-driven closures in quasi-geostrophic turbulence modeling. The models were trained using the scripts and configuration described in the QG Closure repository (see README for details).

## Contents
- Stochastic and deterministic neural network models for all window lengths
- Model scaler files (e.g., *scaler.pkl*)
- Directory structure as output by the pipeline (see README Project Structure)

## Usage
Extract the archive in your workspace:

```bash
tar -xzf models_archive.tar.gz
```

The extracted `models/` directory can be used directly with the QG Closure codebase for evaluation, further training, or analysis.

## Requirements
- The models require the same software environment as described in the QG Closure repository, including JAX with GPU support and the listed Python dependencies.
