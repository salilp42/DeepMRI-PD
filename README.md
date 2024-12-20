# PD-MRI-Classification

Deep Learning Models for Parkinson's Disease Classification using MRI Data

## Overview

This repository contains implementations of various deep learning models for classifying Parkinson's Disease (PD) using structural MRI data. The project implements and compares three main architectures:

- Convolutional Neural Networks (CNN)
- Graph Convolutional Networks (GCN)
- Convolutional Kernel Analytic Networks (ConvKAN)

## Project Structure

```
PD-MRI-Classification/
├── configs/           # Configuration files for models and experiments
├── docs/             # Documentation and additional resources
├── src/              # Source code
│   ├── data_loaders/ # Data loading and preprocessing
│   ├── models/       # Model architectures
│   ├── utils/        # Utility functions
│   └── experiments/  # Experiment scripts
└── tests/            # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PD-MRI-Classification.git
cd PD-MRI-Classification
```

2. Create a virtual environment (Python 3.8+ recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

The project uses three datasets:

1. PPMI (Parkinson's Progression Markers Initiative)
   - Source: https://www.ppmi-info.org/
   - Requires registration and data use agreement

2. Tao Wu Dataset
3. NEUROCON Dataset

## Running Experiments

The project includes several experimental setups:

1. 2D analysis on individual datasets
2. 3D analysis on individual datasets
3. Cross-dataset analysis (2D)
4. Cross-dataset analysis (3D)

To run experiments:
```bash
python src/experiments/run_all_experiments.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:
[Paper citation details]

## Contributors

[Your name and other contributors]
