# gp-s
Gaussian Processes(GPs) is able to define probability distributions over sets of functions. 
motivation is non-linear regression. In project the implementation of GPs works using SMILES strings and molecular fingerprints as input. 
this project deals with training of gaussian process using Gpytorch library.
Tanimoto kernel in Gpytorch: Tanimoto kernel is defined in Gpytorch. Tanimoto kernel GPs on training molecules and evaluate on a heldout test set.

# Motivation
From the work of Henry B. Moss and Ryan-Rhys Griffiths on molecules using the GpFlow library. In the project the tanimoto Kernel is implemented using GpyTorch library.
Reference
"""
@misc{moss2020gaussian,
      title={Gaussian Process Molecule Property Prediction with FlowMO}, 
      author={Henry B. Moss and Ryan-Rhys Griffiths},
      year={2020},
      eprint={2010.01118},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

# installation
recomended using a conda environment

pip install rdkit-pypi
pip install gpytorch #GPyTorch requires Python >= 3.6 and PyTorch >= 1.6
for GpyTorch installation help visit the website: https://gpytorch.ai/
