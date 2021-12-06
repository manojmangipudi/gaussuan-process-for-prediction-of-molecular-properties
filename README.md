# gp-s
--> Gaussian Processes(GPs) is able to define probability distributions over sets of functions. 
--> Gps work well although data is less. Motivation of the project is to check perfromance of GPs in Molecular propert prediction. In the project the implementation of GPs         works using SMILES strings and molecular fingerprints as input. 
--> In the project Gps were modeled uising both Gpytorch and GPFlow libraries, to check the performance.
    Here GpyTorch is implemented using Pytorch
    and GpFlow is implemented using TensorFlow
--> Tanimoto kernel in Gpytorch: Customized Tanimoto kernel using Gpytorch. Tanimoto kernel GPs on training molecules and evaluate on a heldout test set.


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
# Contents

1. Data
   * We can observe Environmental Health and Safety Molecular data, Experimental data obtained from Otto-von-Guericke(Max Plank Institute for dynamics of complex technical        systems)
   * The properties predicted were 
     - Water Octanol Partition Coeff, 
     - Acute Toxicity for water flea, 
     - Bio-transformation constants for discrete organic chemical in fish, 
     - Bio-concentration factor in fish toxicity, 
     - Skin permeability- Adsorption Coefficient.
 2. GP Model implementation in GPyTorch
     - Custom Kernel Implementation
 3. Gp Model implementation in GpFlow
 4. Applicability Domain Implementation
   

# installation
recomended using a conda environment

pip install rdkit-pypi
pip install gpytorch #GPyTorch requires Python >= 3.6 and PyTorch >= 1.6
for GpyTorch installation help visit the website: https://gpytorch.ai/
