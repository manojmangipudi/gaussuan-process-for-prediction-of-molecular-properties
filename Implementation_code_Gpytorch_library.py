# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:07:00 2021

@author: manoj
"""

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
from property_prediction.data_utils import transform_data 


import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50
torch.set_printoptions(precision=10)

# ============================Import Data================================= #
df = pd.read_csv('dataset_DAPHNIA_DEMETRA.csv')

# ============================Data Split================================== #
"""
Data split into training, testing set
"""
Train_data = df.loc[df['Status'] == 'Training']  
Test_data = df.loc[df['Status'] == 'Test']  

X_train_smiles = Train_data['SMILES'].to_list()
y_train_smiles = Train_data['Experimental value [-log(mol/l)]'].to_numpy()
y_train_qsar = Train_data['Predicted value [-log(mol/l)]'].to_numpy()


X_test_smiles = Test_data['SMILES'].to_list()
y_test_smiles = Test_data['Experimental value [-log(mol/l)]'].to_numpy()
y_test_qsar  = Test_data['Predicted value [-log(mol/l)]'].to_numpy()

"""
Loading smiles from the daphnia_demetra dataset, We import MACCSKeys and Morganfingerprint
bit vectors using RDkit library. The molecule is converted into bit vector in this step.
"""
# Convert X_train X_test smiles to MACCSkeys 
rdkit_mols1 = [MolFromSmiles(smiles) for smiles in X_train_smiles]
X_train1 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in rdkit_mols1]
X_train1 = np.asarray(X_train1)

rdkit_mols2 = [MolFromSmiles(smiles) for smiles in X_test_smiles]
X_test1 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in rdkit_mols2]
X_test1 = np.asarray(X_test1)

# Morganfingerprint conversion
rdkit_mols3 = [MolFromSmiles(smiles) for smiles in X_train_smiles]
X_train2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 5, 512) for mol in rdkit_mols3]
X_train2 = np.asarray(X_train2)

rdkit_mols4 = [MolFromSmiles(smiles) for smiles in X_test_smiles]
X_test2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 5, 512 ) for mol in rdkit_mols4]
X_test2 = np.asarray(X_test2)

X_train = np.concatenate((X_train1, X_train2), axis = 1)
X_test = np.concatenate((X_test1, X_test2), axis = 1)

y_train = y_train_smiles.reshape(-1,1)
y_test = y_test_smiles.reshape(-1,1)

# transform the data leaving input as it is
_, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

# torch conversion
train_X = torch.Tensor(X_train)
train_y = torch.Tensor(y_train.reshape(-1,))
test_X = torch.Tensor(X_test)
test_y = torch.Tensor(y_test)
# =================Kerenl implementation=========================================
"""
Kernel is the parameter that defines relationship between the points.
The kernel is implemented using gpytorch library.

We import the kernel implementation file from Kernel_GP
"""
From Kernel_GP import TanimotoKernel()
# ====================GAUSSIAN PROCESSES MODEL=======================================
    
# Use the simplest form of GP model, exact inference
class TanimoGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_X, train_y, likelihood):
        super().__init__(train_X, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = TanimotoKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint= gpytorch.constraints.GreaterThan(1e-4))
model = TanimoGPModel(train_X, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
training_iter = 500
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(torch.Tensor(train_X))
    # Calc loss and backprop gradients
    loss = -mll(output, torch.Tensor(train_y.reshape(-1,)))
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   variance: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.variance.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
 
# =============================================================================    
# gp for predictions
""" 
In the predictions we evaluate the model and estimate the likelihood.
further performing the predictions on the test set.
"""
model.eval()
likelihood.eval()
# prediction on test set
with torch.no_grad():
    observed_pred = likelihood(model(test_X))
    
Z_t = observed_pred.mean.numpy().reshape(-1,1)
y_pred = y_scaler.inverse_transform(Z_t)
y_test1 = y_scaler.inverse_transform(y_test)


score = r2_score(y_test1, y_pred)
rmse = np.sqrt(mean_squared_error(y_test1, y_pred))
mae = mean_absolute_error(y_test1, y_pred)

print("\nR^2: {:.3f}".format(score))
print("RMSE: {:.3f}".format(rmse))
print("MAE: {:.3f}".format(mae))

#=================================================
"""
Note: We predict the training data again to check the model reliability,
      Which we test the training set using the model and use the results 
      in Applicability domain implementation.
"""
with torch.no_grad():
    observed_pred1 = likelihood(model(train_X))
Z_t1 = observed_pred1.mean.numpy().reshape(-1,1)
y_pred_train = y_scaler.inverse_transform(Z_t1)
y_train1 = y_scaler.inverse_transform(y_train)

score1 = r2_score(y_train1, y_pred_train)
rmse1 = np.sqrt(mean_squared_error(y_train1, y_pred_train))
mae1 = mean_absolute_error(y_train1, y_pred_train)

# print("\nR^2: {:.3f}".format(score1))
# print("RMSE: {:.3f}".format(rmse1))
# print("MAE: {:.3f}".format(mae1))

# ================================================
# we plot graph between experimental and predicted values.
f_var = observed_pred.variance
f_covar = observed_pred.covariance_matrix

f_sd = []
for i in f_var:
    f_sd.append(math.sqrt(i))
f_sd1 = np.array(f_sd).reshape(-1,)


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(14, 7))
    #Plot training data as black stars
    ax.plot(test_y, Z_t, 'k*', label = "Molecule Test Data")
    # Plot predictive means as blue line
    ax.plot(test_y, test_y, 'b', label = "Mean prediction")
    # standard deviation of the points
    ax.errorbar(test_y, Z_t, yerr= (f_sd1), fmt="k*", label="standard deviation")
    #plt.title("dataset_DAPHNIA_DEMETRA")
    plt.xlabel("y_experimental")
    plt.ylabel("y_predicted")
    plt.legend(loc="best")











