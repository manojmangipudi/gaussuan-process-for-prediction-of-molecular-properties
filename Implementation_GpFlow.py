# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:07:44 2021

@author: manoj
"""

import sys
import os
sys.path.append('..')  # to import from GP.kernels and property_predition.data_utils

import math
import pandas as pd
import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from property_prediction.data_utils import transform_data, TaskDataLoader, featurise_mols
from gpflow.utilities import positive
from matplotlib import pyplot as plt

#============================DATA==============================================
#Training data is 100 points in [0,1] inclusive regularly spaced
df = pd.read_csv('dataset_BCF_KNN_clean.csv')

# data split into train test validate ======================================

Train_data = df.loc[df['GNN_status'] == 'Train']  
Test_data = df.loc[df['GNN_status'] == 'Test']  
validate_data = df.loc[df['GNN_status'] == 'Valid']

# divide Train Test Smiles and experimental_value
X_train_smiles = Train_data['SMILES'].to_list()
y_train_smiles = Train_data['Experimental value [log(L/kg)]'].to_numpy()
y_train_qsar = Train_data['Predicted value [log(L/kg)]'].to_numpy()

X_test_smiles = Test_data['SMILES'].to_list()
y_test_smiles = Test_data['Experimental value [log(L/kg)]'].to_numpy()
y_test_qsar = Test_data['Predicted value [log(L/kg)]'].to_numpy()

X_validate_smiles = validate_data['SMILES'].to_list()
y_validate_smiles = validate_data['Experimental value [log(L/kg)]'].to_numpy()
y_validate_qsar = validate_data['Predicted value [log(L/kg)]'].to_numpy()

# ===============================smiles to bitvector =================================

# # Convert X_train X_test smiles to MACCSkeys and morgan fingerprint==========
rdkit_mols1 = [MolFromSmiles(smiles) for smiles in X_train_smiles]
X_train1 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in rdkit_mols1]
X_train1 = np.asarray(X_train1)

rdkit_mols2 = [MolFromSmiles(smiles) for smiles in X_test_smiles]
X_test1 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in rdkit_mols2]
X_test1 = np.asarray(X_test1)

rdkit_mols3 = [MolFromSmiles(smiles) for smiles in X_validate_smiles]
X_validate1 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in rdkit_mols3]
X_validate1 = np.asarray(X_validate1)

rdkit_mols4 = [MolFromSmiles(smiles) for smiles in X_train_smiles]
X_train2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048) for mol in rdkit_mols4]
X_train2 = np.asarray(X_train2)

rdkit_mols5 = [MolFromSmiles(smiles) for smiles in X_test_smiles]
X_test2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048) for mol in rdkit_mols5]
X_test2 = np.asarray(X_test2)

rdkit_mols6 = [MolFromSmiles(smiles) for smiles in X_validate_smiles]
X_validate2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048) for mol in rdkit_mols6]
X_validate2 = np.asarray(X_validate2)

X_train = np.concatenate((X_train1, X_train2), axis = 1)
X_test = np.concatenate((X_test1, X_test2), axis = 1)
X_validate = np.concatenate((X_validate1, X_validate2), axis = 1)


def broadcasting_elementwise(op, a, b):
    """
    Apply binary operation `op` to every pair in tensors `a` and `b`.

    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
    return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))


class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
    
# ===============================================================================    
# We define the Gaussian Process Regression training objective

m = None

def objective_closure():
    return -m.log_marginal_likelihood() 


y_train = y_train_smiles.reshape(-1,1)
y_test = y_test_smiles.reshape(-1,1)
y_validate1 = y_validate_smiles.reshape(-1,1)

#  We standardise the outputs but leave the inputs unchanged
_, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

y_scaler_val = StandardScaler()
y_validate = y_scaler_val.fit_transform(y_validate1)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
X_validate = X_validate.astype(np.float64)

k = Tanimoto()
m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

# Optimise the kernel variance and noise level by the marginal likelihood

opt = gpflow.optimizers.Scipy()
opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)  # Model summary

# mean and variance GP prediction ===============================================

# validation data=================
# y_pred1, y_var1 = m.predict_f(X_validate)
# y_pred1 = y_scaler.inverse_transform(y_pred1)
# y_validate = y_scaler.inverse_transform(y_validate)
# score1 = r2_score(y_validate, y_pred1)
# rmse1 = np.sqrt(mean_squared_error(y_validate, y_pred1))
# mae1 = mean_absolute_error(y_validate, y_pred1)

# print("\nR^2: {:.3f}".format(score1))
# print("RMSE: {:.3f}".format(rmse1))
# print("MAE: {:.3f}".format(mae1))
# ======================================================================

y_pred, y_var = m.predict_f(X_test)
y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(y_test)
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(X_test, 10)


# Compute R^2, RMSE and MAE on test set molecules

score = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nR^2: {:.3f}".format(score))
print("RMSE: {:.3f}".format(rmse))
print("MAE: {:.3f}".format(mae))

# train data check ============================================================
# Input foe Applicability Domain
y_pred_train, y_var1 = m.predict_f(X_train)
y_pred_train = y_scaler.inverse_transform(y_pred_train)
y_train1 = y_scaler.inverse_transform(y_train)

score1 = r2_score(y_train1, y_pred_train)
rmse1 = np.sqrt(mean_squared_error(y_train1, y_pred_train))
mae1 = mean_absolute_error(y_train1, y_pred_train)

print("\nR^2: {:.3f}".format(score1))
print("RMSE: {:.3f}".format(rmse1))
print("MAE: {:.3f}".format(mae1))

# ========================================================
f_sd = []
for i in y_var:
    f_sd.append(math.sqrt(i))
f_sd1 = np.array(f_sd).reshape(-1,)

## plot
# plt.figure(figsize=(16, 8))
# plt.plot(y_test , y_pred, "k*", mew=2)
# plt.plot(y_test, y_test, 'b', lw=2)
# plt.errorbar(y_test, y_pred, yerr=(f_sd1), fmt="k*", label="std dev")
# plt.fill_between(
#     X_test[:, 0],
#     y_pred[:, 0] - 1.96 * np.sqrt(y_var[:, 0]),
#     y_pred[:, 0] + 1.96 * np.sqrt(y_var[:, 0]),
#     color="C0",
#     alpha=0.2,
# )

