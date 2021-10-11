"""
In the document we observe the applicability domain of the gaussian process model.
which is the responce in which the model makes predictions with a given reliability.
The parameters considered in the in defining the Applicability Domain were:
1) similar molecules with known experimental value.
2) Accuracy( average error) of prediction for similar molecules.
3) Concordance with similar molecules( average difference between target compound prediction
and experimental values of similar molecules).
4) Maximum error of prediction among similar molecules.
5) Atom Centered Fragments similarity check.
6) Global applicability domain index (Global AD Index). Which takes into account 
all the previous indices.
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 06:10:51 2021

@author: manoj
"""
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
import numpy as np
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

# ============================Import data===========================
# obtain the dataframe
df = pd.read_csv('dataset_DAPHNIA_DEMETRA.csv')

# data split
Train_data = df.loc[df['Status'] == 'Training']  
Test_data = df.loc[df['Status'] == 'Test']  

"""
We import prediction values, prediction values of training set and 
standard deviation values from the gaussian process model implementation.
"""
from Implementation_code_Gpytorch_library import y_pred
from Implementation_code_Gpytorch_library import y_pred_train
from Implementation_code_Gpytorch_library import f_sd
# Data 
X_train_smiles = Train_data['SMILES'].to_list()
X_test_smiles = Test_data['SMILES'].to_list()
X_exp1 = Train_data['Experimental value [-log(mol/l)]'].to_list()
Y_pred1 = y_pred.reshape(-1,)
Y_pred = Y_pred1.tolist()
Y_pred_train = y_pred_train.reshape(-1,)

# =================Similar molecules with known experimental values==================

x_similarity = []
index_val = []
similarity_score=[]
matched = {}
threshold_data = []

def similarity(mol_test, mol_train, thresh = 0.4):
    for moleculesi in mol_test:
        m1 = Chem.MolFromSmiles(moleculesi)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 1024) 
        x_sim1 = []
        for moleculesj in mol_train:
            m2 = Chem.MolFromSmiles(moleculesj)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 1024)
            sim1 = DataStructs.TanimotoSimilarity(fp1, fp2)
            x_sim1.append(sim1)
        x_similarity.append(x_sim1)
    
    scl = []
    for i in x_similarity:
        scl.append(max(i))
    print(scl)
    print("\n==========================================\n")

    #similarity_score = []
    for i in scl:
        if (i >= thresh):
            similarity_score.append(i)
        else:
            similarity_score.append(0)
  
        # **************similarity values with threshold imposed*************
    #threshold_data = []
    for indices, items in enumerate(x_similarity):
        temp = []
        for i in items:
            temp1 = []
            if(i >= thresh):
                temp1.append(i)
            temp.append(temp1)
        threshold_data.append(temp)
     # *************************index values*************************
     
    sim_ind = 0
    for i in x_similarity:
        match_list = []
        thre_ind = 0
        for j in i:
            if (len(threshold_data[sim_ind][thre_ind]) != 0) and (j == threshold_data[sim_ind][thre_ind][0]):
                match_list.append(thre_ind)                
            thre_ind += 1    
        matched[sim_ind] = match_list 
        sim_ind += 1       
    return similarity_score

print(similarity(X_test_smiles,X_train_smiles))

y_pred_train_val = [Y_pred_train[i] for i in matched.values()]

similarity_score_new = [x for x in similarity_score if x != 0]

# ==============ccuracy (average error) of prediction for similar molecules ===================
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
error = []
def accuracy(mol_test1 ,mol_train1):
    avg_error = (1 - (abs((mol_test1 - mol_train1)/mol_train1)))
    #print(avg_error)
    for i in avg_error:
        if (len(i) == 0):
            #error.append(0)
            continue
        else:
            avg_accuracy = sum(i)/ len(i)
            error.append(avg_accuracy)
    return error
                
print(accuracy(Y_pred1, y_pred_train_val))
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

# ====================Concordance with similar molecules=========================
X_exp = np.array(X_exp1)
y_exp_val = [X_exp[i] for i in matched.values()]

conco = []
#conco_error = []
def concordance(mol_test1 ,mol_train1):
    avg_error1 = zip(mol_test1, mol_train1)
    concor_diff=[]
    for mol_test1, mol_train1 in avg_error1:
        diff = (1 - (abs((mol_test1 - mol_train1)/mol_train1)))
        concor_diff.append(diff)
    for i in concor_diff:
        if (len(i) == 0):
            #conco.append("null")
            continue
        else:
            avg_conco = sum(i)/ len(i)
            conco.append(avg_conco)
    return conco
        
print(concordance(y_pred_train_val, y_exp_val))
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

# ======================Maximum error of prediction among similar molecules =======================
max_error = []
# reversed_maxerror = []
def Maxerror(mol_test1, mol_train1):
    error = zip(mol_test1, mol_train1)
    max_error1 = []
    for mol_test1, mol_train1 in error:
        diff = (1 - (abs((mol_test1 - mol_train1)/mol_train1)))
        max_error1.append(diff)
    for i in max_error1:
        if (len(i) == 0):
            # max_error.append("null")
            continue
        else:
            max_error1 = min(i)
            max_error.append(max_error1)
    return max_error

print(Maxerror(y_pred_train_val, y_exp_val))
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
# =====================Atom centered fragment similarity check =============================

atoms_train = []
for i in X_train_smiles:
    for atoms in Chem.MolFromSmiles(i).GetAtoms():
        atoms_train.append(atoms.GetSymbol())
   
mollist = []
for mol in X_test_smiles:
    lib = []
    for atoms in Chem.MolFromSmiles(mol).GetAtoms():
        lib.append(atoms.GetSymbol()) 
    mollist.append(lib)
  
atm = 0
atom_centered_score = []
new_atoms_list = list(set(atoms_train))
for i in mollist:
    new_list = list(set(i))
    len_new_list = len(new_list)
    counter = 0
    for j in new_list:
        if j in new_atoms_list:
            counter += 1
        else:
            break
    if counter == len_new_list:
        atom_centered_score.append(1)
    else:
        atom_centered_score.append(0) 
        
print(atom_centered_score)
print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

null_ind = []
for indices, items in enumerate(similarity_score):
    if (items == 0):
        null_ind.append(indices)

for n, i in enumerate(atom_centered_score):
    for j in null_ind:
        if(j == n):
            atom_centered_score[n] = 0  
atom_centered_score_new = [x for x in atom_centered_score if x !=0]

# ===========================Global Applicability Domain Index=========================
  
ad = [similarity_score_new, error, conco, max_error, atom_centered_score_new] 
arrays = [np.array(x) for x in ad]
Global_AD_index = [np.mean(k) for k in zip(*arrays)]
#print(Global_AD_index) 

# sort standard deviation values=================================================

for n, i in enumerate(f_sd):
    for j in null_ind:
        if(j == n):
            f_sd[n] = 'null'
f_std = [x for x in f_sd if x != 'null']

# ===============================RESULT TABLE:===============================

data_table = pd.DataFrame([Global_AD_index, similarity_score_new, error, conco, max_error, atom_centered_score_new, f_std]).T
data_table.columns = ['Global_AD_index', 'similarity_score', 'reversed_error', 'conco_error', 'reversed_maxerror','atom_centered_score','f_sd']

# ============================Plot the Results =============================
fig, ax = plt.subplots(2, 3, figsize=(22, 16)) 
ax[0, 0].plot(f_std, similarity_score_new, 'k*', label = "similarity")
ax[0, 0].set_title('similarity vs std')
ax[0, 0].set_xlabel('f_std')
ax[0, 0].set_ylabel('similarity')

ax[0, 1].plot(f_std, error, 'k*', label = "accuracy")
ax[0, 1].set_title('accuracy vs std')
ax[0, 1].set_xlabel('f_std')
ax[0, 1].set_ylabel('accuracy')
    
ax[0, 2].plot(f_std, conco, 'k*', label = "concordance")
ax[0, 2].set_title('concordance vs std')
ax[0, 2].set_xlabel('f_std')
ax[0, 2].set_ylabel('concordance')

ax[1, 0].plot(f_std, max_error, 'k*', label = "max_error")
ax[1, 0].set_title('max_error vs std')
ax[1, 0].set_xlabel('f_std')
ax[1, 0].set_ylabel('max_error')

ax[1, 1].plot(f_std, atom_centered_score_new, 'k*', label = "atom_centered")
ax[1, 1].set_title('atom_centered vs std')
ax[1, 1].set_xlabel('f_std')
ax[1, 1].set_ylabel('atom_centered_score')

ax[1, 2].plot(f_std, Global_AD_index, 'k*', label = "ADI")
ax[1, 2].set_title('Global_AD_index vs std')
ax[1, 2].set_xlabel('f_std')
ax[1, 2].set_ylabel('Global_AD_index')




