import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import read_smiles_property_file

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

# GradietBoosting, XGBoost, CatBoost

SEED = 123


data = read_smiles_property_file('./benchmark_datasets/tox21/tox21.csv',
                                 cols_to_read=[13] + list(range(0,12)),
                                 keep_header=False)
smiles = data[0]
labels = np.array(data[1:])

labels[np.where(labels=='')] = '999'
labels = labels.T

from openchem.data.utils import get_tokens
tokens, _, _ = get_tokens(smiles)
tokens = tokens + ' '

train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi',
                              delimiter=',', cols_to_read=list(range(13)),
                              tokens=tokens, augment=True, return_smiles = True)

# TRAIN

X_train = []
y_train = []

for j in train_dataset:
    X_train.append(''.join([chr(i) for i in j['object']]))
    y_train.append(j['labels'])  

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=np.int8)

# TEST

test_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                            delimiter=',', cols_to_read=list(range(13)),
                            tokens=tokens, return_smiles = True)

X_test = []
y_test = []

for j in test_dataset:
    X_test.append(''.join([chr(i) for i in j['object']]))
    y_test.append(j['labels'])  

X_test = np.array(X_test)
y_test = np.array(y_test, dtype = np.int8)

# PREDICT

predict_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                                delimiter=',', cols_to_read=[0],
                                tokens=tokens, return_smiles = True)

# FINGERPRINT EXECUTION

N_BITS = 1024

X_train_SMILES = []
for i in X_train:
    mol = Chem.MolFromSmiles(i)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = N_BITS)
    X_train_SMILES.append(list(fingerprint))

X_train_SMILES = np.array(X_train_SMILES)

X_test_SMILES = []
for i in X_test:
    mol = Chem.MolFromSmiles(i)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = N_BITS)
    X_test_SMILES.append(list(fingerprint))

X_test_SMILES = np.array(X_test_SMILES)

# CLASSIFIERS

param_grid = {
    'grow_policy': ['depthwise', 'lossguide'],
    'learning_rate': [0.01, 0.05, 0.1],
    'booster': ['gbtree', 'gblinear', 'dart']
}






# BEST PARAMS (ROC_AUC: 0.6628)
# n_estimators: 300
# max_depth: 6
# max_leaves: 6
# 
# 
# 
# 2048 - 0.701, 0.804, 0.710, 0.640, 0.637, 0.671, 0.611, 0.594, 0.588, 0.584, 0.704, 0.595
# 1024 - 0.712, 0.804, 0.712, 0.652, 0.615, 0.678, 0.612, 0.606, 0.590, 0.569, 0.717, 0.588
# 512  - 0.675, 0.803, 0.715, 0.596, 0.625, 0.663, 0.611, 0.587, 0.572, 0.561, 0.691, 0.588
# 256  - 0.702, 0.748, 0.699, 0.585, 0.621, 0.664, 0.589, 0.579, 0.553, 0.547, 0.678, 0.563
# 
# 
# 
# 
# 


aucs_total = []



for gp in param_grid[list(param_grid.keys())[0]]:
    for lr in param_grid[list(param_grid.keys())[1]]:
        for booster in param_grid[list(param_grid.keys())[2]]:
            
            aucs = []
            prc = []
            print('-' * 80)
            print(f'{list(param_grid.keys())[0]}: {gp}')
            print(f'{list(param_grid.keys())[1]}: {lr}')
            print(f'{list(param_grid.keys())[2]}: {booster}')
            for i in range(12):
                ind1 = np.where(y_train[:, i] != 9)[0]
                classifier = XGBClassifier(random_state=SEED,
                                           n_estimators = 300,
                                           max_depth = 6,
                                           gpu_id = 0)
                
                classifier.fit(X_train_SMILES[ind1], y_train[ind1, i])
                
                ind2 = np.where(y_test[:, i] != 9)[0]
                y_pred = classifier.predict(X_test_SMILES[ind2])
                aucs.append(roc_auc_score(y_test[ind2, i], y_pred))
                #prc.append(precision_recall_curve(y_test[ind2, i], y_pred))
                # print(f'КЛАСС {i + 1}')
                # print(f'ROC_AUC: {roc_auc_score(y_test[ind2, i], y_pred)}')
                # print(f'Accuracy: {accuracy_score(y_test[ind2, i], y_pred)}')
                # print('-' * 80)
            aucs_total.append(np.mean(aucs))
            
            print(f'ROC_AUC: {np.mean(aucs)}')
            #print(f'PR-AUC: {np.mean(prc)}')
            print('-' * 80)



