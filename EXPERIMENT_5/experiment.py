import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import read_smiles_property_file

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

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

N_BITS = 512

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

# param_grid = {
#     'loss': ['deviance', 'exponential'],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [50, 100, 200]
# }

param_grid = {
    'subsample': [0.2, 0.5, 1.0],
    'criterion': ['friedman_mse', 'squarred_error', 'mse', 'mae'],
    'min_samples_split': [2, 4, 6]
}




# BEST PARAMS (ROC_AUC: 0.6467)
# loss: 'deviance'
# learning_rate: 0.2
# n_estimators: 200
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 


aucs_total = []

for sub in param_grid['subsample']:
    for crit in param_grid['criterion']:
        for ms in param_grid['min_samples_split']:

            aucs = []

            for i in range(12):
                ind1 = np.where(y_train[:, i] != 9)[0]
                classifier = GradientBoostingClassifier(random_state=SEED,
                                                        loss = 'deviance',
                                                        learning_rate = 0.2,
                                                        n_estimators = 200,
                                                        subsample = sub,
                                                        criterion = crit,
                                                        min_samples_split = ms
                                                        )
                
                classifier.fit(X_train_SMILES[ind1], y_train[ind1, i])
                
                ind2 = np.where(y_test[:, i] != 9)[0]
                y_pred = classifier.predict(X_test_SMILES[ind2])
                aucs.append(roc_auc_score(y_test[ind2, i], y_pred))
                # print(f'КЛАСС {i + 1}')
                # print(f'ROC_AUC: {roc_auc_score(y_test[ind2, i], y_pred)}')
                # print(f'Accuracy: {accuracy_score(y_test[ind2, i], y_pred)}')
                # print('-' * 80)
            aucs_total.append(np.mean(aucs))
            print('-' * 80)
            print(f'subsample: {sub}')
            print(f'criterion: {crit}')
            print(f'min_samples_split: {ms}')
            print(f'ROC_AUC: {np.mean(aucs)}')
            print('-' * 80)



