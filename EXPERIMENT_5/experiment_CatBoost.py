import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import read_smiles_property_file

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
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
    'iterations': [100, 200, 300],
    'depth': [3, 6, 12],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'l2_leaf_reg': [2.0, 3.0, 4.0],
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS', 'Poisson'],
    'bagging_temperature': [0.5, 1.0, 2.0, 3.0],
    'subsample': ['Poisson', 'Bernoulli', 'MVS'],
    'sampling_frequency': ['PerTree', 'PerTreeLevel'],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
    'max_leaves': [16, 32, 48],
    'leaf_estimation_method': ['Newton', 'Gradient', 'Exact'],
    'boost_from_average': ['Logloss', 'CrossEntropy']

}



for i in range(12):
   
    ind1 = np.where(y_train[:, i] != 9)[0]
    classifier = CatBoostClassifier(random_state=SEED,
                                    verbose = 0,
                                    eval_metric = roc_auc_score,
                                    task_type = 'GPU',
                                    devices = '0')
    clf = RandomizedSearchCV(classifier, param_grid, random_state=SEED, verbose = 2, n_iter = 30)                          
    search = clf.fit(X_train_SMILES[ind1], y_train[ind1, i])

    print(f'КЛАСС {i + 1}')
    print(search.best_params_)
    

    classifier2 = CatBoostClassifier(random_state=SEED,
                                     eval_metric = roc_auc_score,
                                     gpu_id = 0,
                                     **search.best_params_)
    
    ind2 = np.where(y_test[:, i] != 9)[0]
    y_pred = classifier.predict(X_test_SMILES[ind2])
    print(f'ROC_AUC: {np.mean(roc_auc_score(y_test[ind2, i], y_pred))}') 
    print('-' * 80)
    
   


 
    
    
    
    
    












