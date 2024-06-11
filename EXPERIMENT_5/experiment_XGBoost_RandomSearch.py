import random
import itertools
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import read_smiles_property_file

from mordred import Calculator, descriptors

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, f1_score
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import KNNImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

xgb.set_config(verbosity=0)
# GradietBoosting, XGBoost, CatBoost

class Experiment_Boosting:

    def __init__(self, model):

        """ 
        model - модель МО, которой предстоит решать задачу:
        - 'XGBClassifier'
        - 'CatBoostClassifier'
        - 'LightGBM'
        - 'SVM'

        """

        
        self.model = model
        self.data_loaded = False
        self.model_fitted = False
        self.SEED = 123

    def load_data(self, data, desc):

        """
        data - датасет (к примеру, tox21)

        desc - дескриптор, кодирующий Smiles-строки молекул:

        - MorganFP
        - MordredFP

        """
        
        if data == 'tox21':

            print('ЗАГРУЗКА ДАННЫХ...')
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
            y_train = np.array(y_train)

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
            y_test = np.array(y_test)

            # PREDICT

            predict_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                                            delimiter=',', cols_to_read=[0],
                                            tokens=tokens, return_smiles = True)

            # FINGERPRINT EXECUTION

            if desc == 'MorganFP':

                N_BITS = 1024

                X_train_SMILES = []
                for i in X_train:
                    mol = Chem.MolFromSmiles(i)
                    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = N_BITS)
                    X_train_SMILES.append(list(fingerprint))


                X_test_SMILES = []
                for i in X_test:
                    mol = Chem.MolFromSmiles(i)
                    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits = N_BITS)
                    X_test_SMILES.append(list(fingerprint))

            
            elif desc == 'MordredFP':
                
                directory = r'EXPERIMENT_5/Mordred_desc/tox21'
                if not os.path.exists(directory):
                    print('СОЗДАНИЕ ДИРЕКТОРИИ ДЛЯ ДАТАСЕТОВ')
                    os.makedirs(directory)
                else:
                    print('ДИРЕКТОРИЯ ДЛЯ ДАТАСЕТОВ ОБНАРУЖЕНА')

                def All_Mordred_descriptors(data):
                    calc = Calculator(descriptors, ignore_3D=False)
                    mols = [Chem.MolFromSmiles(smi) for smi in data]
                    
                    df = calc.pandas(mols)
                    return df

                train_csv = os.path.join(directory, "train.csv")
                test_csv = os.path.join(directory, "test.csv")

                if os.path.isfile(train_csv) and os.path.isfile(test_csv):

                    print("ДАТАСЕТЫ уже существуют. Создание новых файлов отменено.")
                    X_train_SMILES = pd.read_csv(train_csv)
                    X_test_SMILES = pd.read_csv(test_csv)
                    
                    

                else:

                    print("ДАТАСЕТЫ не обнаружены. Создание новых файлов...")
                    X_train_SMILES = All_Mordred_descriptors(X_train)
                    X_test_SMILES = All_Mordred_descriptors(X_test)

                    for i in X_train_SMILES.columns:
                        try:
                            X_train_SMILES[i] = X_train_SMILES[i].astype('float32')
                        except:
                            X_train_SMILES[i] = X_train_SMILES[i].apply(pd.to_numeric, errors='coerce')
                    
                    for i in X_test_SMILES.columns:
                        try:
                            X_test_SMILES[i] = X_test_SMILES[i].astype('float32')
                        except:
                            X_test_SMILES[i] = X_test_SMILES[i].apply(pd.to_numeric, errors='coerce')

                    X_train_SMILES = X_train_SMILES.fillna(X_train_SMILES.mean()).astype('float32')
                    X_train_SMILES = X_train_SMILES.fillna(0).astype('float32')

                    X_test_SMILES = X_test_SMILES.fillna(X_test_SMILES.mean()).astype('float32')
                    X_test_SMILES = X_test_SMILES.fillna(0).astype('float32')

                    scaler = MinMaxScaler()
                    pca = PCA(n_components = 500, svd_solver = 'full', random_state = self.SEED)

                    X_train_SMILES = scaler.fit_transform(X_train_SMILES)
                    X_test_SMILES = scaler.transform(X_test_SMILES)

                    X_train_SMILES = pca.fit_transform(X_train_SMILES)
                    X_test_SMILES = pca.transform(X_test_SMILES)

                    pd.DataFrame(X_train_SMILES).to_csv(train_csv, index = False)
                    pd.DataFrame(X_test_SMILES).to_csv(test_csv, index = False)

            self.X_train_SMILES = np.array(X_train_SMILES)
            self.X_test_SMILES = np.array(X_test_SMILES)
            self.y_train = y_train
            self.y_test = y_test
            self.data_loaded = True 
            print('ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ')

    def fit_predict(self, task: str):

        """
        task - задача, которую предстоит решить:
        - 'multiclass_multilabel' - много классов и много задач
        - 'multiclass_singlelabel' - много классов и одна задача (в случае, если задач много, каждая решается отдельно )
        - 'twoclass_multilabel' - два класса и много задач
        - 'twoclass_singlelabel' - два класса и одна задача (в случае, если задач много, каждая решается отдельно)

        """

        if not self.data_loaded:
            raise Exception('Data must be loaded before fitting the model. Use a.load_data() to do it.')
        
        if task not in ['multiclass_multilabel', 'twoclass_multilabel', 'twoclass_singlelabel', 'multiclass_singlelabel']:
            raise Exception('Wrong task. You should select one of these tasks:/n- multiclass_multilabel/n- twoclass_multilabel/n- twoclass_singlelabel/n- multiclass_singlelabel')
        else:
            self.task = task

        if self.model == 'XGBClassifier':
            
            self.param_grid_XGBC = {'learning_rate': [0.01], 'max_depth': [1], 'n_estimators': [1], 'random_state': [123]}

            self.param_grid_XGBC = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 12],
            'learning_rate': [0.01, 0.05, 0.1],
            'random_state': [self.SEED]
            }

            self.param_grid_XGBC2 = list(itertools.product(*self.param_grid_XGBC.values()))
            
            # 'grow_policy': ['depthwise', 'lossguide'],
            # 'booster': ['gbtree', 'gblinear', 'dart'],
            # 'tree_method': ['exact', 'approx', 'hist'],


            if self.task == 'multiclass_multilabel':
              
                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')
                
                self.y_train_mc_ml = self.y_train.copy()
                self.y_train_mc_ml[np.where(self.y_train == 9)] = 2

                self.y_test_mc_ml = self.y_test.copy()
                self.y_test_mc_ml[np.where(self.y_test == 9)] = 2

                for i in self.param_grid_XGBC2:

                    print(f'n_estimators: {i[0]}, max_depth: {i[1]}, learning_rate: {i[2]}')

                    classifier = XGBClassifier(random_state = self.SEED, 
                                               objective = 'multi:softproba',
                                               n_estimators = i[0],
                                               max_depth = i[1],
                                               learning_rate = i[2])
                    
                    self.model_mc_ml = MultiOutputClassifier(classifier).fit(self.X_train_SMILES, self.y_train_mc_ml)
                    self.y_pred_mc_ml = np.array(self.model_mc_ml.predict_proba(self.X_test_SMILES))

                    aucs_ovo = []
                    aucs_ovr = []
                    f1 = []

                    for i in range(12):
                        
                        aucs_ovo.append(roc_auc_score(self.y_test_mc_ml[:, i], self.y_pred_mc_ml[i, :], multi_class = "ovo"))
                        aucs_ovr.append(roc_auc_score(self.y_test_mc_ml[:, i], self.y_pred_mc_ml[i, :], multi_class = "ovr"))
                        f1.append(f1_score(self.y_test_mc_ml[:, i], np.argmax(self.y_pred_mc_ml[i, :], axis = 1), average = 'macro'))

                        # print(f'КЛАСС {i + 1}')
                        # print(f'ROC_AUC ovo: {aucs_ovo[-1]}')
                        # print(f'ROC_AUC ovr: {aucs_ovr[-1]}')
                        # print('-' * 80)

                    # print('GLOBAL')
                    print(f'ROC_AUC ovo: {np.mean(aucs_ovo)}')
                    print(aucs_ovo)
                    print(f'ROC_AUC ovr: {np.mean(aucs_ovr)}')
                    print(aucs_ovr)
                    print(f'f1: {np.mean(f1)}')
                    print(f1)
                    print('*' * 100)

            elif self.task == 'multiclass_singlelabel':
                
                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')

                self.y_train_mc_sl = self.y_train.copy()
                self.y_train_mc_sl[np.where(self.y_train == 9)] = 2

                self.y_test_mc_sl = self.y_test.copy()
                self.y_test_mc_sl[np.where(self.y_test == 9)] = 2

                aucs_ovo = []
                aucs_ovr = []
                f1 = []

                # Class 1 {'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 100, 'random_state': 123}
                # Class 2 {}

                for i in range(12):

                    classifier = XGBClassifier(random_state = self.SEED, objective = 'multi:softproba')
                                               
                    #self.model_mc_sl = RandomizedSearchCV(classifier, 
                    #                                      self.param_grid_XGBC, 
                    #                                      random_state = self.SEED, 
                    #                                      verbose = 2,    
                    #                                      n_iter = 30)
                    
                    search = GridSearchCV(classifier, self.param_grid_XGBC, scoring = 'f1_macro', verbose = 2)

                    self.model_mc_sl = search.fit(self.X_train_SMILES, self.y_train_mc_sl[:, i])
                    print(f'КЛАСС {i + 1}')
                    print(self.model_mc_sl.best_params_)

                    self.y_pred_mc_sl = self.model_mc_sl.predict_proba(self.X_test_SMILES)
                    # print(f'ROC_AUC ovo: {roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovo")}')
                    # print(f'ROC_AUC ovr: {roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovr")}')

                    aucs_ovo.append(roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovo"))
                    aucs_ovr.append(roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovr"))
                    f1.append(f1_score(self.y_test_mc_sl[:, i], np.argmax(self.y_pred_mc_sl, axis = 1), average = 'macro'))
                    print('-' * 80)
                
                print(f'ROC_AUC ovo {np.mean(aucs_ovo)}')
                print(aucs_ovo)
                print(f'ROC_AUC ovr {np.mean(aucs_ovr)}')
                print(aucs_ovr)
                print(f'F1: {np.mean(f1)}')
                print(f1)
                print('*' * 100)

            elif self.task == 'twoclass_multilabel':
                
                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')

                imputer = KNNImputer(n_neighbors = 10)
                self.y_train_tc_ml = imputer.fit_transform(pd.DataFrame(self.y_train).replace(9, np.nan))
                self.y_test_tc_ml = imputer.transform(pd.DataFrame(self.y_test).replace(9, np.nan))

                classifier = XGBClassifier(eval_metric = roc_auc_score, 
                                           eval_set = [self.X_test_SMILES, self.y_train],
                                           random_state=self.SEED, 
                                           objective = 'binary:logistic') 

                
                self.model_tc_ml = GridSearchCV(classifier, self.param_grid_XGBC, scoring = 'f1', verbose = 2)
                search = self.model_tc_ml.fit(self.X_train_SMILES, np.round(self.y_train_tc_ml))
                self.best_params_tc_ml = search.best_params_
                print(self.best_params_tc_ml)
                self.y_pred_tc_ml = search.predict(self.X_test_SMILES)

                aucs = []
                f1 = []

                for i in range(12):
                    # print(f'КЛАСС {i + 1}')
                    # print(f'ROC_AUC: {roc_auc_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i])}')
                    # print(f'-' * 80)

                    aucs.append(roc_auc_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i]))
                    f1.append(f1_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i]))

                print(f'ROC_AUC: {roc_auc_score(np.round(self.y_test_tc_ml), self.y_pred_tc_ml)}')
                print(aucs)
                print(f'F1: {f1_score(np.round(self.y_test_tc_ml), self.y_pred_tc_ml, average = "samples")}')
                print(f1)
                print('*' * 100)

            elif self.task == 'twoclass_singlelabel':
                
                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')

                aucs = []
                f1 = []

                for i in range(12):

                    ind1 = np.where(self.y_train[:, i] != 9)[0]
                    classifier = XGBClassifier(random_state=self.SEED, objective = 'binary:logistic') 
                    self.model_tc_sl = GridSearchCV(classifier, self.param_grid_XGBC, scoring = 'f1', verbose = 1)
                    search = self.model_tc_sl.fit(self.X_train_SMILES[ind1], self.y_train[ind1, i])
                    self.best_params_tc_sl = search.best_params_

                    print(f'КЛАСС {i + 1}')
                    print(self.best_params_tc_sl)
                   
                    ind2 = np.where(self.y_test[:, i] != 9)[0]
                    y_pred = search.predict(self.X_test_SMILES[ind2])

                    # print(f'ROC_AUC: {roc_auc_score(self.y_test[ind2, i], y_pred)}') 

                    aucs.append(roc_auc_score(self.y_test[ind2, i], y_pred))
                    f1.append(f1_score(self.y_test[ind2, i], y_pred))
                    print('-' * 80)

                print(f'ROC_AUC: {np.mean(aucs)}')
                print(aucs) 
                print(f'F1: {np.mean(f1)}')
                print(f1)
                print('*' * 100)


        elif self.model == 'CatBoostClassifier':
            
            self.param_grid_CatBoost = {
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
                                        'boost_from_average': ['Logloss', 'CrossEntropy'],
                                        'random_state': [self.SEED]
                                        }

            self.param_grid_CatBoost = {'n_estimators': [200, 300, 400],
                                        'max_depth': [3, 6, 12],
                                        'learning_rate': [0.01, 0.05, 0.1],
                                        'random_state': [self.SEED]}

            self.param_grid_CatBoost2 = list(itertools.product(*self.param_grid_CatBoost.values()))

            if self.task == 'multiclass_multilabel':

                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')

                self.y_train_mc_ml = self.y_train.copy()
                self.y_train_mc_ml[np.where(self.y_train == 9)] = 2

                self.y_test_mc_ml = self.y_test.copy()
                self.y_test_mc_ml[np.where(self.y_test == 9)] = 2

                for i in self.param_grid_CatBoost2:
                    
                    print(f'n_estimators: {i[0]}, max_depth: {i[1]}, learning_rate: {i[2]}')

                    classifier = CatBoostClassifier(random_state = self.SEED,
                                                    loss_function = 'MultiClass', 
                                                    verbose = 0,
                                                    n_estimators = i[0],
                                                    max_depth = i[1],
                                                    learning_rate = i[2])

                    self.model_mc_ml = MultiOutputClassifier(classifier).fit(self.X_train_SMILES, self.y_train_mc_ml)
                    self.y_pred_mc_ml = np.array(self.model_mc_ml.predict_proba(self.X_test_SMILES))

                    aucs_ovo = []
                    aucs_ovr = []
                    f1 = []

                    for i in range(12):
                        
                        aucs_ovo.append(roc_auc_score(self.y_test_mc_ml[:, i], self.y_pred_mc_ml[i, :], multi_class = "ovo"))
                        aucs_ovr.append(roc_auc_score(self.y_test_mc_ml[:, i], self.y_pred_mc_ml[i, :], multi_class = "ovr"))
                        f1.append(f1_score(self.y_test_mc_ml[:, i], np.argmax(self.y_pred_mc_ml[i, :], axis = 1), average = 'macro'))
                        # print(f'КЛАСС {i + 1}')
                        # print(f'ROC_AUC ovo: {aucs_ovo[-1]}')
                        # print(f'ROC_AUC ovr: {aucs_ovr[-1]}')
                        # print('-' * 80)

                    #print('GLOBAL')
                    print(f'ROC_AUC ovo: {np.mean(aucs_ovo)}')
                    print(aucs_ovo)
                    print(f'ROC_AUC ovr: {np.mean(aucs_ovr)}')
                    print(aucs_ovr)
                    print(f'F1: {np.mean(f1)}')
                    print(f1)
                    print('*' * 100)
                
            elif self.task == 'multiclass_singlelabel':

                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')

                self.y_train_mc_sl = self.y_train.copy()
                self.y_train_mc_sl[np.where(self.y_train == 9)] = 2

                self.y_test_mc_sl = self.y_test.copy()
                self.y_test_mc_sl[np.where(self.y_test == 9)] = 2

                aucs_ovo = []
                aucs_ovr = []
                f1 = []

                for i in range(12):

                    classifier = CatBoostClassifier(random_state = self.SEED, 
                                                    loss_function = 'MultiClass',
                                                    verbose = 0)
                                             
                    
                    #clf = RandomizedSearchCV(classifier, self.param_grid_CatBoost, random_state = self.SEED, verbose = 2, n_iter = 30)
                    self.model_mc_sl = GridSearchCV(classifier, self.param_grid_CatBoost, scoring = 'f1_macro', verbose = 1)
                    search = self.model_mc_sl.fit(self.X_train_SMILES, self.y_train_mc_sl[:, i])
                    print(f'КЛАСС {i + 1}')
                    print(search.best_params_)

                    self.y_pred_mc_sl = search.predict_proba(self.X_test_SMILES)
                    # print(f'ROC_AUC ovo: {roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovo")}')
                    # print(f'ROC_AUC ovr: {roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovr")}')
                    aucs_ovo.append(roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovo"))
                    aucs_ovr.append(roc_auc_score(self.y_test_mc_sl[:, i], self.y_pred_mc_sl, multi_class = "ovr"))
                    f1.append(f1_score(self.y_test_mc_sl[:, i], np.argmax(self.y_pred_mc_sl, axis = 1), average = 'macro'))
                    print('-' * 80)

                print(f'ROC_AUC ovo {np.mean(aucs_ovo)}')
                print(aucs_ovo)
                print(f'ROC_AUC ovr {np.mean(aucs_ovr)}')
                print(aucs_ovr)
                print(f'F1: {np.mean(f1)}')
                print(f1)
                print('*' * 100)

            elif self.task == 'twoclass_multilabel':

                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')

                imputer = KNNImputer(n_neighbors = 10)
                self.y_train_tc_ml = imputer.fit_transform(pd.DataFrame(self.y_train).replace(9, np.nan))
                self.y_test_tc_ml = imputer.transform(pd.DataFrame(self.y_test).replace(9, np.nan))

                classifier = CatBoostClassifier(random_state = self.SEED, 
                                                loss_function = 'MultiLogloss',
                                                verbose = 0) 
                                               
                self.model_tc_ml = GridSearchCV(classifier, self.param_grid_CatBoost, scoring = 'f1', verbose = 1)
                search = self.model_tc_ml.fit(self.X_train_SMILES, np.round(self.y_train_tc_ml))
                
                self.best_params_tc_ml = search.best_params_
                print(self.best_params_tc_ml)
                

                self.y_pred_tc_ml = np.array(search.predict(self.X_test_SMILES))

                aucs = []
                f1 = []

                for i in range(12):
                    # print(f'КЛАСС {i + 1}')
                    # print(f'ROC_AUC: {roc_auc_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i])}')
                    # print(f'-' * 80)
                    aucs.append(roc_auc_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i]))
                    f1.append(f1_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i]))

                print('GLOBAL')
                print(f'ROC_AUC: {np.mean(aucs)}')
                print(aucs)
                print(f'F1: {np.mean(f1)}')
                print(f1)
                print('-' * 80)
                print('*' * 100)

            elif self.task == 'twoclass_singlelabel':
                
                print('*' * 100)
                print(' ' * 40 + self.task)
                print(' ')

                aucs = []
                f1 = []

                for i in range(12):

                    ind1 = np.where(self.y_train[:, i] != 9)[0]
                    classifier = CatBoostClassifier(random_state=self.SEED, loss_function = 'Logloss', verbose = 0) 
                    self.model_tc_sl = GridSearchCV(classifier, self.param_grid_CatBoost, scoring = 'f1', verbose = 1)
                    search = self.model_tc_sl.fit(self.X_train_SMILES[ind1], self.y_train[ind1, i])
                    self.best_params_tc_sl = search.best_params_

                    print(f'КЛАСС {i + 1}')
                    print(self.best_params_tc_sl)
                   
                    ind2 = np.where(self.y_test[:, i] != 9)[0]
                    y_pred = search.predict(self.X_test_SMILES[ind2])
                   
                    # print(f'ROC_AUC: {np.mean(roc_auc_score(self.y_test[ind2, i], y_pred))}') 

                    aucs.append(roc_auc_score(self.y_test[ind2, i], y_pred))
                    f1.append(f1_score(self.y_test[ind2, i], y_pred))
                    print('-' * 80)

                print(f'ROC_AUC: {np.mean(aucs)}')
                print(aucs) 
                print(f'F1: {np.mean(f1)}')
                print(f1)
                print('*' * 100)




        self.model_fitted = True

    def predict(self, task, best_params):
        self.task = task
        self.best_params = best_params

        if self.model == 'XGBClassifier':

            if self.task == 'twoclass_multilabel':

                imputer = KNNImputer(n_neighbors = 10)
                self.y_train_tc_ml = imputer.fit_transform(pd.DataFrame(self.y_train).replace(9, np.nan))
                self.y_test_tc_ml = imputer.transform(pd.DataFrame(self.y_test).replace(9, np.nan))

                classifier = XGBClassifier(random_state=self.SEED, 
                                           objective = 'binary:logistic',
                                           **self.best_params)

                classifier.fit(self.X_train_SMILES, np.round(self.y_train_tc_ml))

                self.y_pred_tc_ml = classifier.predict(self.X_test_SMILES)

                aucs = []
                f1 = []

                for i in range(12):
                    # print(f'КЛАСС {i + 1}')
                    # print(f'ROC_AUC: {roc_auc_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i])}')
                    # print(f'-' * 80)

                    aucs.append(roc_auc_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i]))
                    f1.append(f1_score(np.round(self.y_test_tc_ml[:, i]), self.y_pred_tc_ml[:, i]))

                print(f'ROC_AUC: {np.mean(aucs)}')
                print(aucs)
                print(f'F1: {np.mean(f1)}')
                print(f1)
                print('*' * 100)

# КЛАСС 1
# 0.714
# 
# Меняем 9 на 1
# 

# best_params  = {'tree_method': 'hist', 
#                 'n_estimators': 200, 
#                 'max_depth': 3, 
#                 'learning_rate': 0.01, 
#                 'grow_policy': 'depthwise', 
#                 'booster': 'gbtree'}

#task1 = Experiment_Boosting('XGBClassifier')
#task1.load_data('tox21', desc = 'MordredFP')
#task1.fit_predict('multiclass_multilabel')
#task1.fit_predict('multiclass_singlelabel')
#task1.fit_predict('twoclass_multilabel')
#task1.predict('twoclass_multilabel', {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100})
#task1.fit_predict('twoclass_singlelabel')

task2 = Experiment_Boosting('CatBoostClassifier')
task2.load_data('tox21', desc = 'MordredFP')
task2.fit_predict('multiclass_multilabel')
#task2.fit_predict('multiclass_singlelabel')
#task2.fit_predict('twoclass_multilabel')
#task2.fit_predict('twoclass_singlelabel')

'A'