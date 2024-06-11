import runpy
import torch
import copy
from torch.nn.parallel import DataParallel
from openchem.models.openchem_model import predict, evaluate, build_training
from openchem.data.utils import create_loader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.data.utils import read_smiles_property_file
from openchem.data.utils import get_tokens

N_TASKS = 12

data = read_smiles_property_file('./benchmark_datasets/tox21/tox21.csv',
                                 cols_to_read=[13] + list(range(0,12)),
                                 keep_header=False)


smiles = data[0]
tokens, _, _ = get_tokens(smiles)
tokens = tokens + ' '


val_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                            delimiter=',', cols_to_read=list(range(13)),
                            tokens=tokens)



val_loader = create_loader(val_dataset,
                           batch_size = 64,
                           shuffle=False,
                           num_workers=1,
                           pin_memory=True)

class MyModel:
    def __init__(self, model_number):

       
        self.column_names = ['SMILES', 'Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8', 'Task9', 'Task10', 'Task11', 'Task12']
        self.model_number = model_number
        self.config_file = rf'EXPERIMENT_1/configs/tox21_rnn_config_{self.model_number}.py'
        self.config_module = runpy.run_path(self.config_file)

        self.model_config = self.config_module.get('model_params', None)
        self.model_config['use_cuda'] = torch.cuda.is_available()
        self.model_object = self.config_module.get('model', None)

        self.model = self.model_object(params=self.model_config)
        self.model = DataParallel(self.model)
        self.model.module.load_state_dict(torch.load(rf'EXPERIMENT_1/model_OpenChem_CLASSIFIER_ENSEMBLE_{self.model_number}.pth'))

        self.predict_dataset = copy.deepcopy(self.model_config['predict_data_layer'])

        self.predict_loader = create_loader(self.predict_dataset,
                                        batch_size=self.model_config['batch_size'],
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=True)
        print(f'___---***    МОДЕЛЬ №{self.model_number} УСПЕШНО ИНИЦИАЛИЗИРОВАНА    ***---___')
        
    def predict(self):
        predict(self.model, self.predict_loader)
        self.pred = pd.read_csv(self.model_config['logdir'] + '/predictions.txt', names = self.column_names) 
        print(f'***---___ ПРЕДСКАЗАНИЯ МОДЕЛИ №{self.model_number} УСПЕШНО ЗАГРУЖЕНЫ ___---***')
        return self.pred

    def eval(self):
        self.criterion, self.optimizer, self.lr_scheduler = build_training(self.model, self.model_config)
        print(evaluate(self.model, val_loader, self.criterion))






# Первая модель -    128 размерность, 0.4 1-й Дропаут, 0.4 2-й Дропаут, 3 слоя LSTM
# Вторая модель -    128 размерность, 0.4 1-й Дропаут, 0.4 2-й Дропаут, 4 слоя GRU
# Третья модель -    256 размерность, 0.4 1-й Дропаут, 0.4 2-й Дропаут, 4 слоя LSTM
# Четвертая модель - 128 размерность, 0.5 1-й Дропаут, 0.3 2-й Дропаут, 3 слоя LSTM
# Пятая модель -     128 размерность, 0.3 1-й Дропаут, 0.5 2-й Дропаут, 3 слоя LSTM
# Шестая модель -    64  размерность, 0.4 1-й Дропаут, 0.4 2-й Дропаут, 4 слоя GRU
# Седьмая модель -   64  размерность, 0.5 1-й Дропаут, 0.3 2-й Дропаут, 3 слоя LSTM
# Восьмая модель -   64 размерность,  0.3 1-й Дропаут, 0.5 2-й Дропаут, 3 слоя GRU
# Девятая модель -   64  размерность, 0.2 1-й Дропаут, 0.6 2-й Дропаут, 4 слоя LSTM
# Десятая модель -   512 размерность, 0.6 1-й Дропаут, 0.2 2-й Дропаут, 4 слоя LSTM


numbers_of_models = list(range(1, 11))

models = {}
preds = {}
for i in numbers_of_models:
    models[f'model{i}'] = MyModel(i)
    preds[f'pred_model{i}'] = models[f'model{i}'].predict()



    
   

tn_l = []
fp_l = []
fn_l = []
tp_l = []
auc = []
f1 = []
ground_truth = np.array(val_dataset.target, dtype = np.int8)


for i in range(N_TASKS):

    task = f'Task{i + 1}'
    predicted = np.array([preds[f'pred_model{j}'][task] for j in numbers_of_models]).mean(axis = 0)
    #predicted = pred5[f'Task{i + 1}']
    predicted_rounded = np.round(predicted)


    ind = np.where(ground_truth[:, i] != 9)[0]
    tn, fp, fn, tp = confusion_matrix(ground_truth[ind, i], predicted_rounded[ind]).ravel()
        
    tn_l.append(tn)
    fp_l.append(fp)
    fn_l.append(fn)
    tp_l.append(tp)
    auc.append(roc_auc_score(ground_truth[ind, i], predicted[ind]))
    f1.append(f1_score(ground_truth[ind, i], np.round(predicted[ind])))
    print('-' * 80)
    print('CLASS %s' % (i + 1))
    print('TP: %s' % tp)
    print('FP: %s' % fp)
    print('TN: %s' % tn)
    print('FN: %s' % fn)
    print(f'AUC_ROC: {auc[-1]}')
    print(f'F1 score: {f1[-1]}')

# Global
print('-' * 80)
print('GLOBAL')
print(f'True Positive: {sum(tp_l)}')
print(f'False Positive: {sum(fp_l)}')
print(f'True Negative: {sum(tn_l)}')
print(f'False Negative: {sum(fn_l)}')
print(f'AUC_ROC: {np.mean(auc)}')
print(f'F1 score: {np.mean(f1)}')

print('-' * 80)
print(f'В ансамбле использовались следующие модели: {numbers_of_models}')
a = 1

