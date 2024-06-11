from openchem.models.Smiles2Label import Smiles2Label
from openchem.modules.embeddings.basic_embedding import Embedding
from openchem.modules.encoders.rnn_encoder import RNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.smiles_data_layer import SmilesDataset
from openchem.criterion.multitask_loss import MultitaskLoss

import torch
import torch.nn as nn

import numpy as np

from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import confusion_matrix

from openchem.data.utils import read_smiles_property_file
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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(smiles, labels, test_size=0.2,
                                                    random_state=42)

from openchem.data.utils import save_smiles_property_file
save_smiles_property_file('./benchmark_datasets/tox21/train.smi', X_train, y_train)
save_smiles_property_file('./benchmark_datasets/tox21/test.smi', X_test, y_test)

from openchem.data.smiles_data_layer import SmilesDataset
train_dataset = SmilesDataset('./benchmark_datasets/tox21/train.smi',
                              delimiter=',', cols_to_read=list(range(13)),
                              tokens=tokens, augment=True)
test_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                            delimiter=',', cols_to_read=list(range(13)),
                            tokens=tokens)
predict_dataset = SmilesDataset('./benchmark_datasets/tox21/test.smi',
                                delimiter=',', cols_to_read=[0],
                                tokens=tokens, return_smiles=True)

def multitask_auc(ground_truth, predicted):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import torch

    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)
    
    predicted_rounded = np.round(predicted + 0.05)
    
    n_tasks = ground_truth.shape[1]
    auc = []
    tn_l = []
    fp_l = []
    fn_l = []
    tp_l = []
    for i in range(n_tasks):
        
        ind = np.where(ground_truth[:, i] != 9)[0]

       
        tn, fp, fn, tp = confusion_matrix(ground_truth[ind, i], predicted_rounded[ind, i]).ravel()
        
        tn_l.append(tn)
        fp_l.append(fp)
        fn_l.append(fn)
        tp_l.append(tp)
        
        auc.append(roc_auc_score(ground_truth[ind, i], predicted[ind, i]))
    print('True Positive: %s' % sum(tp_l))
    print('False Positive: %s' % sum(fp_l))
    print('True Negative: %s' % sum(tn_l)) 
    print('False Negative: %s' % sum(fn_l)) 
    return np.mean(auc)

model = Smiles2Label

model_params = {
    'use_cuda': True,
    'task': 'multitask',
    'random_seed': 5,
    'use_clip_grad': True,
    'max_grad_norm': 10.0,
    'batch_size': 64,
    'num_epochs': 21,
    'logdir': './EXPERIMENT_1/logs/model_2',
    'print_every': 1,
    'save_every': 5,
    'train_data_layer': train_dataset,
    'val_data_layer': test_dataset,
    'predict_data_layer': predict_dataset,
    'eval_metrics': multitask_auc,
    'criterion': MultitaskLoss(ignore_index = 9, n_tasks=12).cuda(),
    'optimizer': RMSprop, # default: RMSprop
    'optimizer_params': {
        'lr': 0.001,
        },
    'lr_scheduler': StepLR,
    'lr_scheduler_params': {
        'step_size': 10, # default 10
        'gamma': 0.8 # default 0.8
    },
    'embedding': Embedding,
    'embedding_params': {
        'num_embeddings': train_dataset.num_tokens,
        'embedding_dim': 128, # default: 128
        'padding_idx': train_dataset.tokens.index(' ')
    },
    'encoder': RNNEncoder,
    'encoder_params': {
        'input_size': 128, # default: 128
        'layer': "GRU", # default LSTM
        'encoder_dim': 128, # default: 128
        'n_layers': 4, # default 4
        'dropout': 0.4, # default: 0.8                  best: 0.4 + mlp_dropout 0.4
        'is_bidirectional': False # default: False
    },
    'mlp': OpenChemMLP,
    'mlp_params': {
        'input_size': 128, # default 128
        'n_layers': 2, # default 2
        'hidden_size': [128, 12], # default [128, 12]
        'activation': [F.relu, torch.sigmoid], # default [F.relu, torch.sigmoid]
        'dropout': 0.4 # default: 0.0
    }
}
