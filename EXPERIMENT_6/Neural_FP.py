import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np


class NeuralFingerprint(nn.Module):
    def init(self, input_dim, hidden_dim, output_dim):
        super(NeuralFingerprint, self).init()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, output_dim)

        self.conv = GraphConvolution(hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.W_i(x))
        x = self.conv(x, edge_index)
        x = F.relu(self.W_h(x))
        x = self.W_o(x)
        return x

class GraphConvolution(MessagePassing):
    def init(self, hidden_dim):
        super(GraphConvolution, self).init(aggr='add')
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
    





# Подготовка данных
def prepare_data(graphs, node_features, labels):
    data_list = []
    for i in range(len(graphs)):
        G = graphs[i]
        x = node_features[i]
        y = labels[i]
        edge_index = from_networkx(G).edge_index
        data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, y=torch.tensor([y], dtype=torch.float))
        data_list.append(data)
    return data_list

# Обучение модели
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch {}, Loss: {:.4f}'.format(epoch+1, total_loss / len(train_loader)))























# Пример использования
# Предположим, у вас есть списки графов, признаков узлов и меток
# graphs = [graph1, graph2, ...]
# node_features = [features1, features2, ...]
# labels = [label1, label2, ...]
















# Подготовка данных
data_list = prepare_data(graphs, node_features, labels)

# Разделение на обучающую и тестовую выборки (просто для примера)
train_data = data_list[:int(0.8*len(data_list))]
test_data = data_list[int(0.8*len(data_list)):]

# Создание модели
input_dim = node_features.shape[1]  # Предполагается, что все узлы имеют одинаковое количество признаков
hidden_dim = 64
output_dim = 1  # Для бинарной классификации
model = NeuralFingerprint(input_dim, hidden_dim, output_dim)

# Определение критерия и оптимизатора
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
num_epochs = 10
train_model(model, train_loader, optimizer, criterion, num_epochs)

# Оценка модели на тестовой выборке (просто для примера)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
model.eval()
with torch.no_grad():
    for data in test_loader:
        output = model(data.x, data.edge_index)
        # Здесь можно провести оценку модели, например, сравнить предсказания с реальными метками