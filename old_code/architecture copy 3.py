import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import random
import os

# === Hardcoded parameters from config_default.yaml ===
params = {
    'n_gnn_layers': 6,
    'n_channels': 32,
    'n_channels_preproc': 10,
    'tie_gnn_layers': True,
    'preproc_activation': 'sigmoid',
    'fully_connected_activation': 'relu',
    'gcn_activation': 'relu',
    'output_activation': 'sigmoid',
    'l2_reg': 2.5e-4,
    'dropout_rate': 0.1,
    'loss_function': 'crossentropy',
    'random_seed': 123,
    'plasmid_ambiguous_weight': 1.0,
    'learning_rate': 0.005,
    'epochs': 1000,
    'early_stopping_patience': 100,
    'n_input_features': 6
}

# === Activation Mapping ===
activation_map = {
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    'none': lambda x: x
}

class PlasGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.preproc = nn.Linear(params['n_input_features'], params['n_channels_preproc'])  # 6 input features
        self.preproc_activation = activation_map[params['preproc_activation']]

        self.fc_input_1 = nn.Linear(params['n_channels_preproc'], params['n_channels'])
        self.fc_input_2 = nn.Linear(params['n_channels_preproc'], params['n_channels'])
        self.fully_connected_activation = activation_map[params['fully_connected_activation']]
        self.gcn_activation = activation_map[params['gcn_activation']]

        self.dropout = nn.Dropout(params['dropout_rate'])

        if params['tie_gnn_layers']:
            self.gcn = GCNConv(params['n_channels'], params['n_channels'])
            self.dense = nn.Linear(params['n_channels'] * 2, params['n_channels'])
        else:
            self.gcn_layers = nn.ModuleList([
                GCNConv(params['n_channels'], params['n_channels']) for _ in range(params['n_gnn_layers'])
            ])
            self.dense_layers = nn.ModuleList([
                nn.Linear(params['n_channels'] * 2, params['n_channels']) for _ in range(params['n_gnn_layers'])
            ])

        self.final_fc1 = nn.Linear(params['n_channels'] * 2, params['n_channels'])
        self.final_fc2 = nn.Linear(params['n_channels'], 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.preproc_activation(self.preproc(x))
        node_identity = self.fully_connected_activation(self.fc_input_1(x))
        x = self.fully_connected_activation(self.fc_input_2(x))

        for i in range(params['n_gnn_layers']):
            x = self.dropout(x)
            if params['tie_gnn_layers']:
                x = self.gcn_activation(self.gcn(x, edge_index))
                x = torch.cat([node_identity, x], dim=1)
                x = self.dropout(x)
                x = self.fully_connected_activation(self.dense(x))
            else:
                x = self.gcn_activation(self.gcn_layers[i](x, edge_index))
                x = torch.cat([node_identity, x], dim=1)
                x = self.dropout(x)
                x = self.fully_connected_activation(self.dense_layers[i](x))

        x = torch.cat([node_identity, x], dim=1)
        x = self.dropout(x)
        x = self.fully_connected_activation(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)

        return x