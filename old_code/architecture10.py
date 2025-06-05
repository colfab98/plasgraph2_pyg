import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import random
import os

from thresholds import apply_thresholds


activation_map = {
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    'none': lambda x: x
}


class GCNModel(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self._params = parameters

        self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])  # 6 input features
        self.preproc_activation = activation_map[self['preproc_activation']]

        self.fc_input_1 = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        self.fc_input_2 = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        self.fully_connected_activation = activation_map[self['fully_connected_activation']]
        self.gcn_activation = activation_map[self['gcn_activation']]

        self.dropout = nn.Dropout(self['dropout_rate'])

        if self['tie_gnn_layers']:
            self.gcn = GCNConv(self['n_channels'], self['n_channels'])
            self.dense = nn.Linear(self['n_channels'] * 2, self['n_channels'])
        else:
            self.gcn_layers = nn.ModuleList([
                GCNConv(self['n_channels'], self['n_channels']) for _ in range(self['n_gnn_layers'])
            ])
            self.dense_layers = nn.ModuleList([
                nn.Linear(self['n_channels'] * 2, self['n_channels']) for _ in range(self['n_gnn_layers'])
            ])

        self.final_fc1 = nn.Linear(self['n_channels'] * 2, self['n_channels'])
        self.final_fc2 = nn.Linear(self['n_channels'], 2)

        self.output_activation = activation_map[self['output_activation']]


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.preproc_activation(self.preproc(x))
        node_identity = self.fully_connected_activation(self.fc_input_1(x))
        x = self.fully_connected_activation(self.fc_input_2(x))

        for i in range(self['n_gnn_layers']):
            x = self.dropout(x)
            if self['tie_gnn_layers']:
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

        x = self.output_activation(x)

        return x
    
    def __getitem__(self, key):
        return self._params[key]