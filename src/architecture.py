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

        self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])  
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
    



# Define a custom GGNN layer
class GGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, activation='relu', dropout_rate=0.0, edge_gate_hidden_dim=32):
        super().__init__(aggr='add') # aggregation for sum of messages
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_map[activation]
        self.dropout = nn.Dropout(dropout_rate)

        # Edge Gate layer; +1 for kmer dot product between contigs 
        edge_gate_input_dim = in_channels * 2 + 1        

        self.edge_gate_network = nn.Sequential(
            nn.Linear(edge_gate_input_dim, edge_gate_hidden_dim),
            nn.ReLU(),  
            nn.Linear(edge_gate_hidden_dim, 1) 
        )

        # GRU-like gates
        self.lin_z = nn.Linear(in_channels + out_channels, out_channels) 
        self.lin_r = nn.Linear(in_channels + out_channels, out_channels) 
        self.lin_h = nn.Linear(in_channels + out_channels, out_channels) 

    def forward(self, x, edge_index, edge_attr=None): 

        # add self-loops to all nodes in the graph
        # k-mer dot product feature for self-loops will be 0.0
        edge_index_with_self_loops, edge_attr_with_self_loops = add_self_loops(
            edge_index, edge_attr=edge_attr, num_nodes=x.size(0), fill_value=0.
        )

        # normalization (use the edge_index with self-loops)
        row, col = edge_index_with_self_loops
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index_with_self_loops, x=x, norm=norm, edge_attr=edge_attr_with_self_loops)

    def message(self, x_i, x_j, norm, edge_attr): 
        if edge_attr is None: 
            edge_attr_expanded = torch.zeros((x_i.size(0), 1), device=x_i.device) 
        else:
            edge_attr_expanded = edge_attr

        # features of the target node (x_i), source node (x_j), and the edge attribute (edge_attr_expanded, which is the k-mer dot product for that edge) are concatenated
        edge_gate_input = torch.cat([x_i, x_j, edge_attr_expanded], dim=-1) 
        edge_gate_logit = self.edge_gate_network(edge_gate_input)

        # edge gate value between 0 and 1 for each edge
        edge_gate_value = torch.sigmoid(edge_gate_logit)

        original_message = norm.view(-1, 1) * x_j

        # message that would have been passed from node j is multiplied by this learned edge_gate_value
        gated_message = edge_gate_value * original_message
        return gated_message

    def update(self, aggr_out, x):
        z_input = torch.cat([x, aggr_out], dim=-1)
        r_input = torch.cat([x, aggr_out], dim=-1)
        
        z = torch.sigmoid(self.lin_z(z_input)) 
        r = torch.sigmoid(self.lin_r(r_input)) 

        h_candidate_input = torch.cat([r * x, aggr_out], dim=-1)
        h_candidate = self.lin_h(h_candidate_input)
        h_candidate = self.activation(h_candidate) 

        out = (1 - z) * x + z * h_candidate
        return self.dropout(out)


class GGNNModel(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self._params = parameters

        self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])
        self.preproc_activation = activation_map[self['preproc_activation']]

        self.initial_node_transform = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        self.fully_connected_activation = activation_map[self['fully_connected_activation']]

        self.dropout = nn.Dropout(self['dropout_rate'])

        self.edge_gate_hidden_dim = self['edge_gate_hidden_dim']

        if self['tie_gnn_layers']:
            self.ggnn_layer = GGNNConv(self['n_channels'], 
                                       self['n_channels'],
                                      activation=self['gcn_activation'], 
                                      dropout_rate=self['dropout_rate'],
                                      edge_gate_hidden_dim = self.edge_gate_hidden_dim)
        else:
            self.ggnn_layers = nn.ModuleList([GGNNConv(self['n_channels'], 
                                                       self['n_channels'],
                                        activation=self['gcn_activation'], 
                                        dropout_rate=self['dropout_rate'],
                                        edge_gate_hidden_dim = self.edge_gate_hidden_dim)
                for _ in range(self['n_gnn_layers'])
            ])

        self.final_fc1 = nn.Linear(self['n_channels'] * 2, self['n_channels']) 
        self.final_fc2 = nn.Linear(self['n_channels'], 2)

        self.output_activation = activation_map[self['output_activation']]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.preproc_activation(self.preproc(x)) 

        h_0 = self.fully_connected_activation(self.initial_node_transform(x)) 
        
        node_identity = h_0 

        h = h_0 

        for i in range(self['n_gnn_layers']):

            # extract the edge attributes
            current_edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

            if self['tie_gnn_layers']:
                h = self.ggnn_layer(h, edge_index, edge_attr=current_edge_attr) 
            else:
                h = self.ggnn_layers[i](h, edge_index, edge_attr=current_edge_attr) 

        x = torch.cat([node_identity, h], dim=1) 
        x = self.dropout(x) 
        x = self.fully_connected_activation(self.final_fc1(x)) 
        x = self.dropout(x) 
        x = self.final_fc2(x) 

        x = self.output_activation(x) 

        return x

    def __getitem__(self, key):
        return self._params[key]






def apply_to_graph(model, graph, parameters, apply_thresholds_flag=True):

    device = next(model.parameters()).device
    graph = graph.to(device)

    model.eval()
    with torch.no_grad():

        output = model(graph)

        preds = output

        preds_np = preds.cpu().numpy()

    preds_clipped = np.clip(preds_np, 0, 1)


    if apply_thresholds_flag:
        preds_final = apply_thresholds(preds_clipped, parameters)
        preds_final = np.clip(preds_final, 0, 1)
    else:
        preds_final = preds_clipped


    return preds_final

