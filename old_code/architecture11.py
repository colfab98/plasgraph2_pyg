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
    def __init__(self, in_channels, out_channels, activation='relu', dropout_rate=0.0):
        super().__init__(aggr='add') # "add" aggregation for sum of messages
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation_map[activation]
        self.dropout = nn.Dropout(dropout_rate)

        # GRU-like gates
        self.lin_z = nn.Linear(in_channels + out_channels, out_channels) # Input + current hidden state for update gate
        self.lin_r = nn.Linear(in_channels + out_channels, out_channels) # Input + current hidden state for reset gate
        self.lin_h = nn.Linear(in_channels + out_channels, out_channels) # Input + (reset * current hidden state) for candidate hidden state

    def forward(self, x, edge_index):
        # x: Node feature matrix (H_t-1 for each node)
        # edge_index: Graph connectivity in COO format

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Compute degrees for normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 # Handle nodes with degree 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: Start propagating messages.
        # Pass 'norm' as an argument to the message function
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm): # <-- HERE: Added 'norm' to the signature
        # x_j: Features of neighboring nodes.
        # Messages are simply the neighbor's features for now, multiplied by normalization factor.
        return norm.view(-1, 1) * x_j # <-- HERE: Applied the normalization

    def update(self, aggr_out, x):
        # aggr_out: Aggregated messages from neighbors (summed in this case)
        # x: Previous node features (h_v^(t-1))

        # Concatenate aggregated messages and previous node state
        z = self.lin_z(torch.cat([x, aggr_out], dim=-1))
        r = self.lin_r(torch.cat([x, aggr_out], dim=-1))
        
        z = torch.sigmoid(z) # Update gate
        r = torch.sigmoid(r) # Reset gate

        # Candidate hidden state calculation
        h_candidate = self.lin_h(torch.cat([r * x, aggr_out], dim=-1))
        h_candidate = self.activation(h_candidate) # Apply activation to candidate

        # Final hidden state update
        out = (1 - z) * x + z * h_candidate
        return self.dropout(out) # Apply dropout after update


class GGNNModel(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self._params = parameters

        self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])
        self.preproc_activation = activation_map[self['preproc_activation']]

        # The initial transformation for node features before GRU iterations
        # This will be the initial hidden state for the GGNN layer
        self.initial_node_transform = nn.Linear(self['n_channels_preproc'], self['n_channels'])
        self.fully_connected_activation = activation_map[self['fully_connected_activation']]

        self.dropout = nn.Dropout(self['dropout_rate'])

        # Use GGNNConv layer
        if self['tie_gnn_layers']:
            self.ggnn_layer = GGNNConv(self['n_channels'], self['n_channels'],
                                      activation=self['gcn_activation'], dropout_rate=self['dropout_rate'])
        else:
            self.ggnn_layers = nn.ModuleList([
                GGNNConv(self['n_channels'], self['n_channels'],
                         activation=self['gcn_activation'], dropout_rate=self['dropout_rate'])
                for _ in range(self['n_gnn_layers'])
            ])

        # Final layers remain similar, potentially adjusting input channels if node_identity is still used.
        # For a pure GGNN, node_identity might be integrated differently or not used this way.
        # If you want to keep 'node_identity' as a fixed initial representation,
        # you'd concatenate it with the final recurrent state.
        self.final_fc1 = nn.Linear(self['n_channels'] * 2, self['n_channels']) # Keep * 2 if concatenating initial with final
        self.final_fc2 = nn.Linear(self['n_channels'], 2)

        self.output_activation = activation_map[self['output_activation']]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.preproc_activation(self.preproc(x)) # Preprocessing features

        # Initial transformation to get the starting hidden states for GGNN
        h_0 = self.fully_connected_activation(self.initial_node_transform(x)) # Initial hidden state for GGNN
        
        # The node_identity should probably be removed or rethought if the recurrent state is the primary node representation.
        # If you still want a fixed "identity" and the recurrent state, keep it as:
        node_identity = h_0 # Or a separate linear layer if it's meant to be truly distinct.

        h = h_0 # Initialize recurrent hidden state with h_0

        for i in range(self['n_gnn_layers']): # Iterate through GNN layers
            if self['tie_gnn_layers']: # If layers are tied, use the same GGNNConv instance
                h = self.ggnn_layer(h, edge_index) # Apply GGNN layer
            else:
                h = self.ggnn_layers[i](h, edge_index) # Apply independent GGNN layers

        # Concatenate the initial node identity with the final recurrent state for the final classification layers.
        x = torch.cat([node_identity, h], dim=1) # Concatenate node identity with final recurrent state
        x = self.dropout(x) # Apply dropout
        x = self.fully_connected_activation(self.final_fc1(x)) # Apply final fully connected layers
        x = self.dropout(x) # Apply dropout
        x = self.final_fc2(x) # Final output layer

        x = self.output_activation(x) # Apply output activation

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

