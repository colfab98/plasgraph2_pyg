# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.utils import add_self_loops, degree


# class PlasGraph(torch.nn.Module):
#     def __init__(self, parameters):
#         super().__init__()
#         self._parameters = parameters
#         reg = self['l2_reg']

#         # Input preprocessing
#         self.preproc = nn.Linear(self['n_input_features'], self['n_channels_preproc'])
#         self._fully_connected_input_1 = nn.Linear(self['n_channels_preproc'], self['n_channels'])
#         self._fully_connected_input_2 = nn.Linear(self['n_channels_preproc'], self['n_channels'])

#         # Dropout layers for GNN
#         self._gnn_dropout_pre_gcn = nn.ModuleList([
#             nn.Dropout(self['dropout_rate']) for _ in range(self['n_gnn_layers'])
#         ])
#         self._gnn_dropout_pre_fully_connected = nn.ModuleList([
#             nn.Dropout(self['dropout_rate']) for _ in range(self['n_gnn_layers'])
#         ])

#         # GCN + Dense layers (tied or untied)
#         if self['tie_gnn_layers']:
#             self.gnn_gcn_layer = GCNConv(self['n_channels'], self['n_channels'], add_self_loops=False, bias=True)
#             self.gnn_fully_connected_layer = nn.Linear(2 * self['n_channels'], self['n_channels'])
#             self._gnn_gcn = [self.gnn_gcn_layer] * self['n_gnn_layers']
#             self._gnn_fully_connected = [self.gnn_fully_connected_layer] * self['n_gnn_layers']
#         else:
#             self._gnn_gcn = nn.ModuleList([
#                 GCNConv(self['n_channels'], self['n_channels'], add_self_loops=False, bias=True)
#                 for _ in range(self['n_gnn_layers'])
#             ])
#             self._gnn_fully_connected = nn.ModuleList([
#                 nn.Linear(2 * self['n_channels'], self['n_channels'])
#                 for _ in range(self['n_gnn_layers'])
#             ])

#         # Final layers
#         self._dropout_last_1 = nn.Dropout(self['dropout_rate'])
#         self._dropout_last_2 = nn.Dropout(self['dropout_rate'])
#         self._fully_connected_last_1 = nn.Linear(2 * self['n_channels'], self['n_channels'])
#         self._fully_connected_last_2 = nn.Linear(self['n_channels'], self['n_labels'])

#     def __getitem__(self, key):
#         return self._parameters[key]

#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

#         # Preprocessing layers
#         x = self.activation(self.preproc(x), self['preproc_activation'])
#         node_identity = self.activation(self._fully_connected_input_1(x), self['fully_connected_activation'])
#         x = self.activation(self._fully_connected_input_2(x), self['fully_connected_activation'])

#         # GNN layers
#         for i in range(self['n_gnn_layers']):
#             x = self._gnn_dropout_pre_gcn[i](x)
#             x = self._gnn_gcn[i](x, edge_index, edge_weight)
#             x = torch.cat([node_identity, x], dim=1)
#             x = self._gnn_dropout_pre_fully_connected[i](x)
#             x = self.activation(self._gnn_fully_connected[i](x), self['fully_connected_activation'])

#         # Final layers
#         x = torch.cat([node_identity, x], dim=1)
#         x = self._dropout_last_1(x)
#         x = self.activation(self._fully_connected_last_1(x), self['fully_connected_activation'])
#         x = self._dropout_last_2(x)
#         x = self.activation(self._fully_connected_last_2(x), self['output_activation'])

#         return x

#     def activation(self, x, name):
#         if name == "relu":
#             return F.relu(x)
#         elif name == "sigmoid":
#             return torch.sigmoid(x)
#         elif name == "tanh":
#             return torch.tanh(x)
#         elif name is None:
#             return x
#         else:
#             raise ValueError(f"Unsupported activation: {name}")
