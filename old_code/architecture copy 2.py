# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv
# from torch.nn import functional as F
# from torch_geometric.utils import to_dense_adj


# def concat(a, b):
#     return torch.cat([a, b], dim=-1)

# class PlasGraph(nn.Module):

#     def __init__(self, parameters):
#         super().__init__()
#         self.__dict__["_cfg"] = parameters
#         cfg = self._cfg

#         # Activation mapping
#         act_map = {
#             'relu': nn.ReLU(),
#             'sigmoid': nn.Sigmoid(),
#             'tanh': nn.Tanh(),
#             'linear': nn.Identity()
#         }

#         # Select activation modules
#         self.gcn_activation = act_map[cfg['gcn_activation'].lower()]
#         self.fc_activation = act_map[cfg['fully_connected_activation'].lower()]


#                 # Input preprocessing layers
#         self.preproc = nn.Linear(
#             in_features=len(cfg['features']),
#             out_features=cfg['n_channels_preproc']
#         )
#         self.preproc_activation = act_map[cfg['preproc_activation'].lower()]

#         self._fully_connected_input_1 = nn.Linear(
#             cfg['n_channels_preproc'], cfg['n_channels']
#         )
#         self._fully_connected_input_2 = nn.Linear(
#             cfg['n_channels_preproc'], cfg['n_channels']
#         )

#         # GNN input dropouts
#         self._gnn_dropout_pre_gcn = nn.ModuleList([
#             nn.Dropout(p=cfg['dropout_rate']) for _ in range(cfg['n_gnn_layers'])
#         ])
#         self._gnn_dropout_pre_fully_connected = nn.ModuleList([
#             nn.Dropout(p=cfg['dropout_rate']) for _ in range(cfg['n_gnn_layers'])
#         ])


#         # Handle tied vs untied GNN layers
#         if cfg['tie_gnn_layers']:
#             self.gnn_gcn_layer = GCNConv(cfg['n_channels'], cfg['n_channels'], add_self_loops=True, bias=True)
#             self.gnn_fc_layer = nn.Linear(cfg['n_channels'], cfg['n_channels'])

#             self._gnn_gcn = nn.ModuleList([self.gnn_gcn_layer] * cfg['n_gnn_layers'])
#             self._gnn_fc = nn.ModuleList([self.gnn_fc_layer] * cfg['n_gnn_layers'])

#         else:
#             self._gnn_gcn = nn.ModuleList([
#                 GCNConv(cfg['n_channels'], cfg['n_channels'], add_self_loops=True, bias=True)
#                 for _ in range(cfg['n_gnn_layers'])
#             ])
#             self._gnn_fc = nn.ModuleList([
#                 nn.Linear(cfg['n_channels'], cfg['n_channels'])
#                 for _ in range(cfg['n_gnn_layers'])
#             ])

#                 # Final fully connected layers
#         self._fc_last_1 = nn.Linear(cfg['n_channels'], cfg['n_channels'])
#         self._fc_last_2 = nn.Linear(cfg['n_channels'], cfg['n_labels'])

#         # Dropouts for the final layers
#         self._dropout_last_1 = nn.Dropout(p=cfg['dropout_rate'])
#         self._dropout_last_2 = nn.Dropout(p=cfg['dropout_rate'])

#         # Output activation (optional, set to Identity if None)
#         if cfg['output_activation'] is None:
#             self.output_activation = nn.Identity()
#         else:
#             act_map = {
#                 'relu': nn.ReLU(),
#                 'sigmoid': nn.Sigmoid(),
#                 'tanh': nn.Tanh(),
#                 'linear': nn.Identity()
#             }
#             self.output_activation = act_map[cfg['output_activation'].lower()]


#     def forward(self, x, edge_index):
#         cfg = self._cfg  # access config parameters

#         # Input processing
#         x = self.preproc(x)
#         x = self.preproc_activation(x)

#         node_identity = self._fully_connected_input_1(x)
#         x = self._fully_connected_input_2(x)
#         x = self.fc_activation(x)

#         # GNN Layers
#         for i in range(cfg['n_gnn_layers']):
#             x = self._gnn_dropout_pre_gcn[i](x)
#             x = self._gnn_gcn[i](x, edge_index)
#             x = self.gcn_activation(x)

#             merged = concat(node_identity, x)

#             x = self._gnn_dropout_pre_fully_connected[i](merged)
#             x = self._gnn_fc[i](x)
#             x = self.fc_activation(x)

#         # Final fully connected layers
#         merged = concat(node_identity, x)
#         x = self._dropout_last_1(merged)
#         x = self._fc_last_1(x)
#         x = self.fc_activation(x)
#         x = self._dropout_last_2(x)
#         x = self._fc_last_2(x)
#         x = self.output_activation(x)

#         return x

