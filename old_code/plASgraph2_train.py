# import itertools
# import inspect
# import gzip
# import pandas as pd
# import networkx as nx
# import numpy as np
# import os
# import random
# import argparse
# import shutil

# import torch
# from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix

# import architecture
# import create_graph
# import config
# import thresholds


# def main(config_file : 'YAML configuration file',
#          train_file_list : 'csv file with a list of training samples',
#          file_prefix : 'common prefix to be used for all filenames listed in train_file_list, e.g. ../data/',
#          model_output_dir : 'name of an output folder with the trained model',
#          gml_file : 'optional output file for the graph in GML format' =None,
#          log_dir : 'optional output folder for various training logs' =None) : 
#     """Trains a model for a given dataset and writes out the trained model and optionally also the graph.

#     The train_file_list is a csv with a list of testing data samples. It contains no header and three comma-separated values:
#       name of the gfa.gz file from short read assemby,
#       name of the csv file with correct answers, 
#       id of the sample (string without colons, commas, whitespace, unique within the set)   
#     """
    
#     # Creating a dictionary parameters of parameter values from YAML file
#     parameters = config.config(config_file)


#     # read GFA and CSV files in the training set
#     G = create_graph.read_graph_set(file_prefix, train_file_list, parameters['minimum_contig_length'])

#     node_list = list(G.nodes)  # fix node order
    
#     if gml_file is not None:
#         nx.write_gml(G, path=gml_file)

#     print(nx.number_of_nodes(G), "nodes")
#     print(nx.number_of_edges(G), "edges")

#     data = create_graph.Networkx_to_PyG(G, node_list, parameters)


#     # #-------------------------------- 
#     # print("=" * 60)
#     # print("🔍 PyG Graph Diagnostic")
#     # print("=" * 60)

#     # print()

#     # print(f"type(data): {type(data)}")  # Should be <class 'torch_geometric.data.data.Data'>
#     # print(f"type(data.x): {type(data.x)}")  # Node features → <class 'torch.Tensor'>
#     # print(f"type(data.y): {type(data.y)}")  # Labels → <class 'torch.Tensor'>
#     # print(f"type(data.edge_index): {type(data.edge_index)}")  # Edges → <class 'torch.Tensor'>

#     # print()

#     # x = data.x
#     # y = data.y
#     # edge_index = data.edge_index

#     # # 🧬 Feature matrix
#     # print(f"🧬 Feature matrix shape: {x.shape}")  # [num_nodes, num_features]
#     # print(f"📋 Features used: {parameters['features']}")  # Same as Spektral
#     # print(f"🧾 Sample features (first 2 nodes):\n{x[:2]}")

#     # # 🏷️ Label matrix
#     # if y is not None:
#     #     print(f"\n🏷️ Label shape: {y.shape}")
#     #     print(f"🔢 Label sample (first 5):\n{y[:5]}")

#     #     # 📊 Label distribution
#     #     # Correct: Counts unique label pairs (rows)
#     #     unique_labels, counts = torch.unique(data.y, dim=0, return_counts=True)
#     #     print("\n📊 Label counts:")
#     #     for label, count in zip(unique_labels, counts):
#     #         print(f"  {label.tolist()} → {count.item()} nodes")
#     # else:
#     #     print("\n⚠️ No labels provided.")

#     # # 🔗 Edges (Adjacency)
#     # num_edges = edge_index.shape[1]
#     # num_nodes = x.shape[0]
#     # density = num_edges / (num_nodes ** 2)

#     # print(f"\n🔗 Edges shape (edge_index): {edge_index.shape}")
#     # print(f"  → Number of edges: {num_edges}")
#     # print(f"  → Density: {density:.6f}")

#     # # 🧍 Isolated nodes (nodes with degree 0)
#     # degrees = torch.zeros(num_nodes, dtype=torch.long)
#     # degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
#     # num_isolated = torch.sum(degrees == 0).item()
#     # print(f"🧍 Isolated nodes: {num_isolated}")

#     # #-------------------------------- 



#     # ----- BEFORE -----
#     graph = data[0]  # Access the first graph (since you have only one)
#     A_before = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)

#     self_loops_before = A_before.diagonal().sum()
#     density_before = A_before.nnz / (A_before.shape[0] ** 2)

#     print("🔬 BEFORE GCN Normalization")
#     print(f"  ➤ Shape: {A_before.shape}")
#     print(f"  ➤ Num edges (non-zero): {A_before.nnz}")
#     print(f"  ➤ Self-loops: {int(self_loops_before)}")
#     print(f"  ➤ Density: {density_before:.6f}")

#     # ----- APPLY NORMALIZATION -----
#     graph = data[0]  # Access the first graph from the dataset

#     # Add self-loops
#     edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
#     row, col = edge_index
#     deg = degree(row, graph.num_nodes, dtype=torch.float)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]


#     # Store normalized graph (for after diagnostics)
#     normalized_edge_index = edge_index
#     normalized_edge_weight = edge_weight

#     # ----- AFTER -----
#     A_after = to_scipy_sparse_matrix(normalized_edge_index, normalized_edge_weight, num_nodes=graph.num_nodes)
#     self_loops_after = A_after.diagonal().sum()
#     density_after = A_after.nnz / (A_after.shape[0] ** 2)

#     print("\n🔬 AFTER GCN Normalization")
#     print(f"  ➤ Shape: {A_after.shape}")
#     print(f"  ➤ Num edges (non-zero): {A_after.nnz}")
#     print(f"  ➤ Self-loops: {int(self_loops_after)}")
#     print(f"  ➤ Density: {density_after:.6f}")

#     # Extract diagonals
#     diag_before = A_before.diagonal()
#     diag_after = A_after.diagonal()

#     # Compare sample diagonal entries (self-loops)
#     print("\n🧾 Sample of adjacency diagonal (before vs after):")
#     rows_to_show = 5
#     for i in range(rows_to_show):
#         print(f"  {i:2} | {diag_before[i]:.2f}        | {diag_after[i]:.2f}")


#     #-------------------------------- 

#     if log_dir is not None:
#         if not os.path.isdir(log_dir):
#             os.mkdir(log_dir)

#     print(graph)

#     # sample weights and masking
#     number_total_nodes = len(node_list)

#     labels = [G.nodes[node_id]["text_label"] for node_id in node_list]
#     num_unlabelled = labels.count("unlabeled")
#     num_chromosome = labels.count("chromosome")
#     num_plasmid = labels.count("plasmid")
#     num_ambiguous = labels.count("ambiguous")
#     assert number_total_nodes == num_unlabelled + num_chromosome + num_plasmid + num_ambiguous
    
#     print(
#         "Chromosome contigs:",
#         num_chromosome,
#         "Plasmid contigs:",
#         num_plasmid,
#         "Ambiguous contigs:",
#         num_ambiguous,
#         "Unlabelled contigs:",
#         num_unlabelled,
#     )

#     # for each class, calculate weight. Set unlabelled contigs weight to 0
#     # chromosome_weight = (num_unlabelled + num_plasmid + num_ambiguous) / number_total_nodes
#     # plasmid_weight = (num_unlabelled + num_chromosome + num_ambiguous) / number_total_nodes
#     # ambiguous_weight = (num_unlabelled + num_chromosome + num_plasmid) / number_total_nodes
#     chromosome_weight = 1
#     plasmid_weight = parameters["plasmid_ambiguous_weight"]
#     ambiguous_weight = plasmid_weight
#     # plasmid_weight = 1
#     # ambiguous_weight = 1

#     masks = []

#     for node_id in node_list:
#         label = G.nodes[node_id]["text_label"]

#         if label == "unlabeled":
#             masks.append(0)
#         elif label == "chromosome":
#             masks.append(chromosome_weight)
#         elif label == "plasmid":
#             masks.append(plasmid_weight)
#         elif label == "ambiguous":
#             masks.append(ambiguous_weight)


#     # Set seeds for reproducibility
#     seed_number = parameters["random_seed"]
#     os.environ['PYTHONHASHSEED'] = str(seed_number)
#     random.seed(seed_number)
#     np.random.seed(seed_number + 1)
#     torch.manual_seed(seed_number + 2)  # Replace tf.random.set_seed with torch

#     # Train/validate masks (weighted)
#     masks_train = masks.copy()
#     masks_validate = masks.copy()

#     for i in range(len(masks)):
#         if random.random() > 0.8:
#             masks_train[i] = 0  # Zero for training
#         else:
#             masks_validate[i] = 0  # Zero for validation

#     print(masks_train[0:20])
#     print(masks_validate[0:20])

#     # Convert to torch tensors (instead of numpy arrays)
#     masks_train = torch.tensor(masks_train, dtype=torch.float)
#     masks_validate = torch.tensor(masks_validate, dtype=torch.float)

#     print(len(masks_train))
#     print(len(masks_validate))


#     learning_rate = parameters['learning_rate']

    



# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description=inspect.getdoc(main))
#     parser.add_argument("config_file", help="YAML configuration file")
#     parser.add_argument("train_file_list", help="csv file with a list of training samples")
#     parser.add_argument("file_prefix", help="common prefix to be used for all filenames listed in train_file_list, e.g. ../data/")
#     parser.add_argument("model_output_dir", help="name of the output folder with the trained model")
#     parser.add_argument("-g", dest="gml_file", help="optional output file for the graph in GML format", default = None)
#     parser.add_argument("-l", dest="log_dir", help="optional output folder for various training logs", default = None)
#     args = parser.parse_args()
#     main(** vars(args))