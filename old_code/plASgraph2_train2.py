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
# from torch_geometric.loader import DataLoader

# import architecture
# import create_graph
# import config
# import thresholds

# import pickle  

# def main(config_file : 'YAML configuration file',
#          train_file_list : 'csv file with a list of training samples',
#          file_prefix : 'common prefix to be used for all filenames listed in train_file_list, e.g. ../data/',
#          model_output_dir : 'name of an output folder with the trained model',
#          gml_file : 'optional output file for the graph in GML format' =None,
#          log_dir : 'optional output folder for various training logs' =None) : 

#     parameters = config.config(config_file)

#      # Define the path for the cache file
#     cache_path = os.path.join(model_output_dir, "all_graphs.pkl")

#     if os.path.exists(cache_path):
#         # Load the cached Spektral graph object if available
#         with open(cache_path, 'rb') as f:
#             G, all_graphs, node_list = pickle.load(f)
#     else:
#         G = create_graph.read_graph_set(file_prefix, train_file_list, parameters['minimum_contig_length'])
#         node_list = list(G.nodes)

#         if gml_file is not None:
#             nx.write_gml(G, path=gml_file)

#         all_graphs = create_graph.Networkx_to_PyG(G, node_list, parameters)

#         os.makedirs(os.path.dirname(cache_path), exist_ok=True)

#         # Save the generated graph object to cache for future runs
#         with open(cache_path, 'wb') as f:
#             pickle.dump((G, all_graphs, node_list), f)

#     data = all_graphs[0]

#         # Manually normalize adjacency matrix (equivalent to Spektral GCNFilter):contentReference[oaicite:32]{index=32}
#     # Add self-loops to include identity matrix
#     edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
#     # Compute D^{-1/2} for each node (node degree)
#     row, col = edge_index
#     deg = degree(row, data.num_nodes, dtype=torch.float)
#     deg_inv_sqrt = torch.pow(deg, -0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # handle division by zero
#     # Calculate normalized edge weights: D^{-1/2}[i] * D^{-1/2}[j] for edge (i,j):contentReference[oaicite:33]{index=33}
#     edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#     # Update the data object with normalized edges
#     data.edge_index = edge_index
#     data.edge_weight = edge_weight



#     if log_dir is not None and not os.path.isdir(log_dir):
#         os.mkdir(log_dir)

#         # Prepare sample weights and masking for loss computation
#     number_total_nodes = data.num_nodes
#     labels_text = [G.nodes[node_id]["text_label"] for node_id in node_list]  # original text labels
#     # Count nodes per label category:contentReference[oaicite:41]{index=41}
#     num_chromosome = labels_text.count("chromosome")
#     num_plasmid = labels_text.count("plasmid")
#     num_ambiguous = labels_text.count("ambiguous")
#     num_unlabeled = labels_text.count("unlabeled")
#     assert number_total_nodes == num_chromosome + num_plasmid + num_ambiguous + num_unlabeled

#     print("Chromosome contigs:", num_chromosome,
#           "Plasmid contigs:", num_plasmid,
#           "Ambiguous contigs:", num_ambiguous,
#           "Unlabeled contigs:", num_unlabeled)

#     # Class weights (unlabeled=0, chromosome=1, plasmid=ambiguous=plasmid_ambiguous_weight):contentReference[oaicite:42]{index=42}
#     chromosome_weight = 1.0
#     plasmid_weight = float(parameters["plasmid_ambiguous_weight"])
#     ambiguous_weight = plasmid_weight

#     # Build the mask/weight vector per node (length = num_nodes):contentReference[oaicite:43]{index=43}
#     masks = []
#     for node_id in node_list:
#         label = G.nodes[node_id]["text_label"]
#         if label == "unlabeled":
#             masks.append(0.0)
#         elif label == "chromosome":
#             masks.append(chromosome_weight)
#         elif label == "plasmid":
#             masks.append(plasmid_weight)
#         elif label == "ambiguous":
#             masks.append(ambiguous_weight)

#     # Convert masks to numpy or torch for splitting
#     masks = np.array(masks, dtype=float)
#     # Set random seed for reproducibility (same procedure as original):contentReference[oaicite:44]{index=44}:contentReference[oaicite:45]{index=45}
#     seed_number = parameters["random_seed"]
#     os.environ['PYTHONHASHSEED'] = str(seed_number)
#     random.seed(seed_number)
#     np.random.seed(seed_number + 1)
#     torch.manual_seed(seed_number + 2)

#     # Split masks into training and validation by random 80/20 assignment:contentReference[oaicite:46]{index=46}
#     masks_train = masks.copy()
#     masks_validate = masks.copy()
#     for i in range(len(masks)):
#         if random.random() > 0.8:
#             # With 20% probability, mark this node as *validation* (remove from training)
#             masks_train[i] = 0.0
#         else:
#             # With 80% probability, mark as training (remove from validation)
#             masks_validate[i] = 0.0

#     # Convert masks to torch tensors for use in loss calculations
#     masks_train = torch.tensor(masks_train, dtype=torch.float32)
#     masks_validate = torch.tensor(masks_validate, dtype=torch.float32)


#     model = architecture.PlasGraph(parameters=parameters)
#     # In PyTorch, regularization like L2 (weight decay) is not defined inside the model class like in TensorFlow/Keras. Instead, it's handled externally during optimizer creation.
#     optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['l2_reg'])


#     # Move model and data to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     data = data.to(device)
#     masks_train = masks_train.to(device)
#     masks_validate = masks_validate.to(device)

#     labels = data.y.float()

#     # Binary Cross Entropy for two-label multi-class (sigmoid outputs)
#     loss_fn = torch.nn.BCELoss(reduction='none')

#     best_val_loss = float('inf')
#     patience_counter = 0
#     max_patience = parameters["early_stopping_patience"]

#     for epoch in range(parameters["epochs"]):
#         model.train()
#         optimizer.zero_grad()

#         out = model(data.x, data.edge_index)
#         out = torch.clamp(out, min=1e-7, max=1.0 - 1e-7)  # avoid log(0)

#         # Compute per-node loss, then apply training mask as weights
#         per_node_loss = loss_fn(out, labels)
#         weighted_loss = (per_node_loss.sum(dim=-1) * masks_train).sum() / masks_train.sum()

#         weighted_loss.backward()
#         optimizer.step()

#         # Validation
#         model.eval()
#         with torch.no_grad():
#             out_val = model(data.x, data.edge_index)
#             out_val = torch.clamp(out_val, min=1e-7, max=1.0 - 1e-7)
#             val_loss = loss_fn(out_val, labels)
#             val_loss = (val_loss.sum(dim=-1) * masks_validate).sum() / masks_validate.sum()

#         print(f"Epoch {epoch:03d} | Train Loss: {weighted_loss:.4f} | Val Loss: {val_loss:.4f}")

#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             best_model_state = model.state_dict()
#         else:
#             patience_counter += 1
#             if patience_counter > max_patience:
#                 print("Early stopping triggered.")
#                 break

#     # Restore best model weights
#     model.load_state_dict(best_model_state)

#     # parameters["n_input_features"] = data.x.shape[1]

#     #     # Initialize the PyTorch Geometric model
#     # model = architecture.PlasGraph(parameters=parameters)
#     # # Use Adam optimizer with the configured learning rate:contentReference[oaicite:56]{index=56}
#     # optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["l2_reg"])


#     # # Select loss function: Binary Crossentropy (with logits) or MSE (or Squared Hinge):contentReference[oaicite:57]{index=57}
#     # loss_func_name = parameters["loss_function"]
#     # if loss_func_name == "crossentropy":
#     #     # BCEWithLogitsLoss expects raw logits and applies sigmoid internally
#     #     loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
#     # elif loss_func_name == "mse":
#     #     loss_fn = torch.nn.MSELoss(reduction='none')
#     # elif loss_func_name == "squaredhinge":
#     #     # Squared Hinge: implement as custom element-wise loss: (max(0, 1 - y_true * y_pred))^2
#     #     loss_fn = None  # (we will compute manually in training loop for hinge)
#     # else:
#     #     raise ValueError(f"Bad loss function {loss_func_name}")


#     #     # Determine early stopping patience from config
#     # patience = parameters["early_stopping_patience"]
#     # best_val_loss = float('inf')
#     # best_state_dict = None
#     # epochs_no_improve = 0

#     # # Lists to store loss history for logging
#     # train_loss_history = []
#     # val_loss_history = []

#     # # Training loop
#     # for epoch in range(int(parameters["epochs"])):
#     #     model.train()  # set model to training mode (enables dropout, etc.)
#     #     optimizer.zero_grad()

#     #     # Forward pass on all nodes. The model should use edge_index and edge_weight internally.
#     #     # Note: `PlasGraph.forward` likely accepts (x, edge_index, edge_weight).
#     #     out = model(data.x, data.edge_index, edge_weight=data.edge_weight)
#     #     # out shape: [num_nodes, n_labels]; data.y shape: [num_nodes, n_labels].

#     #     # Compute loss for train nodes:
#     #     if loss_func_name == "squaredhinge":
#     #         # For hinge, ensure targets are -1 or 1 (they are, from data.y if hinge) and compute (max(0, 1 - y * out))^2
#     #         y_true = data.y  # (torch tensor)
#     #         # If model's output activation is not bounded, we use raw output directly
#     #         # Compute element-wise squared hinge loss:
#     #         margin = 1.0 - y_true * out  # y_true in {-1,1}, out in (-inf, inf)
#     #         margin_clipped = torch.clamp(margin, min=0.0)
#     #         loss_matrix = margin_clipped**2  # same shape as out
#     #     else:
#     #         # For BCE or MSE, use the loss function (which returns element-wise loss, shape = [num_nodes, n_labels])
#     #         loss_matrix = loss_fn(out, data.y)
#     #     # Multiply element-wise by masks_train to zero-out loss for non-train nodes:contentReference[oaicite:65]{index=65}
#     #     loss_matrix = loss_matrix * masks_train.view(-1, 1)  # broadcast mask to [num_nodes, n_labels]
#     #     # Sum over all nodes and both labels to get scalar loss
#     #     loss = loss_matrix.sum()
#     #     # Backpropagation and weight update
#     #     loss.backward()
#     #     optimizer.step()

#     #     # Compute training loss and validation loss for monitoring
#     #     train_loss = (loss_matrix.sum()).item()  # this is the sum of losses on train nodes
#     #     model.eval()  # switch to eval mode for validation
#     #     with torch.no_grad():
#     #         val_out = model(data.x, data.edge_index, edge_weight=data.edge_weight)
#     #         if loss_func_name == "squaredhinge":
#     #             y_true = data.y
#     #             margin = 1.0 - y_true * val_out
#     #             margin_clipped = torch.clamp(margin, min=0.0)
#     #             val_loss_matrix = margin_clipped**2
#     #         else:
#     #             val_loss_matrix = loss_fn(val_out, data.y)
#     #         val_loss_matrix = val_loss_matrix * masks_validate.view(-1, 1)
#     #         val_loss_sum = val_loss_matrix.sum().item()
#     #     # Note: We use sum of losses (not averaged) for consistency with original (reduction="sum"):contentReference[oaicite:66]{index=66}.
#     #     # We can normalize or leave as is; early stopping will work either way as it's relative.

#     #     # For logging, you may compute average loss per node as well, but not needed strictly.

#     #     # Store losses
#     #     train_loss_history.append(train_loss)
#     #     val_loss_history.append(val_loss_sum)

#     #     # Print progress (epoch index starting at 1 for display)
#     #     print(f"Epoch {epoch+1}/{parameters['epochs']} - loss: {train_loss:.4f} - val_loss: {val_loss_sum:.4f}")

#     #     # Check early stopping criteria
#     #     if val_loss_sum < best_val_loss:
#     #         best_val_loss = val_loss_sum
#     #         best_state_dict = model.state_dict()  # save best model weights
#     #         epochs_no_improve = 0
#     #     else:
#     #         epochs_no_improve += 1
#     #         if epochs_no_improve >= patience:
#     #             print(f"Early stopping at epoch {epoch+1}, no improvement in val_loss for {patience} epochs.")
#     #             # Restore best weights and break
#     #             if best_state_dict is not None:
#     #                 model.load_state_dict(best_state_dict)
#     #             break


#     #     # After training, log the loss history to CSV files if requested
#     # if log_dir is not None:
#     #     os.makedirs(log_dir, exist_ok=True)
#     #     val_log_path = os.path.join(log_dir, "val_loss.csv")
#     #     train_log_path = os.path.join(log_dir, "train_loss.csv")
#     #     with open(val_log_path, "wt") as f:
#     #         for i, val in enumerate(val_loss_history):
#     #             print(f"{i},{val}", file=f)
#     #     with open(train_log_path, "wt") as f:
#     #         for i, tr in enumerate(train_loss_history):
#     #             print(f"{i},{tr}", file=f)


#     #     # Save the trained model and the config file:contentReference[oaicite:101]{index=101}
#     # model_path = os.path.join(model_output_dir, "model.pt")
#     # torch.save(model.state_dict(), model_path)
#     # # Save the configuration parameters to YAML in the output directory
#     # parameters.write_yaml(os.path.join(model_output_dir, config.DEFAULT_FILENAME))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description=inspect.getdoc(main))
#     parser.add_argument("config_file", help="YAML configuration file")
#     parser.add_argument("train_file_list", help="csv file with a list of training samples")
#     parser.add_argument("file_prefix", help="common prefix for all filenames in train_file_list, e.g. ../data/")
#     parser.add_argument("model_output_dir", help="name of the output folder for the trained model")
#     parser.add_argument("-g", dest="gml_file", help="optional output file for the graph in GML format", default=None)
#     parser.add_argument("-l", dest="log_dir", help="optional output folder for training logs", default=None)
#     args = parser.parse_args()
#     main(**vars(args))

