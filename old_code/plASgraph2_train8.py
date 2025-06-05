#!/usr/bin/env python3
# coding: utf-8

import itertools
import inspect
import gzip
import pandas as pd
import networkx as nx
import numpy as np
import os
import random
import argparse
import shutil

import torch
from torch_geometric.utils import add_self_loops, degree

import architecture
import old_code.create_graph8 as create_graph8
import config
import thresholds

import pickle

# from torchmetrics.classification import (
#     BinaryAccuracy,
#     BinaryPrecision,
#     BinaryRecall,
#     BinaryAUROC,
#     BinaryF1Score
# )

import sys

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryF1Score

import torch.nn.functional as F
from architecture import PlasGraphModel  # assumes you've renamed your PyG model

import numpy as np
from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score # ADD THIS LINE



def main(config_file: 'YAML configuration file',
         train_file_list: 'csv file with a list of training samples',
         file_prefix: 'common prefix to be used for all filenames listed in train_file_list, e.g. ../data/',
         model_output_dir: 'name of an output folder with the trained model',
         gml_file: 'optional output file for the graph in GML format' = None,
         log_dir: 'optional output folder for various training logs' = None):

    # Load parameters from YAML config
    parameters = config.config(config_file)

    # Cache path to save/load graph objects (PyG)
    cache_path = os.path.join(model_output_dir, "all_graphs.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            G, all_graphs, node_list = pickle.load(f)
    else:
        # Read graph from GFA files
        G = create_graph8.read_graph_set(file_prefix, train_file_list,
                                        parameters['minimum_contig_length'])
        node_list = list(G.nodes)
        if gml_file is not None:
            nx.write_gml(G, path=gml_file)
        # Convert NetworkX graph to PyG Data objects
        all_graphs = create_graph8.Networkx_to_PyG(model_output_dir, G, node_list, parameters)
        # Create directory for cache if needed
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((G, all_graphs, node_list), f)

    data = all_graphs[0]


    def diagnose_graph(data, parameters):
        print("=" * 60)
        print("🔍 PyTorch Geometric Graph Diagnostic")
        print("=" * 60)
        print()

        # Types
        print(f"type(data): {type(data)}")  # Should be torch_geometric.data.data.Data
        print(f"type(data.x): {type(data.x)}")  # Node features → torch.Tensor
        print(f"type(data.y): {type(data.y)}")  # Labels → torch.Tensor
        print(f"type(data.edge_index): {type(data.edge_index)}")  # Edge list

        print()

        # Node features
        x = data.x
        print(f"🧬 Feature matrix shape: {x.shape}")
        print(f"📋 Features used: {parameters['features']}")
        print(f"🧾 Sample features (first 2 nodes):\n{x[:2]}")

        # Labels
        y = data.y
        print(f"\n🏷️ Label matrix shape: {y.shape}")
        print(f"🔢 Label sample (first 5):\n{y[:5]}")

        # Label distribution
        y_np = y.detach().cpu().numpy()
        unique_labels, counts = np.unique(y_np, axis=0, return_counts=True)
        print("\n📊 Label counts:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label.tolist()} → {count} nodes")

        # Edges
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        num_nodes = data.num_nodes
        print(f"\n🔗 Edge index shape: {edge_index.shape}")
        print(f"  → Number of edges: {num_edges}")

        # Density
        density = num_edges / (num_nodes ** 2)
        print(f"  → Density: {density:.6f}")

        # Isolated nodes
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        num_isolated = torch.sum(degrees == 0).item()
        print(f"🧍 Isolated nodes: {num_isolated}")

        # Available attributes
        print(f"\n📦 Available attributes: {list(data.keys())}")
    diagnose_graph(data, parameters)


    # Manually normalize adjacency (add self-loops and compute D^-1/2)
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    row, col = edge_index
    deg = degree(row, data.num_nodes, dtype=torch.float)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    data.edge_index = edge_index
    data.edge_weight = edge_weight

    def diagnose_after_gcnfilter(data, edge_weight):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_edges = edge_index.size(1)
        density = num_edges / (num_nodes ** 2)

        # Count self-loops
        self_loops = (edge_index[0] == edge_index[1]).sum().item()

        print("\n🔬 AFTER GCNFilter")
        print(f"  ➤ Shape: ({num_nodes}, {num_nodes}) (implied)")
        print(f"  ➤ Num edges (with self-loops): {num_edges}")
        print(f"  ➤ Self-loops: {self_loops}")
        print(f"  ➤ Density: {density:.6f}")

        # Sample self-loop weights
        print("\n🧾 Sample of diagonal weights (edge_weight[i] where i==j):")
        rows_to_show = 5
        shown = 0
        for idx in range(edge_index.size(1)):
            i, j = edge_index[0, idx], edge_index[1, idx]
            if i == j and shown < rows_to_show:
                print(f"  Node {i.item():2} → edge_weight = {edge_weight[idx]:.4f}")
                shown += 1
    diagnose_after_gcnfilter(data, edge_weight)


    # Ensure output directories exist
    os.makedirs(model_output_dir, exist_ok=True)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    # Prepare sample weights (mask) per node
    number_total_nodes = data.num_nodes
    labels = [G.nodes[node_id]["text_label"] for node_id in node_list]  # original labels
    num_chromosome = labels.count("chromosome")
    num_plasmid = labels.count("plasmid")
    num_ambiguous = labels.count("ambiguous")
    num_unlabeled = labels.count("unlabeled")
    assert number_total_nodes == num_chromosome + num_plasmid + num_ambiguous + num_unlabeled

    print(labels[:50])


    print("Chromosome contigs:", num_chromosome,
          "Plasmid contigs:", num_plasmid,
          "Ambiguous contigs:", num_ambiguous,
          "Unlabeled contigs:", num_unlabeled)

    # Class weights: unlabeled=0, chromosome=1, plasmid/ambiguous=plasmid_ambiguous_weight
    chromosome_weight = 1.0
    plasmid_weight = float(parameters["plasmid_ambiguous_weight"])
    ambiguous_weight = plasmid_weight

    # Build mask/weight vector per node
    masks = []
    for node_id in node_list:
        label = G.nodes[node_id]["text_label"]
        if label == "unlabeled":
            masks.append(0.0)
        elif label == "chromosome":
            masks.append(chromosome_weight)
        elif label == "plasmid":
            masks.append(plasmid_weight)
        elif label == "ambiguous":
            masks.append(ambiguous_weight)
    masks = np.array(masks, dtype=float)

    print("Final masks list:", masks)

    # Reproducibility: set seeds (same as original)
    seed_number = parameters["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number + 1)
    torch.manual_seed(seed_number + 2)

    # Split masks into training (80%) and validation (20%) randomly
    masks_train = masks.copy()
    masks_validate = masks.copy()
    for i in range(len(masks)):
        if random.random() > 0.8:
            masks_train[i] = 0.0
        else:
            masks_validate[i] = 0.0

    print(masks_train[0:20])
    print(masks_validate[0:20])

    # Convert masks to torch tensors for loss weighting
    masks_train = torch.tensor(masks_train, dtype=torch.float32)
    masks_validate = torch.tensor(masks_validate, dtype=torch.float32)

    print(len(masks_train))
    print(len(masks_validate))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlasGraphModel(parameters).to(device)
    data = data.to(device)



    print("🔍 Label breakdown (new format):")
    # Pure plasmid labels (where only the first element is 1, and second is 0)
    num_plasmid_actual = ((data.y[:, 0] == 1) & (data.y[:, 1] == 0)).sum().item()
    # Pure chromosome labels (where only the second element is 1, and first is 0)
    num_chromosome_actual = ((data.y[:, 0] == 0) & (data.y[:, 1] == 1)).sum().item()
    
    # Ambiguous labels [1,1]
    num_ambiguous_actual = ((data.y[:, 0] == 1) & (data.y[:, 1] == 1)).sum().item() 
    # Unlabeled labels [0,0]
    num_unlabeled_actual = ((data.y[:, 0] == 0) & (data.y[:, 1] == 0)).sum().item() 
    
    print(f"  Plasmid ([1,0]):      {num_plasmid_actual}")
    print(f"  Chromosome ([0,1]):   {num_chromosome_actual}")
    print(f"  Ambiguous ([1,1]):    {num_ambiguous_actual}")
    print(f"  Unlabeled ([0,0]):    {num_unlabeled_actual}")
    print(f"  Total label rows:     {len(data.y)}")



    masks_train = masks_train.to(device)
    masks_validate = masks_validate.to(device)

    # Instantiate optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters['learning_rate'],
        weight_decay=parameters['l2_reg']
    )

    # Select loss function
    if parameters["loss_function"] == "crossentropy":
        criterion = torch.nn.BCELoss(reduction='none')

    elif parameters["loss_function"] == "mse":
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss function: {parameters['loss_function']}")

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = 0


    for epoch in range(parameters["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        
        valid_label_mask = data.y.sum(dim=1) != 0
        masked_outputs = outputs[valid_label_mask]
        masked_labels = data.y[valid_label_mask]
        masked_weights = masks_train[valid_label_mask]

        loss_per_node = criterion(masked_outputs, masked_labels)
        train_loss = (loss_per_node.sum(dim=1) * masked_weights).sum() / masked_weights.sum()


        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(data)

            valid_label_mask = data.y.sum(dim=1) != 0
            masked_outputs = val_outputs[valid_label_mask]
            masked_labels = data.y[valid_label_mask]
            masked_weights = masks_validate[valid_label_mask]

            loss_per_node = criterion(masked_outputs, masked_labels)
            val_loss = (loss_per_node.sum(dim=1) * masked_weights).sum() / masked_weights.sum()

            probs_plasmid = val_outputs[:, 0]
            probs_chromosome = val_outputs[:, 1]     


            def find_best_threshold_binary(y_true_binary, y_probs_single_class, sample_weights_binary):
                thresholds_arr = np.linspace(0.1, 0.9, 81)
                best_thresh_val, best_f1_val = 0.5, 0
                for t_val in thresholds_arr:
                    y_pred_binary = (y_probs_single_class >= t_val).astype(int)
                    f1_val = f1_score(y_true_binary, y_pred_binary, sample_weight=sample_weights_binary)
                    if f1_val > best_f1_val:
                        best_f1_val, best_thresh_val = f1_val, t_val
                return best_thresh_val, best_f1_val

            # --- Process Plasmid ---
            labels_plasmid_val_masked = data.y[masks_validate.bool(), 0]
            probs_plasmid_val_masked = probs_plasmid[masks_validate.bool()]

            valid_eval_mask_plasmid = labels_plasmid_val_masked != -1

            y_true_plasmid = labels_plasmid_val_masked[valid_eval_mask_plasmid].cpu().numpy()
            y_probs_plasmid = probs_plasmid_val_masked[valid_eval_mask_plasmid].cpu().numpy()
            sample_weight_plasmid = masks_validate[masks_validate.bool()][valid_eval_mask_plasmid].cpu().numpy()

            best_thresh_plasmid, best_f1_plasmid = find_best_threshold_binary(y_true_plasmid, y_probs_plasmid, sample_weight_plasmid)

            y_pred_plasmid = (y_probs_plasmid >= best_thresh_plasmid).astype(int)

            # Calculate all metrics for Plasmid
            acc_plasmid = accuracy_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid)
            prec_plasmid = precision_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
            rec_plasmid = recall_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
            auroc_plasmid = roc_auc_score(y_true_plasmid, y_probs_plasmid, sample_weight=sample_weight_plasmid)


            # --- Process Chromosome ---
            labels_chromosome_val_masked = data.y[masks_validate.bool(), 1]
            probs_chromosome_val_masked = probs_chromosome[masks_validate.bool()]

            valid_eval_mask_chromosome = labels_chromosome_val_masked != -1

            y_true_chromosome = labels_chromosome_val_masked[valid_eval_mask_chromosome].cpu().numpy()
            y_probs_chromosome = probs_chromosome_val_masked[valid_eval_mask_chromosome].cpu().numpy()
            sample_weight_chromosome = masks_validate[masks_validate.bool()][valid_eval_mask_chromosome].cpu().numpy()

            best_thresh_chromosome, best_f1_chromosome = find_best_threshold_binary(y_true_chromosome, y_probs_chromosome, sample_weight_chromosome)

            y_pred_chromosome = (y_probs_chromosome >= best_thresh_chromosome).astype(int)

            # Calculate all metrics for Chromosome
            acc_chromosome = accuracy_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome)
            prec_chromosome = precision_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
            rec_chromosome = recall_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
            auroc_chromosome = roc_auc_score(y_true_chromosome, y_probs_chromosome, sample_weight=sample_weight_chromosome)

        print(f"Epoch {epoch:03} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"PLASMID F1: {best_f1_plasmid:.4f} @ {best_thresh_plasmid:.2f} | Acc: {acc_plasmid:.4f} | Prec: {prec_plasmid:.4f} | Rec: {rec_plasmid:.4f} | AUROC: {auroc_plasmid:.4f} | "
                f"CHROMOSOME F1: {best_f1_chromosome:.4f} @ {best_thresh_chromosome:.2f} | Acc: {acc_chromosome:.4f} | Prec: {prec_chromosome:.4f} | Rec: {rec_chromosome:.4f} | AUROC: {auroc_chromosome:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(model_output_dir, "best_model.pt"))
            parameters.write_yaml(os.path.join(model_output_dir, config.DEFAULT_FILENAME))
        else:
            patience += 1
            if patience >= parameters["early_stopping_patience"]:
                print("Early stopping triggered.")
                break





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=inspect.getdoc(main))
    parser.add_argument("config_file", help="YAML configuration file")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames in train_file_list, e.g. ../data/")
    parser.add_argument("model_output_dir", help="Name of the output folder for the trained model")
    parser.add_argument("-g", dest="gml_file", help="Optional output GML graph file", default=None)
    parser.add_argument("-l", dest="log_dir", help="Optional output folder for training logs", default=None)
    args = parser.parse_args()
    main(**vars(args))
