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
import old_code.create_graph5 as create_graph5
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

import torch.nn.functional as F
from architecture import PlasGraphModel  # assumes you've renamed your PyG model

import numpy as np
from sklearn.metrics import f1_score


def main(config_file: 'YAML configuration file',
         train_file_list: 'csv file with a list of training samples',
         file_prefix: 'common prefix to be used for all filenames listed in train_file_list, e.g. ../data/',
         model_output_dir: 'name of an output folder with the trained model',
         gml_file: 'optional output file for the graph in GML format' = None,
         log_dir: 'optional output folder for various training logs' = None):

    # Load parameters from YAML config
    parameters = config.config(config_file)

    dataset = create_graph5.PlasGraphDataset(
        root=model_output_dir,
        train_file_list=train_file_list,
        file_prefix=file_prefix,
        parameters=parameters,
        gml_file=gml_file
    )

    data = dataset[0]



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
    current_nx_graph = dataset.nx_graph 
    current_node_order = dataset.node_order  
    labels = [current_nx_graph.nodes[node_id]["text_label"] for node_id in current_node_order] # original labels
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
    for node_id in current_node_order:
        label = current_nx_graph.nodes[node_id]["text_label"]
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

    labels = torch.tensor([
        [1, 0] if current_nx_graph.nodes[node_id]["text_label"] == "plasmid" else
        [0, 1] if current_nx_graph.nodes[node_id]["text_label"] == "chromosome" else
        [0, 0]
        for node_id in current_node_order
    ], dtype=torch.float32).to(device)



    print("🔍 Label breakdown (new format):")
    num_plasmid_actual = (labels[:, 0] == 1).sum().item()
    num_chromosome_actual = (labels[:, 1] == 1).sum().item()
    num_other_actual = ((labels[:, 0] == 0) & (labels[:, 1] == 0)).sum().item()
    print(f"  Plasmid ([1,0]):      {num_plasmid_actual}")
    print(f"  Chromosome ([0,1]):   {num_chromosome_actual}")
    print(f"  Other ([0,0]):        {num_other_actual}")
    print(f"  Total label rows:     {len(labels)}")




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
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    elif parameters["loss_function"] == "mse":
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss function: {parameters['loss_function']}")

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = 0

    metric_accuracy = MulticlassAccuracy(num_classes=2).to(device)
    metric_precision = MulticlassPrecision(num_classes=2, average='macro').to(device)
    metric_recall = MulticlassRecall(num_classes=2, average='macro').to(device)
    metric_f1 = MulticlassF1Score(num_classes=2, average='macro').to(device)
    metric_auroc = MulticlassAUROC(num_classes=2).to(device)



    for epoch in range(parameters["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        
        # Mask out [0, 0] labels (ambiguous/unlabeled) like in TF
        valid_label_mask = labels.sum(dim=1) != 0
        masked_outputs = outputs[valid_label_mask]
        masked_labels = labels[valid_label_mask]
        masked_weights = masks_train[valid_label_mask]

        loss_per_node = criterion(masked_outputs, masked_labels)  # shape: [n_valid_nodes, 2]
        train_loss = (loss_per_node.sum(dim=1) * masked_weights).sum() / masked_weights.sum()


        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(data)

            valid_label_mask = labels.sum(dim=1) != 0
            masked_outputs = val_outputs[valid_label_mask]
            masked_labels = labels[valid_label_mask]
            masked_weights = masks_validate[valid_label_mask]

            loss_per_node = criterion(masked_outputs, masked_labels)
            val_loss = (loss_per_node.sum(dim=1) * masked_weights).sum() / masked_weights.sum()



            probs = torch.softmax(val_outputs, dim=1)[:, 1]  

            labels_val_masked = labels[masks_validate.bool()]
            probs_val_masked = probs[masks_validate.bool()]

            valid_eval_mask = labels_val_masked.sum(dim=1) != 0

            labels_eval = labels_val_masked[valid_eval_mask]
            probs_eval = probs_val_masked[valid_eval_mask]

            y_true = labels_eval.argmax(dim=1).cpu().numpy()  
            y_probs = probs_eval.cpu().numpy()                



            # Find best threshold for F1
            def find_best_threshold(y_true, y_probs):
                thresholds = np.linspace(0.1, 0.9, 81)
                best_thresh, best_f1 = 0.5, 0
                for t in thresholds:
                    y_pred = (y_probs >= t).astype(int)
                    f1 = f1_score(y_true, y_pred)
                    if f1 > best_f1:
                        best_f1, best_thresh = f1, t
                return best_thresh, best_f1

            best_thresh, best_f1 = find_best_threshold(y_true, y_probs)

            print("🧪 y_probs stats:", y_probs.min(), y_probs.max(), y_probs.mean())
            print("🧪 y_pred counts:", (y_probs >= best_thresh).sum(), "out of", len(y_probs))

            y_pred = (y_probs >= best_thresh).astype(int)

            # Convert predictions back to torch for metrics
            y_pred_tensor = torch.tensor(y_pred).to(device)
            y_true_tensor = torch.tensor(y_true).to(device)

            # Reset and update metrics
            metric_accuracy.reset()
            metric_precision.reset()
            metric_recall.reset()
            metric_f1.reset()
            metric_auroc.reset()

            metric_accuracy.update(y_pred_tensor, y_true_tensor)
            metric_precision.update(y_pred_tensor, y_true_tensor)
            metric_recall.update(y_pred_tensor, y_true_tensor)
            metric_f1.update(y_pred_tensor, y_true_tensor)
            probs_stacked = torch.stack([1 - probs_val_masked, probs_val_masked], dim=1)
            metric_auroc.update(probs_stacked, labels_val_masked.argmax(dim=1))



        # Logging
        print(f"Epoch {epoch:03} | "
              f"Train Loss: {train_loss.item():.4f} | "
              f"Val Loss: {val_loss.item():.4f} | "
              f"F1: {best_f1:.4f} @ threshold={best_thresh:.2f} | "
              f"Acc: {metric_accuracy.compute():.4f} | "
              f"Prec: {metric_precision.compute():.4f} | "
              f"Rec: {metric_recall.compute():.4f} | "
              f"AUROC: {metric_auroc.compute():.4f}")



        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(model_output_dir, "best_model.pt"))
            # Save config YAML
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
