#!/usr/bin/env python3
# coding: utf-8

import inspect
import pandas as pd
import networkx as nx
import numpy as np
import os
import random
import argparse

import torch

import create_graph as create_graph
import config
import thresholds

from architecture import GCNModel  

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score 



def main(config_file: 'YAML configuration file',
         train_file_list: 'csv file with a list of training samples',
         file_prefix: 'common prefix to be used for all filenames listed in train_file_list, e.g. ../data/',
         model_output_dir: 'name of an output folder with the trained model',
         gml_file: 'optional output file for the graph in GML format' = None,
         log_dir: 'optional output folder for various training logs' = None):

    parameters = config.config(config_file)

    all_graphs = create_graph.Dataset_Pytorch(
        root = model_output_dir,
        file_prefix = file_prefix,
        train_file_list = train_file_list,
        parameters = parameters
    )

    data = all_graphs[0]

    G = all_graphs.G
    node_list = all_graphs.node_list


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

    os.makedirs(model_output_dir, exist_ok=True)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    number_total_nodes = data.num_nodes
    labels = [G.nodes[node_id]["text_label"] for node_id in node_list]  
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

    chromosome_weight = 1.0
    plasmid_weight = float(parameters["plasmid_ambiguous_weight"])
    ambiguous_weight = plasmid_weight

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

    seed_number = parameters["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number + 1)
    torch.manual_seed(seed_number + 2)

    masks_train = masks.copy()
    masks_validate = masks.copy()
    for i in range(len(masks)):
        if random.random() > 0.8:
            masks_train[i] = 0.0
        else:
            masks_validate[i] = 0.0

    print(masks_train[0:20])
    print(masks_validate[0:20])

    masks_train = torch.tensor(masks_train, dtype=torch.float32)
    masks_validate = torch.tensor(masks_validate, dtype=torch.float32)

    print(len(masks_train))
    print(len(masks_validate))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCNModel(parameters).to(device)
    data = data.to(device)



    print("🔍 Label breakdown (new format):")
    num_plasmid_actual = ((data.y[:, 0] == 1) & (data.y[:, 1] == 0)).sum().item()
    num_chromosome_actual = ((data.y[:, 0] == 0) & (data.y[:, 1] == 1)).sum().item()
    num_ambiguous_actual = ((data.y[:, 0] == 1) & (data.y[:, 1] == 1)).sum().item() 
    num_unlabeled_actual = ((data.y[:, 0] == 0) & (data.y[:, 1] == 0)).sum().item() 
    
    print(f"  Plasmid ([1,0]):      {num_plasmid_actual}")
    print(f"  Chromosome ([0,1]):   {num_chromosome_actual}")
    print(f"  Ambiguous ([1,1]):    {num_ambiguous_actual}")
    print(f"  Unlabeled ([0,0]):    {num_unlabeled_actual}")
    print(f"  Total label rows:     {len(data.y)}")



    masks_train = masks_train.to(device)
    masks_validate = masks_validate.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters['learning_rate'],
        weight_decay=parameters['l2_reg']
    )

    if parameters["loss_function"] == "crossentropy":
        criterion = torch.nn.BCELoss(reduction='none')

    elif parameters["loss_function"] == "mse":
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss function: {parameters['loss_function']}")

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

        print(f"Epoch {epoch:03} | "
              f"Train Loss: {train_loss.item():.4f} | "
              f"Val Loss: {val_loss.item():.4f}")
        
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

    
    if os.path.exists(os.path.join(model_output_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(model_output_dir, "best_model.pt")))
        model.eval()

    if parameters['set_thresholds']:
        thresholds.set_thresholds(model, data, masks_validate, parameters, log_dir) 
        parameters.write_yaml(os.path.join(model_output_dir, config.DEFAULT_FILENAME))
        print("Optimal thresholds saved to config file and logs.")



    print("\n--- Final Evaluation Metrics on Validation Set ---")

    model.eval()
    with torch.no_grad():
        final_val_outputs = model(data) 

        probs_plasmid_final = final_val_outputs[:, 0]
        probs_chromosome_final = final_val_outputs[:, 1]

        # --- Process Plasmid ---
        labels_plasmid_val_masked = data.y[masks_validate.bool(), 0]
        probs_plasmid_val_masked = probs_plasmid_final[masks_validate.bool()] 

        valid_eval_mask_plasmid = labels_plasmid_val_masked != -1 

        y_true_plasmid = labels_plasmid_val_masked[valid_eval_mask_plasmid].cpu().numpy()
        y_probs_plasmid = probs_plasmid_val_masked[valid_eval_mask_plasmid].cpu().numpy()
        sample_weight_plasmid = masks_validate[masks_validate.bool()][valid_eval_mask_plasmid].cpu().numpy()

        final_best_thresh_plasmid = parameters['plasmid_threshold']

        y_pred_plasmid = (y_probs_plasmid >= final_best_thresh_plasmid).astype(int)

        acc_plasmid = accuracy_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid)
        prec_plasmid = precision_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
        rec_plasmid = recall_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
        f1_plasmid = f1_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0) # Calculate F1 directly
        auroc_plasmid = roc_auc_score(y_true_plasmid, y_probs_plasmid, sample_weight=sample_weight_plasmid)

        # --- Process Chromosome ---
        labels_chromosome_val_masked = data.y[masks_validate.bool(), 1]
        probs_chromosome_val_masked = probs_chromosome_final[masks_validate.bool()] # Use final_val_outputs

        valid_eval_mask_chromosome = labels_chromosome_val_masked != -1

        y_true_chromosome = labels_chromosome_val_masked[valid_eval_mask_chromosome].cpu().numpy()
        y_probs_chromosome = probs_chromosome_val_masked[valid_eval_mask_chromosome].cpu().numpy()
        sample_weight_chromosome = masks_validate[masks_validate.bool()][valid_eval_mask_chromosome].cpu().numpy()

        final_best_thresh_chromosome = parameters['chromosome_threshold']

        y_pred_chromosome = (y_probs_chromosome >= final_best_thresh_chromosome).astype(int)

        acc_chromosome = accuracy_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome)
        prec_chromosome = precision_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
        rec_chromosome = recall_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
        f1_chromosome = f1_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0) # Calculate F1 directly
        auroc_chromosome = roc_auc_score(y_true_chromosome, y_probs_chromosome, sample_weight=sample_weight_chromosome)

    print(f"Final PLASMID Metrics | F1: {f1_plasmid:.4f} @ Thresh: {final_best_thresh_plasmid:.2f} | Acc: {acc_plasmid:.4f} | Prec: {prec_plasmid:.4f} | Rec: {rec_plasmid:.4f} | AUROC: {auroc_plasmid:.4f}")
    print(f"Final CHROMOSOME Metrics | F1: {f1_chromosome:.4f} @ Thresh: {final_best_thresh_chromosome:.2f} | Acc: {acc_chromosome:.4f} | Prec: {prec_chromosome:.4f} | Rec: {rec_chromosome:.4f} | AUROC: {auroc_chromosome:.4f}")



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
